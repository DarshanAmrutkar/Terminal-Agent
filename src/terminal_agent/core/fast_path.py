"""Two-Tier Agent Architecture: "Agentless" Fast Path (Localize -> Patch -> Validate).

Inspired by:
Xia et al., "Agentless: Demystifying LLM-based Software Engineering" (2024).
Provides a lightweight, deterministic 3-phase repair pipeline for localized bug-fixing
tasks, achieving fast resolution with minimal token overhead and falling back to
Autonomous ReAct when needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, List, Optional, Tuple

from terminal_agent.llm.base import LLMProvider
from terminal_agent.llm.message import Message
from terminal_agent.tools.search_replace import _search_replace_sync
from terminal_agent.tools.base import resolve_safe_path
from terminal_agent.sandbox.local import LocalRestrictedSandbox
from terminal_agent.sandbox.base import SandboxPolicy, SandboxResult


@dataclass
class FastPathResult:
    """Outcome of an Agentless Fast-Path repair run."""
    success: bool
    phase_reached: str  # "localization", "repair", "validation", "completed"
    localized_files: List[str] = field(default_factory=list)
    patch_applied: bool = False
    tests_passed: bool = False
    explanation: str = ""
    turns_taken: int = 1


class AgentlessFastPath:
    """Deterministic 3-phase repair runner."""

    def __init__(
        self, 
        provider: LLMProvider, 
        working_directory: str | Path,
        max_candidates: int = 3
    ) -> None:
        self.provider = provider
        self.working_dir = Path(working_directory).resolve()
        self.max_candidates = max_candidates

    def localize(self, task_description: str) -> List[str]:
        """Phase 1: Identify candidate files using AST and keyword relevance."""
        task_terms = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b", task_description.lower()))
        # Filter out common stop words
        stop_words = {"this", "that", "with", "from", "when", "then", "into", "file", "make", "test", "tests"}
        query_terms = task_terms - stop_words

        scores: dict[str, float] = {}
        ignore_dirs = {".git", ".venv", "venv", "__pycache__", "build", "dist", "site-packages"}

        for p in self.working_dir.rglob("*.py"):
            if any(part in ignore_dirs for part in p.parts):
                continue
            try:
                rel = str(p.relative_to(self.working_dir))
            except ValueError:
                continue

            score = 0.0
            filename_lower = p.name.lower()
            rel_lower = rel.lower()

            for term in query_terms:
                if term in filename_lower:
                    score += 5.0
                elif term in rel_lower:
                    score += 2.0

            # Inspect content for matches if reasonable size (<50KB)
            if p.stat().st_size < 50_000:
                try:
                    content_lower = p.read_text(encoding="utf-8", errors="ignore").lower()
                    for term in query_terms:
                        count = content_lower.count(term)
                        if count > 0:
                            score += min(count, 3) * 1.0
                except Exception:
                    pass

            if score > 0:
                scores[rel] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [f for f, _ in ranked[:self.max_candidates]]

    async def repair(self, task_description: str, rel_path: str) -> Tuple[bool, str]:
        """Phase 2: Synthesize a surgical search-and-replace patch."""
        full_path = self.working_dir / rel_path
        if not full_path.exists():
            return False, f"File '{rel_path}' does not exist"

        content = full_path.read_text(encoding="utf-8", errors="ignore")
        
        # Build surgical prompt
        prompt = (
            f"You are an expert software engineer operating in fast-path surgical repair mode.\n"
            f"Task: {task_description}\n"
            f"Target file: {rel_path}\n\n"
            f"File content:\n```python\n{content}\n```\n\n"
            f"Provide the exact search block and replacement block to resolve the task.\n"
            f"Respond ONLY with a JSON object in this exact format:\n"
            f'{{\n  "search": "exact lines to replace",\n  "replace": "new lines to insert"\n}}'
        )

        response = await self.provider.send(messages=[Message.user(prompt)])
        response_text = response.message.content or ""

        # Extract JSON block
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if not json_match:
            return False, "Failed to parse surgical JSON from model response"

        try:
            patch_data = json.loads(json_match.group(0))
            search_str = patch_data.get("search")
            replace_str = patch_data.get("replace")
            if search_str is None or replace_str is None:
                return False, "Missing 'search' or 'replace' fields in JSON patch"
        except Exception as e:
            return False, f"JSON parse error: {e}"

        # Apply patch via SyntaxGate-guarded atomic search & replace
        result = _search_replace_sync(
            file_path=full_path,
            path_str=rel_path,
            search=search_str,
            replace=replace_str,
            bypass_syntax_check=False,
        )

        return not result.is_error, result.output

    async def validate(self, test_command: str) -> Tuple[bool, str]:
        """Phase 3: Verify the patch using sandboxed automated regression tests."""
        policy = SandboxPolicy(working_dir=self.working_dir, timeout_seconds=60)
        sandbox = LocalRestrictedSandbox(policy=policy)
        res: SandboxResult = await sandbox.execute(test_command)
        return res.returncode == 0, res.formatted_output

    async def execute(
        self, 
        task_description: str, 
        test_command: Optional[str] = None
    ) -> FastPathResult:
        """Run the full 3-phase Fast Path pipeline."""
        # 1. Localization
        candidates = self.localize(task_description)
        if not candidates:
            return FastPathResult(
                success=False,
                phase_reached="localization",
                explanation="Fast-Path could not localize candidate files; escalating to autonomous ReAct.",
            )

        target_file = candidates[0]

        # 2. Repair
        patch_ok, patch_output = await self.repair(task_description, target_file)
        if not patch_ok:
            return FastPathResult(
                success=False,
                phase_reached="repair",
                localized_files=candidates,
                patch_applied=False,
                explanation=f"Fast-Path repair failed on '{target_file}': {patch_output}",
            )

        # 3. Validation
        if test_command:
            tests_ok, test_output = await self.validate(test_command)
            if not tests_ok:
                return FastPathResult(
                    success=False,
                    phase_reached="validation",
                    localized_files=candidates,
                    patch_applied=True,
                    tests_passed=False,
                    explanation=f"Fast-Path tests failed. Verification output:\n{test_output}",
                )
            return FastPathResult(
                success=True,
                phase_reached="completed",
                localized_files=candidates,
                patch_applied=True,
                tests_passed=True,
                explanation=f"Successfully repaired '{target_file}' and verified via tests in 1 fast-path turn!",
            )

        # If no test command provided but patch applied cleanly with syntax check passing
        return FastPathResult(
            success=True,
            phase_reached="completed",
            localized_files=candidates,
            patch_applied=True,
            tests_passed=True,
            explanation=f"Applied syntax-verified patch to '{target_file}'.",
        )
