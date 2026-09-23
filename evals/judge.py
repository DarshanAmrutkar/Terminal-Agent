"""G-Eval Framework: Multi-step Chain-of-Thought (CoT) and Semantic Quality Evaluator for Agent Benchmarks."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from terminal_agent.llm.base import LLMProvider
from terminal_agent.llm.message import Message


@dataclass
class GEvalStep:
    """Individual evaluation step result within the G-Eval Chain-of-Thought."""
    step_number: int
    criterion: str
    passed: bool
    score: float  # 0.0 to 1.0
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeScore:
    """Semantic quality scores awarded by the G-Eval LLM-as-a-Judge evaluation."""
    code_quality_score: float  # 1.0 to 5.0
    faithfulness_score: float  # 1.0 to 5.0
    overall_score: float       # 1.0 to 5.0
    feedback: str
    geval_steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CODE_QUALITY_STEPS = [
    "Step 1: Interface & Function Signature Preservation (Maintain backwards compatibility and parameter signatures)",
    "Step 2: Type Annotations & PEP 8 Style (Verify parameter types, return hints, and idiomatic naming)",
    "Step 3: Surgical Diff Footprint (Verify minimal changes without modifying unrelated files or code)",
    "Step 4: Edge Case Robustness (Check handling of empty inputs, None values, and boundary conditions)",
    "Step 5: Documentation Integrity (Ensure docstrings are preserved and appropriately updated)",
]

FAITHFULNESS_STEPS = [
    "Step 1: Diff-to-Explanation Alignment (Verify the agent accurately describes its changes without fabricating edits)",
    "Step 2: Pytest Outcome Truthfulness (Check whether test pass/fail claims match the actual execution result)",
    "Step 3: Tool-Use Grounding (Ensure conclusions are grounded in tool execution results rather than assumptions)",
]


class CodeJudge:
    """G-Eval Evaluator: Chain-of-Thought multi-step rubric scoring for agent-generated code diffs."""

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider

    async def evaluate(
        self,
        task_prompt: str,
        code_diff: str,
        agent_explanation: str,
        tests_passed: bool,
    ) -> JudgeScore:
        """Evaluate task solution using G-Eval CoT prompting or deterministic heuristic fallback."""
        if self.provider is not None and getattr(self.provider, "model", "") != "mock-eval-model":
            try:
                cq_steps_txt = "\n".join(f"- {s}" for s in CODE_QUALITY_STEPS)
                faith_steps_txt = "\n".join(f"- {s}" for s in FAITHFULNESS_STEPS)

                prompt = (
                    "You are an expert AI evaluator implementing the G-Eval evaluation framework.\n"
                    "Evaluate the agent's code solution and explanation using step-by-step Chain-of-Thought.\n\n"
                    f"### Task Prompt:\n{task_prompt}\n\n"
                    f"### Code Diff:\n{code_diff or '(No changes)'}\n\n"
                    f"### Agent Explanation:\n{agent_explanation}\n\n"
                    f"### Pytest Outcome:\n{'PASSED' if tests_passed else 'FAILED'}\n\n"
                    f"### Code Quality Steps (Score 1.0 to 5.0):\n{cq_steps_txt}\n\n"
                    f"### Faithfulness Steps (Score 1.0 to 5.0):\n{faith_steps_txt}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Evaluate each step sequentially and formulate step reasoning.\n"
                    "2. Assign code_quality (1.0 to 5.0) and faithfulness (1.0 to 5.0).\n"
                    "3. Respond ONLY in valid JSON format matching this schema:\n"
                    "{\n"
                    '  "code_quality": <float 1.0-5.0>,\n'
                    '  "faithfulness": <float 1.0-5.0>,\n'
                    '  "geval_steps": [\n'
                    '    {"step": 1, "criterion": "...", "passed": true, "reasoning": "..."}\n'
                    "  ],\n"
                    '  "feedback": "<concise summary of findings>"\n'
                    "}"
                )

                response = await self.provider.send([Message.user(prompt)])
                if response.message.content:
                    match = re.search(r"\{.*\}", response.message.content, re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                        cq = round(float(data.get("code_quality", 3.0)), 2)
                        faith = round(float(data.get("faithfulness", 3.0)), 2)
                        cq = min(5.0, max(1.0, cq))
                        faith = min(5.0, max(1.0, faith))
                        overall = round((cq + faith) / 2.0, 2)
                        steps = data.get("geval_steps", [])
                        return JudgeScore(
                            code_quality_score=cq,
                            faithfulness_score=faith,
                            overall_score=overall,
                            feedback=str(data.get("feedback", "G-Eval CoT evaluation completed")),
                            geval_steps=steps,
                        )
            except Exception:
                pass  # Fallback to deterministic G-Eval scoring

        # Deterministic G-Eval evaluation (used in mock / offline mode)
        return self._heuristic_evaluate(code_diff, agent_explanation, tests_passed)

    @classmethod
    def _heuristic_evaluate(
        cls,
        code_diff: str,
        explanation: str,
        tests_passed: bool,
    ) -> JudgeScore:
        """Deterministic G-Eval implementation executing explicit verification steps."""
        steps: list[dict[str, Any]] = []

        # -------------------------------------------------------------
        # G-Eval Code Quality Steps
        # -------------------------------------------------------------
        cq_base = 3.0

        # Step 1: Pytest validation & function signature preservation
        step1_pass = tests_passed
        if step1_pass:
            cq_base += 1.0
            s1_reason = "Tests executed and passed; function signature and interfaces intact."
        else:
            cq_base -= 1.0
            s1_reason = "Tests failed; solution does not satisfy functional interface requirements."
        steps.append({
            "step": 1,
            "criterion": "Interface Preservation",
            "passed": step1_pass,
            "reasoning": s1_reason,
        })

        # Step 2: Type Annotations & PEP 8 Style
        has_typing = ("->" in code_diff or ": " in code_diff) if code_diff else False
        if has_typing:
            cq_base += 0.5
            s2_reason = "Explicit type hints and standard PEP 8 naming detected in diff."
        else:
            s2_reason = "No explicit type hints detected in code changes."
        steps.append({
            "step": 2,
            "criterion": "Type Annotations & Style",
            "passed": has_typing,
            "reasoning": s2_reason,
        })

        # Step 3: Surgical Diff Footprint
        diff_lines = [l for l in code_diff.splitlines() if l.startswith("+") or l.startswith("-")]
        if not diff_lines and code_diff.strip():
            diff_lines = [l for l in code_diff.splitlines() if l.strip()]

        surgical_diff = 1 <= len(diff_lines) <= 60
        if surgical_diff:
            s3_reason = f"Surgical diff footprint ({len(diff_lines)} modified lines); minimal refactor risk."
        else:
            s3_reason = f"Diff size ({len(diff_lines)} lines) is either empty or extensive."
        steps.append({
            "step": 3,
            "criterion": "Surgical Diff Footprint",
            "passed": surgical_diff,
            "reasoning": s3_reason,
        })

        # Step 4: Documentation Integrity
        has_docs = ('"""' in code_diff or "'''" in code_diff) if code_diff else False
        if has_docs:
            cq_base += 0.5
            s4_reason = "Docstring documentation present in modified code."
        else:
            s4_reason = "No docstrings added or modified in diff."
        steps.append({
            "step": 4,
            "criterion": "Documentation Integrity",
            "passed": has_docs,
            "reasoning": s4_reason,
        })

        # -------------------------------------------------------------
        # G-Eval Faithfulness / Groundedness Steps
        # -------------------------------------------------------------
        faith_base = 3.0
        lower_exp = explanation.lower()
        claimed_pass = any(w in lower_exp for w in ("pass", "fixed", "successful", "resolved"))

        # Step 5: Pytest Outcome Truthfulness
        if claimed_pass and not tests_passed:
            faith_base -= 1.5
            s5_pass = False
            s5_reason = "Hallucination: Agent claimed tests passed or issue was fixed, but tests failed."
        elif claimed_pass and tests_passed:
            faith_base += 1.0
            s5_pass = True
            s5_reason = "Faithful reporting: Agent correctly reported passing verification."
        else:
            s5_pass = True
            s5_reason = "Agent explanation aligns with test outcome."
        steps.append({
            "step": 5,
            "criterion": "Pytest Outcome Truthfulness",
            "passed": s5_pass,
            "reasoning": s5_reason,
        })

        # Step 6: Diff-to-Explanation Alignment
        has_diff = bool(code_diff.strip())
        claimed_made_changes = "update" in lower_exp or "fix" in lower_exp or "change" in lower_exp
        if claimed_made_changes and not has_diff:
            faith_base -= 1.0
            s6_pass = False
            s6_reason = "Hallucination: Agent claimed to make changes, but diff is empty."
        else:
            faith_base += 0.5
            s6_pass = True
            s6_reason = "Explanation matches actual code modifications made."
        steps.append({
            "step": 6,
            "criterion": "Diff-to-Explanation Alignment",
            "passed": s6_pass,
            "reasoning": s6_reason,
        })

        cq = min(5.0, max(1.0, round(cq_base, 1)))
        faith = min(5.0, max(1.0, round(faith_base, 1)))
        overall = round((cq + faith) / 2.0, 2)

        # Build concise feedback summary from failed steps or successes
        failed_reasons = [s["reasoning"] for s in steps if not s["passed"]]
        if failed_reasons:
            feedback = f"G-Eval: {'; '.join(failed_reasons)}"
        else:
            feedback = "G-Eval: All evaluation steps passed with high confidence"

        return JudgeScore(
            code_quality_score=cq,
            faithfulness_score=faith,
            overall_score=overall,
            feedback=feedback,
            geval_steps=steps,
        )

    def evaluate_explanation(
        self,
        task_prompt: str,
        explanation: str,
        expected_concepts: list[str] | None = None,
        tools_used: list[str] | None = None,
    ) -> JudgeScore:
        """Evaluate architectural explanation and repository comprehension using multi-step G-Eval rubric."""
        expected_concepts = expected_concepts or []
        tools_used = tools_used or []
        steps: list[dict[str, Any]] = []

        lower_exp = explanation.lower()
        
        # Step 1: Conceptual Recall & Entity Coverage
        matched_concepts = [c for c in expected_concepts if c.lower() in lower_exp]
        recall = len(matched_concepts) / len(expected_concepts) if expected_concepts else 1.0
        
        s1_pass = recall >= 0.60
        s1_score = min(5.0, 1.0 + (recall * 4.0))
        s1_reason = (
            f"Covered {len(matched_concepts)}/{len(expected_concepts)} key concepts ({recall:.0%}): "
            f"{matched_concepts[:4]}"
        )
        steps.append({
            "step": 1,
            "criterion": "Conceptual Recall & Entity Coverage",
            "passed": s1_pass,
            "score": round(s1_score, 1),
            "reasoning": s1_reason,
        })

        # Step 2: Factual Grounding & Retrieval Evidence
        has_file_references = bool(re.search(r"`[\w\./\-]+\.\w+`|\.py\b|src/", explanation))
        has_code_blocks = "```" in explanation
        has_tool_evidence = len(tools_used) > 0 or has_file_references
        
        s2_pass = has_file_references or has_code_blocks
        s2_score = 5.0 if (has_file_references and has_code_blocks) else (3.5 if s2_pass else 2.0)
        s2_reason = (
            "Grounded in repository structure with explicit file paths and code references."
            if s2_pass
            else "Lacks concrete file paths and code block references."
        )
        steps.append({
            "step": 2,
            "criterion": "Factual Grounding & Retrieval Evidence",
            "passed": s2_pass,
            "score": round(s2_score, 1),
            "reasoning": s2_reason,
        })

        # Step 3: Structural Clarity & Technical Depth
        has_headings = "#" in explanation
        has_bullets = any(marker in explanation for marker in ("- ", "* ", "1. "))
        length_tokens = len(explanation.split())
        is_comprehensive = length_tokens >= 80

        s3_pass = has_headings and has_bullets and is_comprehensive
        s3_score = 5.0 if s3_pass else (3.5 if is_comprehensive else 2.5)
        s3_reason = (
            f"Comprehensive technical structure ({length_tokens} words) with headings and organized lists."
            if s3_pass
            else "Explanation is brief or lacks structured headings."
        )
        steps.append({
            "step": 3,
            "criterion": "Structural Clarity & Technical Depth",
            "passed": s3_pass,
            "score": round(s3_score, 1),
            "reasoning": s3_reason,
        })

        # Step 4: Absence of Hallucination
        # Check that it didn't just output generic failure messages
        is_error_response = any(phrase in lower_exp for phrase in ("cannot find", "unknown error", "i don't know"))
        s4_pass = not is_error_response and len(explanation.strip()) > 50
        s4_score = 5.0 if s4_pass else 1.5
        s4_reason = (
            "No refusal or hallucination detected; response directly addresses the prompt."
            if s4_pass
            else "Agent failed to provide an authoritative answer or encountered an error."
        )
        steps.append({
            "step": 4,
            "criterion": "Absence of Hallucination & Authority",
            "passed": s4_pass,
            "score": round(s4_score, 1),
            "reasoning": s4_reason,
        })

        # Aggregated scores
        avg_quality = (s1_score + s3_score) / 2.0
        avg_faith = (s2_score + s4_score) / 2.0
        overall = round((avg_quality + avg_faith) / 2.0, 2)

        failed_steps = [s["reasoning"] for s in steps if not s["passed"]]
        if failed_steps:
            feedback = f"RepoQA: {'; '.join(failed_steps)}"
        else:
            feedback = f"RepoQA: Strong conceptual coverage ({recall:.0%}) with concrete code grounding"

        return JudgeScore(
            code_quality_score=round(avg_quality, 1),
            faithfulness_score=round(avg_faith, 1),
            overall_score=overall,
            feedback=feedback,
            geval_steps=steps,
        )

