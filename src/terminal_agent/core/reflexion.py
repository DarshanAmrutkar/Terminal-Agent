"""Reflexion episodic failure memory and verbal reinforcement engine.

Inspired by:
Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023).
Provides a bounded episodic memory buffer of failed actions and verbal self-reflections
to prevent amnesic repetition loops and guide corrective strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from typing import Any, List, Optional


@dataclass
class FailureEpisode:
    """Record of a failed action and its associated self-reflection."""
    tool_name: str
    action_summary: str
    error_output: str
    reflection: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "action_summary": self.action_summary,
            "error_output": self.error_output[:500],
            "reflection": self.reflection,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
        }


class ReflexionMemory:
    """Bounded episodic memory buffer maintaining verbal self-reflections on failures."""

    def __init__(self, max_episodes: int = 5) -> None:
        self.max_episodes = max_episodes
        self.episodes: List[FailureEpisode] = []

    def record_failure(
        self,
        tool_name: str,
        args: dict[str, Any] | None,
        error_output: str,
        custom_reflection: Optional[str] = None,
    ) -> FailureEpisode:
        """Synthesize a verbal reflection and store the failure episode."""
        action_summary = self._summarize_action(tool_name, args or {})
        reflection = custom_reflection or self._synthesize_heuristic_reflection(tool_name, error_output)

        # Check for existing similar episode to update rather than duplicate
        for ep in self.episodes:
            if not ep.resolved and ep.tool_name == tool_name and ep.action_summary == action_summary:
                ep.error_output = error_output
                ep.reflection = reflection
                ep.timestamp = datetime.now()
                return ep

        episode = FailureEpisode(
            tool_name=tool_name,
            action_summary=action_summary,
            error_output=error_output,
            reflection=reflection,
        )

        self.episodes.append(episode)

        # Maintain bounded buffer size
        if len(self.episodes) > self.max_episodes:
            # Drop oldest resolved episode first, or oldest episode
            resolved_indices = [i for i, ep in enumerate(self.episodes) if ep.resolved]
            if resolved_indices:
                self.episodes.pop(resolved_indices[0])
            else:
                self.episodes.pop(0)

        return episode

    def mark_resolved(self, tool_name: Optional[str] = None) -> int:
        """Mark failures for a specific tool or all pending failures as resolved."""
        count = 0
        for ep in self.episodes:
            if not ep.resolved and (tool_name is None or ep.tool_name == tool_name):
                ep.resolved = True
                count += 1
        return count

    def has_active_reflections(self) -> bool:
        """True if there are unresolved failure episodes in memory."""
        return any(not ep.resolved for ep in self.episodes)

    def get_active_episodes(self) -> List[FailureEpisode]:
        """Return list of unresolved failure episodes."""
        return [ep for ep in self.episodes if not ep.resolved]

    def format_context_block(self) -> str:
        """Format active failure reflections for injection into LLM context."""
        active = self.get_active_episodes()
        if not active:
            return ""

        lines = [
            "## Episodic Self-Reflections (Learn from Past Mistakes - Reflexion)",
            "Prior actions encountered failures. Do not repeat the same errors. Review past reflections:",
        ]

        for i, ep in enumerate(active[-self.max_episodes:], 1):
            lines.append(
                f"{i}. [Failed Tool: `{ep.tool_name}` on {ep.action_summary}]\n"
                f"   Error: {ep.error_output.strip().splitlines()[0] if ep.error_output.strip() else 'Unknown'}\n"
                f"   Reflection & Advice: {ep.reflection}"
            )

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all failure episodes."""
        self.episodes.clear()

    def _summarize_action(self, tool_name: str, args: dict[str, Any]) -> str:
        """Create a compact, human-readable summary of the attempted action."""
        if tool_name in ("read_file", "write_file", "search_replace"):
            path = args.get("path") or args.get("file_path", "unknown")
            return f"path='{path}'"
        elif tool_name == "run_command":
            cmd = args.get("command") or args.get("cmd", "")
            return f"command='{cmd[:60]}'"
        elif tool_name == "grep_search":
            return f"query='{args.get('query', '')}'"
        return json.dumps(args)[:60]

    def _synthesize_heuristic_reflection(self, tool_name: str, error_output: str) -> str:
        """Synthesize an actionable verbal self-reflection from the error output."""
        err_lower = error_output.lower()

        if "syntax gate" in err_lower or "syntaxerror" in err_lower:
            return (
                "Pre-commit syntax validation rejected the edit. "
                "Ensure valid Python syntax, closed parentheses/brackets, and correct colons before rewriting."
            )
        elif "indentationerror" in err_lower or "unexpected indent" in err_lower:
            return (
                "Indentation mismatch detected. Inspect existing file indentation (spaces vs tabs) "
                "and align the replacement block exactly with surrounding indentation."
            )
        elif "jsondecodeerror" in err_lower:
            return "Malformed JSON syntax. Ensure double-quoted keys, valid commas, and matching braces."
        elif "path traversal" in err_lower or "outside working directory" in err_lower:
            return (
                "Path traversal was blocked by sandbox policy. "
                "All operations must target paths strictly within the repository workspace."
            )
        elif "file not found" in err_lower or "does not exist" in err_lower:
            return (
                "Target file or path does not exist. "
                "Inspect directory contents first using list_directory to locate the correct file path."
            )
        elif "permission denied" in err_lower:
            return (
                "Permission denied accessing this file or command. "
                "Verify file permissions or select an alternative accessible approach."
            )
        elif "pytest" in err_lower or "failed" in err_lower or "assertionerror" in err_lower:
            return (
                "Test verification failed. "
                "Examine the failure trace and assertion mismatch carefully before altering implementation."
            )
        elif "could not find match for the search string" in err_lower:
            return (
                "Search text did not match the file content. "
                "Read the latest file content using read_file to verify line numbers and exact whitespace before replacing."
            )

        return (
            f"Action on {tool_name} failed. "
            "Re-assess the preconditions and avoid repeating the exact same arguments."
        )
