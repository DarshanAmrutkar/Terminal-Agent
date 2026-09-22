"""Trajectory and behavioral process analytics for Agent evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

from terminal_agent.llm.message import Message, Role


@dataclass
class TrajectoryMetrics:
    """Quantitative behavioral metrics extracted from an agent's execution trajectory."""
    total_tool_calls: int = 0
    unique_tools_used: list[str] = field(default_factory=list)
    redundant_file_reads: int = 0
    failed_tool_calls: int = 0
    recovery_attempts: int = 0
    oscillation_detected: bool = False
    tool_sequence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def tool_distribution(self) -> dict[str, int]:
        """Distribution count of each tool invoked."""
        dist: dict[str, int] = {}
        for t in self.tool_sequence:
            dist[t] = dist.get(t, 0) + 1
        return dist

    @property
    def tool_efficiency_ratio(self) -> float:
        """Ratio of unique tools to total calls (higher means less wasteful repeated calls)."""
        if self.total_tool_calls == 0:
            return 1.0
        return round(len(self.unique_tools_used) / self.total_tool_calls, 2)


class TrajectoryAnalyzer:
    """Analyzes conversation history to evaluate agent problem-solving behavior."""

    @classmethod
    def analyze(cls, messages: list[Message]) -> TrajectoryMetrics:
        tool_sequence: list[str] = []
        unique_tools: set[str] = set()
        read_files_since_last_write: set[str] = set()
        redundant_reads = 0
        failed_tools = 0
        recovery_attempts = 0
        had_failure = False

        call_signatures: list[str] = []

        for msg in messages:
            if msg.role == Role.ASSISTANT and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.name
                    tool_sequence.append(name)
                    unique_tools.add(name)

                    args = tc.arguments or {}
                    sig = f"{name}:{json.dumps(args, sort_keys=True)}"
                    call_signatures.append(sig)

                    if name == "read_file":
                        path = args.get("path")
                        if path:
                            if path in read_files_since_last_write:
                                redundant_reads += 1
                            else:
                                read_files_since_last_write.add(path)
                    elif name in ("write_file", "search_replace"):
                        # Modifying a file resets read history
                        read_files_since_last_write.clear()

            elif msg.role == Role.TOOL_RESULT and msg.tool_results:
                for tr in msg.tool_results:
                    if tr.is_error or tr.output.startswith("Error:") or tr.output.startswith("FAILED"):
                        failed_tools += 1
                        had_failure = True
                    else:
                        if had_failure:
                            recovery_attempts += 1
                            had_failure = False

        # Detect oscillation (repeated identical calls or cyclical loops)
        oscillation = cls._detect_oscillation(call_signatures)

        return TrajectoryMetrics(
            total_tool_calls=len(tool_sequence),
            unique_tools_used=sorted(unique_tools),
            redundant_file_reads=redundant_reads,
            failed_tool_calls=failed_tools,
            recovery_attempts=recovery_attempts,
            oscillation_detected=oscillation,
            tool_sequence=tool_sequence,
        )

    @staticmethod
    def _detect_oscillation(signatures: list[str]) -> bool:
        """Detect repetitive cyclical tool call patterns."""
        if len(signatures) < 3:
            return False

        # Check 1-cycle repeated identical call: A, A, A
        for i in range(len(signatures) - 2):
            if signatures[i] == signatures[i+1] == signatures[i+2]:
                return True

        if len(signatures) < 4:
            return False

        # Check 2-cycle oscillation: A, B, A, B
        for i in range(len(signatures) - 3):
            if signatures[i] == signatures[i+2] and signatures[i+1] == signatures[i+3] and signatures[i] != signatures[i+1]:
                return True

        # Check 3-cycle oscillation: A, B, C, A, B, C
        if len(signatures) >= 6:
            for i in range(len(signatures) - 5):
                if (signatures[i] == signatures[i+3] and 
                    signatures[i+1] == signatures[i+4] and 
                    signatures[i+2] == signatures[i+5]):
                    return True

        return False
