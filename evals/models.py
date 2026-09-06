"""Data models for the evaluation and benchmarking framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any


class TaskCategory(str, Enum):
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    FEATURE = "feature"
    MULTI_FILE = "multi_file"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class EvalTask:
    """A coding task to evaluate an agent against."""
    task_id: str
    name: str
    category: TaskCategory
    difficulty: TaskDifficulty
    prompt: str
    initial_files: dict[str, str]
    test_files: dict[str, str]
    expected_files_modified: list[str] = field(default_factory=list)
    max_iterations: int = 15


@dataclass
class EvalResult:
    """The outcome of running the agent on a single evaluation task."""
    task_id: str
    task_name: str
    passed: bool
    category: str
    difficulty: str
    duration_seconds: float
    iterations: int
    input_tokens: int
    output_tokens: int
    cost: float
    error_message: str | None = None
    test_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    """Aggregated report of an entire benchmark run."""
    timestamp: str
    provider: str
    model_name: str
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    pass_rate_pct: float
    total_duration_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    results: list[EvalResult] = field(default_factory=list)

    @classmethod
    def from_results(
        cls,
        results: list[EvalResult],
        provider: str,
        model_name: str,
    ) -> EvalReport:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = (passed / total * 100.0) if total > 0 else 0.0

        total_duration = sum(r.duration_seconds for r in results)
        total_in_tokens = sum(r.input_tokens for r in results)
        total_out_tokens = sum(r.output_tokens for r in results)
        total_cost = sum(r.cost for r in results)

        return cls(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model_name=model_name,
            total_tasks=total,
            passed_tasks=passed,
            failed_tasks=failed,
            pass_rate_pct=round(pass_rate, 2),
            total_duration_seconds=round(total_duration, 2),
            total_input_tokens=total_in_tokens,
            total_output_tokens=total_out_tokens,
            total_cost=round(total_cost, 4),
            results=results,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["results"] = [r.to_dict() for r in self.results]
        return data

    def to_markdown(self) -> str:
        lines = [
            f"# Terminal Agent Evaluation Benchmark Report",
            f"",
            f"- **Timestamp:** `{self.timestamp}`",
            f"- **Provider / Model:** `{self.provider}` / `{self.model_name}`",
            f"- **Overall Pass Rate:** **{self.pass_rate_pct}%** ({self.passed_tasks}/{self.total_tasks} passed)",
            f"- **Total Duration:** {self.total_duration_seconds}s",
            f"- **Total Tokens:** {self.total_input_tokens + self.total_output_tokens:,} (Input: {self.total_input_tokens:,}, Output: {self.total_output_tokens:,})",
            f"- **Total Estimated Cost:** ${self.total_cost:.4f}",
            f"",
            f"## Task Results Breakdown",
            f"",
            f"| Task ID | Task Name | Category | Diff | Status | Iter | Time (s) | Cost ($) |",
            f"|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|",
        ]

        for r in self.results:
            status_badge = "✅ PASS" if r.passed else "❌ FAIL"
            lines.append(
                f"| `{r.task_id}` | {r.task_name} | {r.category} | {r.difficulty} | {status_badge} | {r.iterations} | {r.duration_seconds:.2f} | ${r.cost:.4f} |"
            )

        lines.append("")
        return "\n".join(lines)

    def save_json(self, output_path: Path | str) -> Path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    def save_markdown(self, output_path: Path | str) -> Path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")
        return p
