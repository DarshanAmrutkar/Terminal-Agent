"""Data models for the evaluation and benchmarking framework."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TaskCategory(str, Enum):
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    FEATURE = "feature"
    MULTI_FILE = "multi_file"
    REPO_QA = "repo_qa"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class EvalTask:
    """A coding or comprehension task to evaluate an agent against."""
    task_id: str
    name: str
    category: TaskCategory
    difficulty: TaskDifficulty
    prompt: str
    initial_files: dict[str, str] = field(default_factory=dict)
    test_files: dict[str, str] = field(default_factory=dict)
    expected_files_modified: list[str] = field(default_factory=list)
    expected_concepts: list[str] = field(default_factory=list)
    max_iterations: int = 15


from evals.judge import JudgeScore
from evals.trajectory import TrajectoryMetrics


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
    trial_number: int = 1
    trajectory: TrajectoryMetrics | None = None
    judge_score: JudgeScore | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.trajectory:
            data["trajectory"] = self.trajectory.to_dict()
        if self.judge_score:
            data["judge_score"] = self.judge_score.to_dict()
        return data


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
    trials_per_task: int = 1
    pass_at_k_pct: float | None = None
    pass_all_k_pct: float | None = None
    avg_code_quality: float | None = None
    avg_faithfulness: float | None = None

    @classmethod
    def from_results(
        cls,
        results: list[EvalResult],
        provider: str,
        model_name: str,
        trials_per_task: int = 1,
    ) -> EvalReport:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = (passed / total * 100.0) if total > 0 else 0.0

        total_duration = sum(r.duration_seconds for r in results)
        total_in_tokens = sum(r.input_tokens for r in results)
        total_out_tokens = sum(r.output_tokens for r in results)
        total_cost = sum(r.cost for r in results)

        # Multi-trial Pass@k and consistency calculation
        pass_at_k = None
        pass_all_k = None
        if trials_per_task > 1 and total > 0:
            task_runs: dict[str, list[bool]] = {}
            for r in results:
                task_runs.setdefault(r.task_id, []).append(r.passed)
            unique_tasks = len(task_runs)
            at_least_one = sum(1 for outcomes in task_runs.values() if any(outcomes))
            all_passed = sum(1 for outcomes in task_runs.values() if all(outcomes))
            pass_at_k = round((at_least_one / unique_tasks) * 100.0, 2)
            pass_all_k = round((all_passed / unique_tasks) * 100.0, 2)

        # Judge averages if present
        judged = [r.judge_score for r in results if r.judge_score is not None]
        avg_cq = round(sum(j.code_quality_score for j in judged) / len(judged), 2) if judged else None
        avg_faith = round(sum(j.faithfulness_score for j in judged) / len(judged), 2) if judged else None

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
            trials_per_task=trials_per_task,
            pass_at_k_pct=pass_at_k,
            pass_all_k_pct=pass_all_k,
            avg_code_quality=avg_cq,
            avg_faithfulness=avg_faith,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["results"] = [r.to_dict() for r in self.results]
        return data

    def to_markdown(self) -> str:
        lines = [
            "# Terminal Agent Evaluation Benchmark Report",
            "",
            f"- **Timestamp:** `{self.timestamp}`",
            f"- **Provider / Model:** `{self.provider}` / `{self.model_name}`",
            f"- **Overall Pass Rate:** **{self.pass_rate_pct}%** ({self.passed_tasks}/{self.total_tasks} passed)",
        ]

        if self.trials_per_task > 1:
            lines.extend([
                f"- **Trials per Task (k):** `{self.trials_per_task}`",
                f"- **Pass@k Rate:** **{self.pass_at_k_pct}%**",
                f"- **Pass^k (Consistency):** **{self.pass_all_k_pct}%**",
            ])

        if self.avg_code_quality is not None:
            lines.append(f"- **Avg Code Quality (Judge):** **{self.avg_code_quality} / 5.0**")
        if self.avg_faithfulness is not None:
            lines.append(f"- **Avg Faithfulness (Judge):** **{self.avg_faithfulness} / 5.0**")

        lines.extend([
            f"- **Total Duration:** {self.total_duration_seconds}s",
            f"- **Total Tokens:** {self.total_input_tokens + self.total_output_tokens:,} (Input: {self.total_input_tokens:,}, Output: {self.total_output_tokens:,})",
            f"- **Total Estimated Cost:** ${self.total_cost:.4f}",
            "",
            "## Task Results Breakdown",
            "",
            "| Task ID | Task Name | Category | Status | Iter | Tools | Quality | Faith | Cost ($) |",
            "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])

        for r in self.results:
            status_badge = "✅ PASS" if r.passed else "❌ FAIL"
            tools_used = f"{r.trajectory.total_tool_calls}" if r.trajectory else "-"
            cq = f"{r.judge_score.code_quality_score:.1f}" if r.judge_score else "-"
            faith = f"{r.judge_score.faithfulness_score:.1f}" if r.judge_score else "-"
            lines.append(
                f"| `{r.task_id}` | {r.task_name} | {r.category} | {status_badge} | {r.iterations} | {tools_used} | {cq} | {faith} | ${r.cost:.4f} |"
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
