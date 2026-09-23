"""Evaluation runner for benchmarking Terminal Agent."""

from __future__ import annotations

import argparse
import asyncio
import difflib
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evals.judge import CodeJudge
from evals.mock_provider import MockEvalProvider
from evals.models import EvalReport, EvalResult, EvalTask, TaskCategory
from evals.tasks.fixtures import get_all_tasks
from evals.trajectory import TrajectoryAnalyzer
from terminal_agent.core.agent import Agent
from terminal_agent.core.config import AgentConfig

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(safe_box=True)


class EvalRunner:
    """Orchestrates running benchmark tasks in isolated environments."""

    def __init__(
        self,
        provider: str = "mock",
        model: str | None = None,
        output_dir: Path | str = "evals/results",
        trials: int = 1,
        enable_judge: bool = False,
        include_security: bool = False,
    ):
        self.provider = provider
        self.model = model or ("mock-eval-model" if provider == "mock" else "claude-sonnet-4-20250514")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trials = max(1, trials)
        self.enable_judge = enable_judge
        self.include_security = include_security

    async def run_task(self, task: EvalTask, trial_number: int = 1) -> EvalResult:
        """Run a single evaluation task in an isolated temporary directory."""
        start_time = time.perf_counter()
        
        with tempfile.TemporaryDirectory(prefix=f"eval_{task.task_id}_") as temp_dir:
            temp_path = Path(temp_dir)

            # 1. Populate initial broken code files
            for rel_path, content in task.initial_files.items():
                file_path = temp_path / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            # 2. Populate evaluation tests
            for rel_path, content in task.test_files.items():
                file_path = temp_path / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            # 3. Configure agent in automated 'yolo' mode
            env_override = {
                "AGENT_PERMISSION_MODE": "yolo",
                "AGENT_MAX_ITERATIONS": str(task.max_iterations),
            }
            
            if self.provider != "mock":
                env_override["AGENT_PROVIDER"] = self.provider
                env_override["AGENT_MODEL_NAME"] = self.model
            else:
                # Provide dummy key for mock so validation passes
                env_override["AGENT_PROVIDER"] = "openai"
                env_override["OPENAI_API_KEY"] = "mock-eval-key"

            # Apply env overrides and switch cwd
            old_env = {}
            for k, v in env_override.items():
                old_env[k] = os.environ.get(k)
                os.environ[k] = v

            original_cwd = os.getcwd()
            os.chdir(temp_path)

            try:
                config = AgentConfig()
                agent = Agent(config=config, working_directory=str(temp_path))

                if self.provider == "mock":
                    mock_provider = MockEvalProvider(model=self.model, current_task_id=task.task_id)
                    agent.provider = mock_provider

                # 4. Run the ReAct loop
                await agent.process_message(task.prompt)

                iterations = agent.session.iteration_count
                input_tokens = agent.session.total_input_tokens
                output_tokens = agent.session.total_output_tokens
                cost = agent.cost_tracker.get_session_cost()
                messages = list(agent.session.messages)

            finally:
                # Restore directory & environment
                os.chdir(original_cwd)
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            duration = time.perf_counter() - start_time

            # 5. Verification step: Pytest (for coding tasks) or RepoQA evaluation (for comprehension tasks)
            judge_score = None
            if task.category == TaskCategory.REPO_QA:
                last_assistant_msg = ""
                for m in reversed(messages):
                    if m.role.value == "assistant" and m.content:
                        last_assistant_msg = m.content
                        break

                tools_used = [
                    tc.name for m in messages if m.tool_calls for tc in m.tool_calls
                ]

                judge_provider = agent.provider if self.provider != "mock" else None
                judge = CodeJudge(provider=judge_provider)
                judge_score = judge.evaluate_explanation(
                    task_prompt=task.prompt,
                    explanation=last_assistant_msg,
                    expected_concepts=task.expected_concepts,
                    tools_used=tools_used,
                )

                # Passes if overall score >= 3.5
                passed = judge_score.overall_score >= 3.5
                output = f"RepoQA Score: {judge_score.overall_score}/5.0 | {judge_score.feedback}"
            else:
                test_run = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q"],
                    cwd=str(temp_path),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                passed = (test_run.returncode == 0)
                output = test_run.stdout or test_run.stderr

            # 6. Trajectory analytics
            trajectory = TrajectoryAnalyzer.analyze(messages)

            # 7. Compute code diff of changes made
            diff_lines: list[str] = []
            for rel_path, initial_content in task.initial_files.items():
                cur_file = temp_path / rel_path
                if cur_file.exists():
                    try:
                        cur_content = cur_file.read_text(encoding="utf-8")
                    except Exception:
                        cur_content = ""
                    if cur_content != initial_content:
                        diff = difflib.unified_diff(
                            initial_content.splitlines(keepends=True),
                            cur_content.splitlines(keepends=True),
                            fromfile=f"a/{rel_path}",
                            tofile=f"b/{rel_path}",
                        )
                        diff_lines.extend(diff)
                else:
                    diff = difflib.unified_diff(
                        initial_content.splitlines(keepends=True),
                        [],
                        fromfile=f"a/{rel_path}",
                        tofile="/dev/null",
                    )
                    diff_lines.extend(diff)

            for p in temp_path.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(temp_path).as_posix()
                    if rel not in task.initial_files and rel not in task.test_files and not rel.startswith(".pytest_cache"):
                        try:
                            new_content = p.read_text(encoding="utf-8")
                            diff = difflib.unified_diff(
                                [],
                                new_content.splitlines(keepends=True),
                                fromfile="/dev/null",
                                tofile=f"b/{rel}",
                            )
                            diff_lines.extend(diff)
                        except Exception:
                            pass

            code_diff = "".join(diff_lines)

            # 8. LLM-as-a-Judge semantic evaluation for code changes
            if self.enable_judge and task.category != TaskCategory.REPO_QA:
                last_assistant_msg = ""
                for m in reversed(messages):
                    if m.role.value == "assistant" and m.content:
                        last_assistant_msg = m.content
                        break

                judge_provider = agent.provider if self.provider != "mock" else None
                judge = CodeJudge(provider=judge_provider)
                judge_score = await judge.evaluate(
                    task_prompt=task.prompt,
                    code_diff=code_diff,
                    agent_explanation=last_assistant_msg,
                    tests_passed=passed,
                )

            return EvalResult(
                task_id=task.task_id,
                task_name=task.name,
                passed=passed,
                category=task.category.value,
                difficulty=task.difficulty.value,
                duration_seconds=round(duration, 2),
                iterations=iterations,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=round(cost, 4),
                error_message=None if passed else output.strip()[:300],
                test_output=output.strip(),
                trial_number=trial_number,
                trajectory=trajectory,
                judge_score=judge_score,
            )

    async def run_suite(self, tasks: Sequence[EvalTask]) -> EvalReport:
        """Run all specified tasks across configured trials and generate a benchmark report."""
        effective_tasks = list(tasks)
        if self.include_security:
            from evals.tasks.security_fixtures import get_security_tasks
            sec_tasks = get_security_tasks()
            existing_ids = {t.task_id for t in effective_tasks}
            effective_tasks.extend([st for st in sec_tasks if st.task_id not in existing_ids])

        trial_info = f" | Trials per task: [yellow]{self.trials}[/yellow]" if self.trials > 1 else ""
        judge_info = " | Judge: [green]ENABLED[/green]" if self.enable_judge else ""
        console.print(
            Panel.fit(
                f"[bold cyan]Terminal Agent Benchmark Harness[/bold cyan]\n"
                f"Provider: [green]{self.provider}[/green] | Model: [green]{self.model}[/green]\n"
                f"Total Tasks: [yellow]{len(effective_tasks)}[/yellow]{trial_info}{judge_info}",
                title="Starting Evaluation Run",
            )
        )

        results: list[EvalResult] = []
        for trial in range(1, self.trials + 1):
            trial_prefix = f"[Trial {trial}/{self.trials}] " if self.trials > 1 else ""
            for i, task in enumerate(effective_tasks, 1):
                console.print(f"[dim][{i}/{len(effective_tasks)}][/dim] {trial_prefix}Evaluating [bold]{task.task_id}[/bold]: {task.name} ...")
                try:
                    res = await self.run_task(task, trial_number=trial)
                    badge = "[green]PASS[/green]" if res.passed else "[red]FAIL[/red]"
                    judge_str = f" | Quality: {res.judge_score.code_quality_score}/5" if res.judge_score else ""
                    tools_str = f", {res.trajectory.total_tool_calls} tools" if res.trajectory else ""
                    console.print(f"       -> {badge} in {res.duration_seconds:.2f}s ({res.iterations} iters{tools_str}, ${res.cost:.4f}{judge_str})")
                    results.append(res)
                except Exception as e:
                    console.print(f"       -> [bold red]CRASHED[/bold red]: {e}")
                    results.append(
                        EvalResult(
                            task_id=task.task_id,
                            task_name=task.name,
                            passed=False,
                            category=task.category.value,
                            difficulty=task.difficulty.value,
                            duration_seconds=0.0,
                            iterations=0,
                            input_tokens=0,
                            output_tokens=0,
                            cost=0.0,
                            error_message=str(e),
                            trial_number=trial,
                        )
                    )

        report = EvalReport.from_results(
            results=results,
            provider=self.provider,
            model_name=self.model,
            trials_per_task=self.trials,
        )

        self._display_summary(report)

        # Save results
        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"report_{ts_slug}.json"
        md_path = self.output_dir / f"report_{ts_slug}.md"
        
        report.save_json(json_path)
        report.save_markdown(md_path)

        console.print("\n[bold green][*][/bold green] Saved benchmark report to:")
        console.print(f"  - JSON: [dim]{json_path}[/dim]")
        console.print(f"  - Markdown: [dim]{md_path}[/dim]\n")

        return report

    def _display_summary(self, report: EvalReport) -> None:
        """Render a formatted summary table in the terminal."""
        table = Table(title=f"Benchmark Results — Pass Rate: {report.pass_rate_pct}%")
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Task Name")
        table.add_column("Category", style="magenta")
        table.add_column("Status", justify="center")
        table.add_column("Iters", justify="right")
        table.add_column("Tools", justify="right")
        has_judge = report.avg_code_quality is not None or any(r.judge_score for r in report.results)
        if has_judge:
            table.add_column("Quality", justify="right")
            table.add_column("Faith", justify="right")
        table.add_column("Duration", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Cost", justify="right")

        for r in report.results:
            status = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
            tokens_str = f"{r.input_tokens + r.output_tokens:,}"
            tools_str = str(r.trajectory.total_tool_calls) if r.trajectory else "-"
            
            row = [
                r.task_id,
                r.task_name,
                r.category,
                status,
                str(r.iterations),
                tools_str,
            ]
            if has_judge:
                cq = f"{r.judge_score.code_quality_score:.1f}" if r.judge_score else "-"
                faith = f"{r.judge_score.faithfulness_score:.1f}" if r.judge_score else "-"
                row.extend([cq, faith])

            row.extend([
                f"{r.duration_seconds:.2f}s",
                tokens_str,
                f"${r.cost:.4f}",
            ])
            table.add_row(*row)

        console.print("\n")
        console.print(table)
        
        pass_color = "green" if report.pass_rate_pct >= 80 else ("yellow" if report.pass_rate_pct >= 50 else "red")
        summary_lines = [
            f"Pass Rate: [{pass_color} bold]{report.pass_rate_pct}%[/{pass_color} bold] "
            f"([bold]{report.passed_tasks}/{report.total_tasks}[/bold] passed)"
        ]
        if report.trials_per_task > 1 and report.pass_at_k_pct is not None:
            summary_lines.append(
                f"Pass@{report.trials_per_task}: [bold]{report.pass_at_k_pct}%[/bold] | "
                f"Pass^{report.trials_per_task} (Consistency): [bold]{report.pass_all_k_pct}%[/bold]"
            )
        if report.avg_code_quality is not None:
            summary_lines.append(
                f"Avg Quality: [bold]{report.avg_code_quality}/5.0[/bold] | "
                f"Avg Faithfulness: [bold]{report.avg_faithfulness}/5.0[/bold]"
            )
        summary_lines.extend([
            f"Total Duration: [bold]{report.total_duration_seconds:.2f}s[/bold]",
            f"Total Tokens: [bold]{report.total_input_tokens + report.total_output_tokens:,}[/bold]",
            f"Total Cost: [bold]${report.total_cost:.4f}[/bold]",
        ])

        summary_panel = Panel.fit(
            "\n".join(summary_lines),
            title="Benchmark Summary",
            border_style=pass_color,
        )
        console.print(summary_panel)


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Terminal Agent Evaluation Runner")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock provider (zero API cost, instant offline run)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider: anthropic, openai, nvidia (default: mock)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to evaluate",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task IDs to run (default: all tasks)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evals/results",
        help="Directory to save JSON & Markdown reports",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per task to evaluate Pass@k and consistency",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable LLM-as-a-Judge semantic quality and faithfulness evaluation",
    )
    parser.add_argument(
        "--security",
        action="store_true",
        help="Run or include adversarial security red-teaming tasks",
    )

    args = parser.parse_args()

    provider = "mock" if args.mock or not args.provider else args.provider
    runner = EvalRunner(
        provider=provider,
        model=args.model,
        output_dir=args.output_dir,
        trials=args.trials,
        enable_judge=args.judge,
        include_security=args.security,
    )

    all_tasks = get_all_tasks()
    if args.security:
        from evals.tasks.security_fixtures import get_security_tasks
        sec_tasks = get_security_tasks()
        if args.tasks:
            combined = all_tasks + sec_tasks
            selected_ids = {t.strip() for t in args.tasks.split(",")}
            tasks = [t for t in combined if t.task_id in selected_ids]
        else:
            tasks = sec_tasks
    elif args.tasks:
        from evals.tasks.security_fixtures import get_security_tasks
        combined = all_tasks + get_security_tasks()
        selected_ids = {t.strip() for t in args.tasks.split(",")}
        tasks = [t for t in combined if t.task_id in selected_ids]
    else:
        tasks = all_tasks

    if not tasks:
        console.print("[red]No tasks selected for evaluation.[/red]")
        return 1

    report = await runner.run_suite(tasks)
    return 0 if report.passed_tasks > 0 else 1


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
