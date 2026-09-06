"""Evaluation runner for benchmarking Terminal Agent."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evals.models import EvalReport, EvalResult, EvalTask
from evals.mock_provider import MockEvalProvider
from evals.tasks.fixtures import BENCHMARK_TASKS, get_all_tasks, get_task
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
    ):
        self.provider = provider
        self.model = model or ("mock-eval-model" if provider == "mock" else "claude-sonnet-4-20250514")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_task(self, task: EvalTask) -> EvalResult:
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

            finally:
                # Restore directory & environment
                os.chdir(original_cwd)
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            duration = time.perf_counter() - start_time

            # 5. Run pytest in the temporary directory to verify the fix
            test_run = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=str(temp_path),
                capture_output=True,
                text=True,
                timeout=30,
            )

            passed = (test_run.returncode == 0)
            output = test_run.stdout or test_run.stderr

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
            )

    async def run_suite(self, tasks: Sequence[EvalTask]) -> EvalReport:
        """Run all specified tasks and generate an aggregate benchmark report."""
        console.print(
            Panel.fit(
                f"[bold cyan]Terminal Agent Benchmark Harness[/bold cyan]\n"
                f"Provider: [green]{self.provider}[/green] | Model: [green]{self.model}[/green]\n"
                f"Total Tasks: [yellow]{len(tasks)}[/yellow]",
                title="Starting Evaluation Run",
            )
        )

        results: list[EvalResult] = []
        for i, task in enumerate(tasks, 1):
            console.print(f"[dim][{i}/{len(tasks)}][/dim] Evaluating [bold]{task.task_id}[/bold]: {task.name} ...")
            try:
                res = await self.run_task(task)
                badge = "[green]PASS[/green]" if res.passed else "[red]FAIL[/red]"
                console.print(f"       -> {badge} in {res.duration_seconds:.2f}s ({res.iterations} iters, ${res.cost:.4f})")
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
                    )
                )

        report = EvalReport.from_results(
            results=results,
            provider=self.provider,
            model_name=self.model,
        )

        self._display_summary(report)

        # Save results
        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"report_{ts_slug}.json"
        md_path = self.output_dir / f"report_{ts_slug}.md"
        
        report.save_json(json_path)
        report.save_markdown(md_path)

        console.print(f"\n[bold green][*][/bold green] Saved benchmark report to:")
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
        table.add_column("Duration", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Cost", justify="right")

        for r in report.results:
            status = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
            tokens_str = f"{r.input_tokens + r.output_tokens:,}"
            table.add_row(
                r.task_id,
                r.task_name,
                r.category,
                status,
                str(r.iterations),
                f"{r.duration_seconds:.2f}s",
                tokens_str,
                f"${r.cost:.4f}",
            )

        console.print("\n")
        console.print(table)
        
        pass_color = "green" if report.pass_rate_pct >= 80 else ("yellow" if report.pass_rate_pct >= 50 else "red")
        summary_panel = Panel.fit(
            f"Pass Rate: [{pass_color} bold]{report.pass_rate_pct}%[/{pass_color} bold] "
            f"([bold]{report.passed_tasks}/{report.total_tasks}[/bold] passed)\n"
            f"Total Duration: [bold]{report.total_duration_seconds:.2f}s[/bold]\n"
            f"Total Tokens: [bold]{report.total_input_tokens + report.total_output_tokens:,}[/bold]\n"
            f"Total Cost: [bold]${report.total_cost:.4f}[/bold]",
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

    args = parser.parse_args()

    provider = "mock" if args.mock or not args.provider else args.provider
    runner = EvalRunner(
        provider=provider,
        model=args.model,
        output_dir=args.output_dir,
    )

    all_tasks = get_all_tasks()
    if args.tasks:
        selected_ids = {t.strip() for t in args.tasks.split(",")}
        tasks = [t for t in all_tasks if t.task_id in selected_ids]
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
