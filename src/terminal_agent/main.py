"""Terminal Agent — Entry Point.

Usage:
    agent                    Start interactive session
    agent --provider openai  Use a specific LLM provider
    agent --model gpt-4o     Use a specific model
    agent --yolo             Auto-approve all commands (dangerous!)
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console

from terminal_agent.core.config import AgentConfig, PermissionMode

app = typer.Typer(
    name="agent",
    help="Terminal Agent — A CLI-based AI coding assistant.",
    add_completion=False,
    no_args_is_help=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    profile: Optional[str] = typer.Option(
        None, "--profile", "-P",
        help="Model profile preset (e.g., sonnet, haiku, nemotron, gpt4o, deepseek)",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p",
        help="LLM provider: anthropic, nvidia, openai, openrouter",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model name (e.g., claude-sonnet-4-20250514, gpt-4o)",
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens",
        help="Max output tokens per response",
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations",
        help="Max tool-use loops per user turn",
    ),
    permission: Optional[str] = typer.Option(
        None, "--permission",
        help="Permission mode: safe, auto-test, yolo",
    ),
    yolo: bool = typer.Option(
        False, "--yolo",
        help="Auto-approve all commands (equivalent to --permission yolo)",
    ),
) -> None:
    """Start an interactive Terminal Agent session."""
    if ctx.invoked_subcommand is not None:
        return

    # Build config overrides from CLI arguments
    overrides = {}
    if profile:
        overrides["profile"] = profile
    if provider:
        overrides["provider"] = provider
    if model:
        overrides["model_name"] = model
    if max_tokens:
        overrides["max_tokens"] = max_tokens
    if max_iterations:
        overrides["max_iterations"] = max_iterations
    if permission:
        overrides["permission_mode"] = PermissionMode(permission)
    if yolo:
        overrides["permission_mode"] = PermissionMode.YOLO

    try:
        config = AgentConfig(**overrides)
    except Exception as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        raise typer.Exit(1)

    # Run the interactive loop
    from terminal_agent.cli import run_interactive

    try:
        asyncio.run(run_interactive(config))
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


@app.command(name="eval")
def eval_benchmark(
    mock: bool = typer.Option(
        True, "--mock/--live",
        help="Use deterministic mock provider (zero API cost, instant offline run)",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p",
        help="LLM provider: anthropic, openai, nvidia",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model name to evaluate",
    ),
    tasks: Optional[str] = typer.Option(
        None, "--tasks", "-t",
        help="Comma-separated task IDs to run (default: all tasks)",
    ),
    output_dir: str = typer.Option(
        "evals/results", "--output-dir", "-o",
        help="Directory to save JSON & Markdown reports",
    ),
) -> None:
    """Run automated benchmark evaluations on Terminal Agent."""
    from evals.runner import EvalRunner
    from evals.tasks.fixtures import get_all_tasks

    active_provider = "mock" if mock or not provider else provider
    runner = EvalRunner(
        provider=active_provider,
        model=model,
        output_dir=output_dir,
    )

    all_tasks = get_all_tasks()
    if tasks:
        selected_ids = {t.strip() for t in tasks.split(",")}
        tasks_to_run = [t for t in all_tasks if t.task_id in selected_ids]
    else:
        tasks_to_run = all_tasks

    if not tasks_to_run:
        console.print("[red]No matching tasks found.[/red]")
        raise typer.Exit(1)

    report = asyncio.run(runner.run_suite(tasks_to_run))
    if report.failed_tasks > 0:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

