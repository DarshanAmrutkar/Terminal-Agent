"""Terminal Agent — Entry Point.

Usage:
    agent                    Start interactive session
    agent --provider openai  Use a specific LLM provider
    agent --model gpt-4o     Use a specific model
    agent --yolo             Auto-approve all commands (dangerous!)
"""

from __future__ import annotations

import asyncio

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
    profile: str | None = typer.Option(
        None, "--profile", "-P",
        help="Model profile preset (e.g., sonnet, haiku, nemotron, gpt4o, deepseek)",
    ),
    provider: str | None = typer.Option(
        None, "--provider", "-p",
        help="LLM provider: anthropic, nvidia, openai, openrouter",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Model name (e.g., claude-sonnet-4-20250514, gpt-4o)",
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens",
        help="Max output tokens per response",
    ),
    max_iterations: int | None = typer.Option(
        None, "--max-iterations",
        help="Max tool-use loops per user turn",
    ),
    permission: str | None = typer.Option(
        None, "--permission",
        help="Permission mode: safe, auto-test, yolo",
    ),
    yolo: bool = typer.Option(
        False, "--yolo",
        help="Auto-approve all commands (equivalent to --permission yolo)",
    ),
    resume: str | None = typer.Option(
        None, "--resume", "-r",
        help="Resume a prior session by ID or 'latest'",
    ),
    fallback_provider: str | None = typer.Option(
        None, "--fallback-provider",
        help="Fallback LLM provider to switch to on rate limits/outages",
    ),
    enable_failover: bool = typer.Option(
        False, "--enable-failover",
        help="Enable automated circuit-breaker provider failover",
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
    if fallback_provider:
        overrides["fallback_provider"] = fallback_provider
        overrides["enable_failover"] = True
    if enable_failover:
        overrides["enable_failover"] = True

    try:
        config = AgentConfig(**overrides)
    except Exception as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        raise typer.Exit(1)

    # Run the interactive loop
    from terminal_agent.cli import run_interactive

    try:
        asyncio.run(run_interactive(config, resume_id=resume))
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


@app.command(name="eval")
def eval_benchmark(
    mock: bool = typer.Option(
        True, "--mock/--live",
        help="Use deterministic mock provider (zero API cost, instant offline run)",
    ),
    provider: str | None = typer.Option(
        None, "--provider", "-p",
        help="LLM provider: anthropic, openai, nvidia",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Model name to evaluate",
    ),
    tasks: str | None = typer.Option(
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


@app.command(name="sessions")
def list_sessions(
    limit: int = typer.Option(
        15, "--limit", "-n",
        help="Max number of sessions to list",
    ),
) -> None:
    """List recent saved sessions and status."""
    from rich.table import Table

    from terminal_agent.core.session_store import SessionStore

    store = SessionStore()
    sessions = store.list_sessions(limit=limit)

    if not sessions:
        console.print("[dim]No saved sessions found.[/dim]")
        return

    table = Table(title="Recent Agent Sessions", border_style="cyan")
    table.add_column("Session ID", style="bold green")
    table.add_column("Updated", style="dim")
    table.add_column("Working Dir", style="blue")
    table.add_column("Msgs", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("First Prompt Preview", style="italic")

    for s in sessions:
        table.add_row(
            s["session_id"],
            s["updated_at"],
            s["working_directory"][:30] + ("..." if len(s["working_directory"]) > 30 else ""),
            str(s["message_count"]),
            f"{s['total_tokens']:,}",
            s["preview"],
        )

    console.print(table)
    console.print("[dim]Resume any session via: agent --resume <session_id>[/dim]\n")


if __name__ == "__main__":
    app()

