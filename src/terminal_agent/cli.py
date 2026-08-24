"""CLI interface for Terminal Agent.

Provides the interactive terminal experience with rich output,
streaming responses, command history, and slash commands.
"""

from __future__ import annotations

import asyncio
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from terminal_agent.core.agent import Agent
from terminal_agent.core.config import AgentConfig
from terminal_agent.llm.profiles import list_profiles
from terminal_agent.utils.display import (
    console,
    display_welcome,
    display_status_bar,
)
from terminal_agent.utils.cost import CostTracker


def _get_history_path() -> str:
    """Get the path for command history file."""
    from pathlib import Path
    history_dir = Path.home() / ".terminal-agent"
    history_dir.mkdir(exist_ok=True)
    return str(history_dir / "history.txt")


async def run_interactive(config: AgentConfig | None = None) -> None:
    """Run the interactive CLI session.
    
    This is the main interactive loop that:
    1. Displays the welcome banner
    2. Accepts user input via prompt_toolkit
    3. Passes input to the agent for processing
    4. Handles slash commands
    5. Loops until the user exits
    """
    if config is None:
        config = AgentConfig()

    # Display welcome
    display_welcome()
    console.print(f"[dim]Provider: {config.provider} | Model: {config.model_name}[/dim]")
    console.print(f"[dim]Permission mode: {config.permission_mode.value}[/dim]")
    console.print()

    # Initialize the agent
    try:
        agent = Agent(config)
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        console.print(
            "[dim]Set your API key: export ANTHROPIC_API_KEY=your-key-here[/dim]"
        )
        return

    # Set up prompt with history
    prompt_session: PromptSession = PromptSession(
        history=FileHistory(_get_history_path()),
        auto_suggest=AutoSuggestFromHistory(),
    )

    console.print("[dim]Type your message to start. Use /help for commands, /exit to quit.[/dim]\n")

    while True:
        try:
            # Get user input
            user_input = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: prompt_session.prompt(
                    "You: ",
                    multiline=False,
                ),
            )

            # Skip empty input
            if not user_input.strip():
                continue

            # Handle slash commands
            if user_input.strip().startswith("/"):
                should_continue = _handle_slash_command(
                    user_input.strip(), agent, config
                )
                if not should_continue:
                    break
                continue

            # Process through the agent
            console.print()
            await agent.process_message(user_input)
            console.print()

            # Show status after each turn
            display_status_bar(
                config.model_name,
                agent.session.total_input_tokens + agent.session.total_output_tokens,
                config.max_context_tokens,
            )
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Use /exit to quit.[/dim]")
            continue
        except EOFError:
            _goodbye(agent)
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            console.print("[dim]The agent encountered an error. You can continue or /exit.[/dim]\n")


def _handle_slash_command(command: str, agent: Agent, config: AgentConfig) -> bool:
    """Handle a slash command. Returns False if the session should end."""
    cmd = command.lower().split()[0]
    args = command[len(cmd):].strip()

    if cmd in ("/exit", "/quit", "/q"):
        _goodbye(agent)
        return False

    elif cmd == "/help":
        help_text = Text()
        help_text.append("/help", style="bold cyan")
        help_text.append("             — Show this help message\n")
        help_text.append("/clear", style="bold cyan")
        help_text.append("            — Clear conversation history\n")
        help_text.append("/cost", style="bold cyan")
        help_text.append("             — Show token usage and estimated cost\n")
        help_text.append("/status", style="bold cyan")
        help_text.append("           — Show current session status\n")
        help_text.append("/model", style="bold cyan")
        help_text.append("            — Show active model & list available profiles\n")
        help_text.append("/model <name>", style="bold cyan")
        help_text.append("     — Switch active model profile (e.g. /model deepseek)\n")
        help_text.append("/exit", style="bold cyan")
        help_text.append("             — Exit the agent")
        console.print(Panel(help_text, title="📖 Commands", border_style="cyan"))

    elif cmd == "/clear":
        agent.session.clear_history()
        console.print("[green]✓ Conversation history cleared.[/green]")

    elif cmd == "/cost":
        cost_summary = agent.cost_tracker.format_cost_summary()
        console.print(Panel(cost_summary, title="💰 Cost Summary", border_style="green"))

    elif cmd == "/status":
        status_lines = [
            f"Session ID: {agent.session.session_id}",
            f"Working Dir: {agent.working_dir}",
            f"Messages: {len(agent.session.messages)}",
            f"Active Files: {len(agent.session.active_files)}",
            f"Total Tokens: {agent.session.total_input_tokens + agent.session.total_output_tokens:,}",
        ]
        console.print(
            Panel("\n".join(status_lines), title="📊 Session Status", border_style="blue")
        )

    elif cmd in ("/model", "/profile"):
        if args:
            # Switch profile
            target_profile = args.strip()
            try:
                activated = agent.switch_model_profile(target_profile)
                console.print(
                    f"[bold green]✓ Switched to profile '{activated.name}'[/bold green] "
                    f"([dim]{activated.provider} / {activated.model_name}[/dim])\n"
                )
            except KeyError as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
            except ValueError as e:
                console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        else:
            # Show current model info & list available profiles
            active_profile_name = config.profile or config.provider
            info_lines = [
                f"[bold]Active Profile:[/bold] {active_profile_name}",
                f"[bold]Provider:[/bold] {config.provider}",
                f"[bold]Model:[/bold] {config.model_name}",
                f"[bold]Max Output Tokens:[/bold] {config.max_tokens:,}",
                f"[bold]Context Budget:[/bold] {config.max_context_tokens:,}",
                "",
                "[bold cyan]Available Profiles:[/bold cyan] (switch with `/model <name>`):",
            ]
            for name, prof in list_profiles().items():
                is_active = (config.provider == prof.provider and config.model_name == prof.model_name)
                marker = " [bold green]● (active)[/bold green]" if is_active else ""
                desc = f" — {prof.description}" if prof.description else ""
                info_lines.append(f"  • [bold magenta]{name}[/bold magenta] ({prof.provider} / {prof.model_name}){marker}{desc}")

            console.print(
                Panel("\n".join(info_lines), title="🤖 Model Profiles", border_style="magenta")
            )

    else:
        console.print(f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/yellow]")

    return True


def _goodbye(agent: Agent) -> None:
    """Display goodbye message with session summary."""
    console.print()
    cost = agent.cost_tracker.format_cost_summary()
    console.print(
        Panel(
            f"Thanks for using Terminal Agent!\n\n{cost}",
            title="👋 Goodbye",
            border_style="cyan",
        )
    )
