from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.text import Text

console = Console()

def display_welcome():
    """Show welcome banner with version info."""
    welcome_text = Text("Terminal Agent v0.1.0", style="bold cyan")
    welcome_text.append("\nYour CLI-based AI coding assistant.", style="italic white")
    console.print(Panel(welcome_text, title="🚀 Welcome", border_style="cyan"))

def display_tool_call(name: str, args: dict):
    """Show tool invocation in a panel."""
    args_str = "\n".join(f"{k}: {v}" for k, v in args.items())
    console.print(Panel(args_str, title=f"🔧 Tool: {name}", border_style="yellow"))

def display_tool_result(name: str, result: str, is_error: bool = False):
    """Show tool result."""
    style = "red" if is_error else "green"
    title = f"❌ Error: {name}" if is_error else f"✅ Result: {name}"
    console.print(Panel(result, title=title, border_style=style))

def display_agent_message(text: str):
    """Render markdown response from the agent."""
    console.print(Panel(Markdown(text), title="🤖 Agent", border_style="blue"))

def display_streaming_token(token: str):
    """Print token without newline for streaming."""
    console.print(token, end="")

def display_approval_prompt(reason: str) -> bool:
    """Ask the user to approve a tool action before it executes.

    Args:
        reason: A human-readable description of what the agent wants to do
                (from PermissionDecision.reason).
    """
    console.print(f"\n[bold yellow]⚠ Approval required:[/bold yellow] {reason}")
    return Confirm.ask("Allow this action?", default=False)

def display_status_bar(
    model: str,
    tokens_used: int,
    max_tokens: int,
    cumulative_tokens: int | None = None,
    session_cost: float | None = None,
) -> None:
    """Show status info separating active context from cumulative session traffic.

    Args:
        model: Active model identifier.
        tokens_used: Active context tokens in current conversation window.
        max_tokens: Maximum allowed context window for the model.
        cumulative_tokens: Optional cumulative billed tokens across all turns in this session.
        session_cost: Optional total dollar cost for this session.
    """
    pct = (tokens_used / max_tokens) * 100 if max_tokens else 0
    parts = [
        f"Model: {model}",
        f"Context Window: {tokens_used:,}/{max_tokens:,} ({pct:.1f}%)",
    ]
    if cumulative_tokens is not None and cumulative_tokens > 0:
        parts.append(f"Session Traffic: {cumulative_tokens:,} tokens")
    if session_cost is not None and session_cost > 0:
        parts.append(f"Cost: ${session_cost:.4f}")

    status = " | ".join(parts)
    console.print(f"[dim]{status}[/dim]")
