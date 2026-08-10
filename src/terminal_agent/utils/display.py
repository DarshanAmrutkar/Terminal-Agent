from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
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

def display_approval_prompt(command: str) -> bool:
    """Ask user y/n for command approval."""
    console.print(f"\n[bold yellow]Agent wants to run command:[/bold yellow] {command}")
    return Confirm.ask("Allow execution?", default=False)

def display_status_bar(model: str, tokens_used: int, max_tokens: int):
    """Show status info."""
    pct = (tokens_used / max_tokens) * 100 if max_tokens else 0
    status = f"Model: {model} | Context: {tokens_used}/{max_tokens} ({pct:.1f}%)"
    console.print(f"[dim]{status}[/dim]")
