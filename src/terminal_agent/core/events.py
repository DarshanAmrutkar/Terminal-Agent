"""Agent Event Bus and Observer Pattern.

Decouples domain orchestration logic from UI/presentation logic (Single Responsibility Principle).
Agent emits typed domain events (tokens, tool executions, turn transitions).
Listeners (CLI Rich renderer, WebSocket broadcasters, Langfuse loggers) subscribe to events.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from terminal_agent.llm.message import LLMResponse


@dataclass
class AgentEvent:
    """Base class for all domain events emitted by Agent."""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TurnStartEvent(AgentEvent):
    """Emitted when a new ReAct iteration starts."""
    iteration: int = 1


@dataclass
class TurnEndEvent(AgentEvent):
    """Emitted when a ReAct iteration concludes."""
    iteration: int = 1
    response: LLMResponse | None = None


@dataclass
class TextDeltaEvent(AgentEvent):
    """Emitted when the LLM streams a token/text delta."""
    text: str = ""


@dataclass
class ToolCallStartEvent(AgentEvent):
    """Emitted when the agent decides to invoke a tool."""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallEndEvent(AgentEvent):
    """Emitted when a tool finishes execution."""
    tool_name: str = ""
    output: str = ""
    is_error: bool = False


class AgentEventListener(ABC):
    """Observer interface for agent events."""

    def on_turn_start(self, event: TurnStartEvent) -> None:
        """Called when an iteration turn begins."""

    def on_turn_end(self, event: TurnEndEvent) -> None:
        """Called when an iteration turn ends."""

    def on_text_delta(self, event: TextDeltaEvent) -> None:
        """Called on each streamed token."""

    def on_tool_call_start(self, event: ToolCallStartEvent) -> None:
        """Called when a tool execution starts."""

    def on_tool_call_end(self, event: ToolCallEndEvent) -> None:
        """Called when a tool execution completes."""


class RichConsoleListener(AgentEventListener):
    """Concrete observer rendering agent events to the interactive terminal via Rich."""

    def __init__(self) -> None:
        from terminal_agent.utils.display import console
        self.console = console
        self._streaming_text = False

    def on_text_delta(self, event: TextDeltaEvent) -> None:
        from terminal_agent.utils.display import display_streaming_token
        if not self._streaming_text:
            self.console.print("\n[bold blue]🤖 Agent:[/bold blue] ", end="")
            self._streaming_text = True
        display_streaming_token(event.text)

    def on_tool_call_start(self, event: ToolCallStartEvent) -> None:
        from terminal_agent.utils.display import display_tool_call
        if self._streaming_text:
            self.console.print()
            self._streaming_text = False
        display_tool_call(event.tool_name, event.arguments)

    def on_tool_call_end(self, event: ToolCallEndEvent) -> None:
        from terminal_agent.utils.display import display_tool_result
        # Truncate preview for terminal readability
        preview = event.output[:500] + ("..." if len(event.output) > 500 else "")
        display_tool_result(event.tool_name, preview, is_error=event.is_error)

    def on_turn_end(self, event: TurnEndEvent) -> None:
        if self._streaming_text:
            self.console.print()
            self._streaming_text = False
