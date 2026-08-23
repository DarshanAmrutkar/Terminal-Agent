"""Session state and conversation history management.

Design note: Why no separate ContextManager?
--------------------------------------------
An earlier version of this module included a `context/manager.py` that maintained
a parallel message list (as raw dicts) and a stub `compact()` method. It was never
wired up to the actual agent loop, which used Session directly.

We deleted ContextManager because:
  1. It was dead code — nothing in the agent loop called it.
  2. It maintained a duplicate message list in a different format (raw dicts vs.
     typed Message dataclasses), creating a risk of the two lists diverging.
  3. Context budget logic (does_context_need_trimming?) is a natural responsibility
     of Session, which already owns the message list and token counts.

So: context budget methods now live here, on Session. The actual trimming
logic lives in Agent, which decides *policy* (what to drop, in what order).
Session owns the *data*; Agent owns the *decisions about that data*.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from terminal_agent.llm.message import Message, Role


@dataclass
class Session:
    """Manages conversation state for an agent session.
    
    Tracks conversation history, active files, working directory,
    and session metadata.
    """

    working_directory: str
    messages: list[Message] = field(default_factory=list)
    active_files: set[str] = field(default_factory=set)
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    created_at: datetime = field(default_factory=datetime.now)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    iteration_count: int = 0

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        self.messages.append(message)

    def add_user_message(self, content: str) -> Message:
        """Create and add a user message."""
        msg = Message.user(content)
        self.messages.append(msg)
        return msg

    def add_assistant_message(self, message: Message) -> None:
        """Add an assistant response message."""
        self.messages.append(message)

    def add_tool_result_message(self, message: Message) -> None:
        """Add a tool result message."""
        self.messages.append(message)

    def get_messages(self) -> list[Message]:
        """Get all messages in the conversation."""
        return list(self.messages)

    def get_system_message(self) -> Message | None:
        """Get the system message if it exists."""
        for msg in self.messages:
            if msg.role == Role.SYSTEM:
                return msg
        return None

    def set_system_message(self, content: str) -> None:
        """Set or replace the system message.
        
        The system message is always the first message and there is only one.
        """
        # Remove any existing system message
        self.messages = [m for m in self.messages if m.role != Role.SYSTEM]
        # Insert at the beginning
        self.messages.insert(0, Message.system(content))

    def get_conversation_messages(self) -> list[Message]:
        """Get all non-system messages (for sending to LLM)."""
        return [m for m in self.messages if m.role != Role.SYSTEM]

    def track_file(self, path: str) -> None:
        """Mark a file as actively being worked on."""
        self.active_files.add(os.path.normpath(path))

    def untrack_file(self, path: str) -> None:
        """Remove a file from the active set."""
        self.active_files.discard(os.path.normpath(path))

    def update_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Update cumulative token usage counts."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def increment_iteration(self) -> int:
        """Increment and return the iteration count."""
        self.iteration_count += 1
        return self.iteration_count

    def reset_iteration_count(self) -> None:
        """Reset iteration count (called at the start of each user turn)."""
        self.iteration_count = 0

    def clear_history(self) -> None:
        """Clear conversation history, keeping the system message."""
        system_msg = self.get_system_message()
        self.messages.clear()
        if system_msg:
            self.messages.append(system_msg)
        self.active_files.clear()
        self.iteration_count = 0

    # ------------------------------------------------------------------
    # Context budget
    # ------------------------------------------------------------------

    def token_usage_ratio(self, max_context_tokens: int) -> float:
        """Return the fraction of the context budget currently used.

        This is an *estimate* based on cumulative token counts tracked during
        the session. It is approximate because:
          - We count tokens per-response, not per full message list.
          - The provider's token count includes system prompt overhead.

        Returns a float in [0.0, inf). Values > 1.0 indicate over-budget.
        """
        if max_context_tokens <= 0:
            return 0.0
        total = self.total_input_tokens + self.total_output_tokens
        return total / max_context_tokens

    def needs_compaction(self, max_context_tokens: int, threshold: float = 0.75) -> bool:
        """Return True when context usage exceeds the given threshold.

        Args:
            max_context_tokens: The provider's context window size.
            threshold: Fraction of window (0.0–1.0) at which to trigger compaction.
                       Default 0.75 means "compact when 75% full".
        """
        return self.token_usage_ratio(max_context_tokens) >= threshold

    def to_dict(self) -> dict:
        """Serialize session to a dictionary for persistence."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "working_directory": self.working_directory,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "tool_calls": (
                        [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in msg.tool_calls
                        ]
                        if msg.tool_calls
                        else None
                    ),
                    "tool_results": (
                        [
                            {
                                "tool_call_id": tr.tool_call_id,
                                "output": tr.output,
                                "is_error": tr.is_error,
                            }
                            for tr in msg.tool_results
                        ]
                        if msg.tool_results
                        else None
                    ),
                }
                for msg in self.messages
            ],
        }

    def save(self, path: Path) -> None:
        """Save session state to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"Session(id={self.session_id!r}, "
            f"messages={len(self.messages)}, "
            f"tokens={self.total_input_tokens + self.total_output_tokens})"
        )
