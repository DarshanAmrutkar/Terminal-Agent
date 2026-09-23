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

from terminal_agent.llm.message import Message, Role, ToolCall, ToolResultContent


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
    current_context_tokens: int = 0
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
        """Update cumulative token usage counts and current context tokens."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        if input_tokens > 0:
            self.current_context_tokens = input_tokens + output_tokens

    def apply_stream_usage(
        self, 
        current_input: int, 
        current_output: int, 
        last_input: int, 
        last_output: int
    ) -> None:
        """Apply incremental streaming token deltas to cumulative and active counters.
        
        Encapsulates token accounting within Session (Information Expert principle).
        """
        delta_in = current_input - last_input
        delta_out = current_output - last_output
        self.total_input_tokens += delta_in
        self.total_output_tokens += delta_out
        if current_input > 0:
            self.current_context_tokens = current_input + current_output

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
        self.current_context_tokens = 0

    # ------------------------------------------------------------------
    # Context budget
    # ------------------------------------------------------------------

    def estimate_active_tokens(self) -> int:
        """Estimate token count of the currently active message list."""
        char_count = 0
        for m in self.messages:
            if m.content:
                char_count += len(m.content)
            if m.tool_calls:
                for tc in m.tool_calls:
                    char_count += len(tc.name) + len(json.dumps(tc.arguments))
            if m.tool_results:
                for tr in m.tool_results:
                    char_count += len(tr.output)
        return max(1, char_count // 4)

    def token_usage_ratio(self, max_context_tokens: int) -> float:
        """Return the fraction of the context budget currently used by active messages."""
        if max_context_tokens <= 0:
            return 0.0
        active_tokens = (
            self.current_context_tokens
            if self.current_context_tokens > 0
            else self.estimate_active_tokens()
        )
        return active_tokens / max_context_tokens

    def needs_compaction(self, max_context_tokens: int, threshold: float = 0.75) -> bool:
        """Return True when active context usage exceeds the given threshold."""
        return self.token_usage_ratio(max_context_tokens) >= threshold

    def compact(self) -> int:
        """Compact older tool result messages in history to reclaim context window.

        Truncates large tool outputs in messages preceding the last 4 turns.
        Returns the number of characters saved.
        """
        saved_chars = 0
        if len(self.messages) <= 4:
            return 0

        # Protect system message (index 0) and the most recent 4 messages
        for msg in self.messages[1:-4]:
            if msg.role == Role.TOOL_RESULT and msg.tool_results:
                for tr in msg.tool_results:
                    if len(tr.output) > 500:
                        original_len = len(tr.output)
                        tr.output = (
                            tr.output[:200]
                            + "\n[...prior tool output compacted to reclaim context...]\n"
                            + tr.output[-200:]
                        )
                        saved_chars += (original_len - len(tr.output))

        # Reset current context count to force re-estimation
        self.current_context_tokens = self.estimate_active_tokens()
        return saved_chars


    def to_dict(self) -> dict:
        """Serialize session to a dictionary for persistence."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "working_directory": self.working_directory,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "current_context_tokens": self.current_context_tokens,
            "active_files": sorted(list(self.active_files)),
            "messages": [
                {
                    "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
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

    @classmethod
    def from_dict(cls, data: dict) -> Session:
        """Deserialize a Session instance from a dictionary."""
        messages: list[Message] = []
        for m_data in data.get("messages", []):
            role_raw = m_data.get("role")
            try:
                role = Role(role_raw)
            except (ValueError, TypeError):
                role = Role.USER

            tool_calls = None
            if m_data.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        name=tc["name"],
                        arguments=tc.get("arguments", {}),
                    )
                    for tc in m_data["tool_calls"]
                ]

            tool_results = None
            if m_data.get("tool_results"):
                tool_results = [
                    ToolResultContent(
                        tool_call_id=tr["tool_call_id"],
                        output=tr.get("output", ""),
                        is_error=tr.get("is_error", False),
                    )
                    for tr in m_data["tool_results"]
                ]

            messages.append(
                Message(
                    role=role,
                    content=m_data.get("content"),
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                )
            )

        created_at_str = data.get("created_at")
        try:
            created_at = (
                datetime.fromisoformat(created_at_str)
                if created_at_str
                else datetime.now()
            )
        except Exception:
            created_at = datetime.now()

        session = cls(
            working_directory=data.get("working_directory", str(Path.cwd())),
            messages=messages,
            active_files=set(data.get("active_files", [])),
            session_id=data.get("session_id", datetime.now().strftime("%Y%m%d_%H%M%S")),
            created_at=created_at,
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            current_context_tokens=data.get("current_context_tokens", 0),
        )
        return session

    def save(self, path: Path) -> None:
        """Save session state to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> Session:
        """Load session state from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"Session(id={self.session_id!r}, "
            f"messages={len(self.messages)}, "
            f"tokens={self.total_input_tokens + self.total_output_tokens})"
        )
