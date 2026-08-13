from typing import List, Dict, Any, Tuple
from terminal_agent.context.tokenizer import count_messages_tokens

class ContextManager:
    def __init__(self, max_context_tokens: int = 100000, summarization_threshold: float = 0.75):
        self.max_context_tokens = max_context_tokens
        self.summarization_threshold = summarization_threshold
        self.messages: List[Dict[str, Any]] = []

    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the history."""
        self.messages.append(message)

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get the current message history."""
        return self.messages

    def get_token_usage(self) -> Tuple[int, int]:
        """Return (used_tokens, max_tokens)."""
        used = count_messages_tokens(self.messages)
        return used, self.max_context_tokens

    def needs_compaction(self) -> bool:
        """Check if context size is over the summarization threshold."""
        used, max_tokens = self.get_token_usage()
        return (used / max_tokens) >= self.summarization_threshold

    async def compact(self, llm_provider: Any) -> None:
        if not self.needs_compaction():
            return
        if len(self.messages) <= 8:
            return  # Not enough messages to compact
        
        # Keep system message(s) and last 6 messages
        system_msgs = [m for m in self.messages if m.get('role') == 'system']
        recent_msgs = self.messages[-6:]
        middle_msgs = [m for m in self.messages if m not in system_msgs and m not in recent_msgs]
        
        if not middle_msgs:
            return
        
        # Build summarization prompt
        summary_prompt = self._build_summary_prompt(middle_msgs)
        
        # Call LLM for summarization
        from terminal_agent.llm.message import Message
        summary_messages = [Message.user(summary_prompt)]
        try:
            response = await llm_provider.send(summary_messages)
            summary_text = response.message.content or "Previous conversation summary unavailable."
        except Exception:
            summary_text = "Previous conversation summary unavailable."
        
        summary_msg = {"role": "system", "content": f"[Conversation Summary]\n{summary_text}"}
        self.messages = system_msgs + [summary_msg] + recent_msgs

    def _build_summary_prompt(self, messages: list) -> str:
        lines = ["Summarize the following conversation concisely. Focus on:",
                 "- What the user asked for",
                 "- What files were read or modified", 
                 "- What commands were run and their results",
                 "- Key decisions made",
                 "- Current state of the task\n"]
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if content:
                lines.append(f"{role}: {content[:500]}")
        return "\n".join(lines)

    def truncate_output(self, text: str, max_chars: int) -> str:
        """Truncate text with a marker if it exceeds max_chars."""
        if len(text) > max_chars:
            return text[:max_chars] + "\n[...truncated...]"
        return text
