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
        """
        Summarize older messages using the LLM to save context.
        This is a stub for where the actual LLM call would go.
        """
        if not self.needs_compaction():
            return
            
        # Example logic: keep system prompt and recent messages, summarize the middle
        if len(self.messages) > 10:
            system_msgs = [m for m in self.messages if m.get('role') == 'system']
            recent_msgs = self.messages[-5:]
            
            # The middle messages to summarize
            middle_msgs = [m for m in self.messages if m not in system_msgs and m not in recent_msgs]
            
            # Call LLM to summarize middle_msgs...
            # summary = await llm_provider.generate_summary(middle_msgs)
            summary = "Summary of previous interactions."
            
            summary_msg = {"role": "system", "content": f"Previous conversation summary: {summary}"}
            
            self.messages = system_msgs + [summary_msg] + recent_msgs

    def truncate_output(self, text: str, max_chars: int) -> str:
        """Truncate text with a marker if it exceeds max_chars."""
        if len(text) > max_chars:
            return text[:max_chars] + "\n[...truncated...]"
        return text
