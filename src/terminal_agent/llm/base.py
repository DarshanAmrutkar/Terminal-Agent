from __future__ import annotations
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any

from .message import Message, StreamEvent, LLMResponse

class LLMProvider(ABC):
    def __init__(self, model: str, api_key: str, max_tokens: int = 8192):
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
    
    @abstractmethod
    async def send(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        """Send messages and get a complete response."""
        ...
    
    @abstractmethod
    async def stream(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> AsyncGenerator[StreamEvent, None]:
        """Stream response events."""
        ...
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...
    
    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tool definitions for this provider's API format. Override in subclasses."""
        return tools
