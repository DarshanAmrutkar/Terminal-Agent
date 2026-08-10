from __future__ import annotations
import asyncio
import json
import logging
from typing import AsyncGenerator, Any

import tiktoken
from anthropic import AsyncAnthropic, APIError, APIConnectionError, RateLimitError

from .message import Message, Role, ToolCall, ToolResultContent, StreamEvent, LLMResponse
from .base import LLMProvider

logger = logging.getLogger(__name__)

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, max_tokens: int = 8192, max_retries: int = 3):
        super().__init__(model, api_key, max_tokens)
        self.client = AsyncAnthropic(api_key=api_key)
        self.max_retries = max_retries
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback if cl100k_base is not available for some reason
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # fallback simple approximation
        return len(text.split()) * 4 // 3

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted_tools = []
        for tool in tools:
            formatted_tool = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", tool.get("input_schema", {"type": "object", "properties": {}}))
            }
            formatted_tools.append(formatted_tool)
        return formatted_tools

    def _prepare_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system_content = ""
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_content += (msg.content or "") + "\n"
            elif msg.role == Role.USER:
                anthropic_messages.append({"role": "user", "content": msg.content or ""})
            elif msg.role == Role.ASSISTANT:
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments
                        })
                
                # Anthropic doesn't allow empty content for assistant
                if not content:
                    content.append({"type": "text", "text": " "})
                    
                anthropic_messages.append({"role": "assistant", "content": content})
            elif msg.role == Role.TOOL_RESULT:
                content: list[dict[str, Any]] = []
                if msg.tool_results:
                    for tr in msg.tool_results:
                        content.append({
                            "type": "tool_result",
                            "tool_use_id": tr.tool_call_id,
                            "content": tr.output,
                            "is_error": tr.is_error
                        })
                anthropic_messages.append({"role": "user", "content": content})

        return system_content.strip(), anthropic_messages

    async def _with_retry(self, func, *args, **kwargs):
        retries = 0
        while True:
            try:
                return await func(*args, **kwargs)
            except (APIConnectionError, RateLimitError, APIError) as e:
                retries += 1
                if retries > self.max_retries:
                    raise
                wait_time = 2 ** retries
                logger.warning(f"Anthropic API error: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    async def send(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        system, anthropic_messages = self._prepare_messages(messages)
        formatted_tools = self.format_tools(tools) if tools else []
        
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
            "system": system,
        }
        if formatted_tools:
            kwargs["tools"] = formatted_tools

        response = await self._with_retry(self.client.messages.create, **kwargs)
        
        text_content = []
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                text_content.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input
                ))
                
        content_str = "\n".join(text_content) if text_content else None
        message = Message.assistant(content=content_str, tool_calls=tool_calls if tool_calls else None)
        
        stop_reason = response.stop_reason
        if stop_reason == "tool_use":
            pass # remains tool_use
        
        return LLMResponse(
            message=message,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=stop_reason
        )

    async def stream(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> AsyncGenerator[StreamEvent, None]:
        system, anthropic_messages = self._prepare_messages(messages)
        formatted_tools = self.format_tools(tools) if tools else []
        
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
            "system": system,
        }
        if formatted_tools:
            kwargs["tools"] = formatted_tools

        async def stream_generator():
            async with self.client.messages.stream(**kwargs) as stream:
                current_tool_call_id = None
                current_tool_name = None
                current_tool_input_json = ""
                
                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "text":
                            yield StreamEvent(type="text_delta")
                        elif event.content_block.type == "tool_use":
                            current_tool_call_id = event.content_block.id
                            current_tool_name = event.content_block.name
                            current_tool_input_json = ""
                            tc = ToolCall(id=current_tool_call_id, name=current_tool_name, arguments={})
                            yield StreamEvent(type="tool_call_start", tool_call=tc)
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield StreamEvent(type="text_delta", text=event.delta.text)
                        elif event.delta.type == "input_json_delta":
                            current_tool_input_json += event.delta.partial_json
                            yield StreamEvent(type="tool_call_delta", text=event.delta.partial_json)
                    elif event.type == "content_block_stop":
                        if current_tool_call_id is not None:
                            try:
                                args = json.loads(current_tool_input_json) if current_tool_input_json else {}
                            except json.JSONDecodeError:
                                args = {}
                            tc = ToolCall(id=current_tool_call_id, name=current_tool_name, arguments=args)
                            yield StreamEvent(type="tool_call_end", tool_call=tc)
                            current_tool_call_id = None
                            current_tool_name = None
                            current_tool_input_json = ""
                    elif event.type == "message_stop":
                        yield StreamEvent(type="message_end")
                        
        # _with_retry with streaming is complex, simple approach here is just retry stream opening if fails, 
        # but AsyncAnthropic's stream handles some internal retries. For a custom generator, let's wrap it lightly.
        # However, to keep it simple and given _with_retry on generator is tricky, we can just return the stream directly
        # and let the caller handle stream interruptions, or wrap initialization.
        
        # Here we just iterate
        try:
            async for ev in stream_generator():
                yield ev
        except (APIConnectionError, RateLimitError, APIError) as e:
            logger.error(f"Error during stream: {e}")
            raise
