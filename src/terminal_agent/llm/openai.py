from __future__ import annotations
import asyncio
import json
import logging
from typing import AsyncGenerator, Any, Dict, List

import tiktoken
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, InternalServerError

from .message import Message, Role, ToolCall, StreamEvent, LLMResponse, ToolResultContent
from .base import LLMProvider

logger = logging.getLogger(__name__)

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, base_url: str | None = None, max_tokens: int = 8192, max_retries: int = 3):
        super().__init__(model, api_key, max_tokens)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = max_retries
        try:
            if "gpt-4o" in model:
                self.tokenizer = tiktoken.get_encoding("o200k_base") # The prompt says cl200k_base, but tiktoken uses o200k_base. I will use cl200k_base as requested or handle exception.
            else:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return len(text.split()) * 4 // 3

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted_tools = []
        for tool in tools:
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            })
        return formatted_tools

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                openai_messages.append({"role": "system", "content": msg.content or ""})
            elif msg.role == Role.USER:
                openai_messages.append({"role": "user", "content": msg.content or ""})
            elif msg.role == Role.ASSISTANT:
                oai_msg: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    oai_msg["content"] = msg.content
                else:
                    oai_msg["content"] = None

                if msg.tool_calls:
                    oai_tool_calls = []
                    for tc in msg.tool_calls:
                        oai_tool_calls.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                            }
                        })
                    oai_msg["tool_calls"] = oai_tool_calls
                openai_messages.append(oai_msg)
            elif msg.role == Role.TOOL_RESULT:
                if msg.tool_results:
                    for tr in msg.tool_results:
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": tr.tool_call_id,
                            "content": tr.output
                        })
        return openai_messages

    async def _with_retry(self, func, *args, **kwargs):
        retries = 0
        while True:
            try:
                return await func(*args, **kwargs)
            except (RateLimitError, APIConnectionError, InternalServerError) as e:
                retries += 1
                if retries > self.max_retries:
                    raise
                wait_time = 2 ** retries
                logger.warning(f"OpenAI API error: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    async def send(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        openai_messages = self._prepare_messages(messages)
        formatted_tools = self.format_tools(tools) if tools else []

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": openai_messages,
        }
        if formatted_tools:
            kwargs["tools"] = formatted_tools

        response = await self._with_retry(self.client.chat.completions.create, **kwargs)
        
        choice = response.choices[0]
        text_content = choice.message.content
        
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                if tc.type == "function":
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args
                    ))

        message = Message.assistant(content=text_content, tool_calls=tool_calls if tool_calls else None)
        
        finish_reason = choice.finish_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = finish_reason

        return LLMResponse(
            message=message,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            stop_reason=stop_reason
        )

    async def stream(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> AsyncGenerator[StreamEvent, None]:
        openai_messages = self._prepare_messages(messages)
        formatted_tools = self.format_tools(tools) if tools else []

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": openai_messages,
            "stream": True
        }
        if formatted_tools:
            kwargs["tools"] = formatted_tools

        async def stream_generator():
            stream = await self.client.chat.completions.create(**kwargs)
            
            current_tool_call_index: int | None = None
            current_tool_id: str = ""
            current_tool_name: str = ""
            current_tool_args: str = ""

            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                if delta.content is not None:
                    yield StreamEvent(type="text_delta", text=delta.content)
                
                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        if tc_chunk.index != current_tool_call_index:
                            if current_tool_call_index is not None:
                                try:
                                    args = json.loads(current_tool_args) if current_tool_args else {}
                                except json.JSONDecodeError:
                                    args = {}
                                completed_tc = ToolCall(id=current_tool_id, name=current_tool_name, arguments=args)
                                yield StreamEvent(type="tool_call_end", tool_call=completed_tc)
                            
                            current_tool_call_index = tc_chunk.index
                            current_tool_id = tc_chunk.id or ""
                            if tc_chunk.function and tc_chunk.function.name:
                                current_tool_name = tc_chunk.function.name
                            else:
                                current_tool_name = ""
                            current_tool_args = ""
                        
                        if tc_chunk.function and tc_chunk.function.arguments:
                            current_tool_args += tc_chunk.function.arguments
                            
            if current_tool_call_index is not None:
                try:
                    args = json.loads(current_tool_args) if current_tool_args else {}
                except json.JSONDecodeError:
                    args = {}
                completed_tc = ToolCall(id=current_tool_id, name=current_tool_name, arguments=args)
                yield StreamEvent(type="tool_call_end", tool_call=completed_tc)
                
            yield StreamEvent(type="message_end")

        try:
            async for ev in stream_generator():
                yield ev
        except Exception as e:
            logger.error(f"Error during OpenAI stream: {e}")
            raise
