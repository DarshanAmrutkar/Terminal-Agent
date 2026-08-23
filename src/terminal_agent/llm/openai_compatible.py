"""OpenAI-compatible LLM provider.

Why "OpenAI-compatible" instead of a dedicated NvidiaProvider?
--------------------------------------------------------------
NVIDIA NIM, OpenAI, Groq, Together AI, Mistral, Ollama (with its OpenAI compat
layer), and many others all implement the same HTTP API contract: the OpenAI
Chat Completions API. The tool-calling format, streaming protocol, and message
structure are identical across all of them.

Building a single `OpenAICompatibleProvider` that accepts a configurable
`base_url` is therefore far superior to building per-vendor classes:

  - One codebase serves every OpenAI-compatible provider.
  - Switching providers is a config change (base_url + api_key), not a code change.
  - The abstraction is stable: the OpenAI API spec is an industry standard.
  - We avoid duplication and the "vendor class explosion" antipattern.

The only thing that varies between providers is:
  1. The base URL (e.g. https://integrate.api.nvidia.com/v1)
  2. The API key
  3. The model name
  4. Occasionally minor per-model quirks (handled at config level)

Message format differences vs. Anthropic
-----------------------------------------
Anthropic and OpenAI use different wire formats for tool calling. Our internal
`Message` type is provider-neutral; this class translates to/from the OpenAI
wire format.

  Internal Role.TOOL_RESULT  → OpenAI role "tool" (separate message per result)
  Internal ToolCall           → OpenAI tool_calls array in assistant message
  Anthropic "input_schema"   → OpenAI "parameters" in function definition

The translation is done in _prepare_messages() and format_tools().
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Any

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

from .base import LLMProvider
from .message import Message, Role, ToolCall, ToolResultContent, StreamEvent, LLMResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible API endpoint.

    Instantiate with a base_url to target different services:

        # NVIDIA NIM
        provider = OpenAICompatibleProvider(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            api_key="nvapi-...",
            base_url="https://integrate.api.nvidia.com/v1",
        )

        # OpenAI
        provider = OpenAICompatibleProvider(
            model="gpt-4o",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
        )

        # Local Ollama (with OpenAI compat)
        provider = OpenAICompatibleProvider(
            model="llama3",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int = 8192,
        max_retries: int = 3,
    ) -> None:
        super().__init__(model=model, api_key=api_key, max_tokens=max_tokens)
        self.base_url = base_url
        self.max_retries = max_retries
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def count_tokens(self, text: str) -> int:
        """Approximate token count using the common 4-chars-per-token heuristic.

        The OpenAI tiktoken library works accurately for GPT models (cl100k_base
        encoding). For NVIDIA or other models it's an approximation. We use a
        simple character-based estimate here because:
          1. For non-OpenAI models, tiktoken gives the wrong count anyway.
          2. The real token count comes from the API's usage field in each response.
          3. This method is used only as a fallback and for rough estimates.
        """
        return max(1, len(text) // 4)

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert our internal tool schema to the OpenAI function-calling format.

        Internal (Anthropic-style) format:
            {"name": "...", "description": "...", "input_schema": {...}}

        OpenAI format:
            {"type": "function", "function": {"name": "...", "description": "...",
                                               "parameters": {...}}}

        Why the difference? Anthropic and OpenAI designed their tool-calling APIs
        independently. Anthropic uses "input_schema" with JSON Schema; OpenAI
        wraps everything in a "function" object and calls it "parameters".
        The underlying JSON Schema content is the same.
        """
        formatted = []
        for tool in tools:
            # Support both our internal schema key and the pre-formatted Anthropic key
            parameters = tool.get("input_schema") or tool.get("parameters") or {
                "type": "object",
                "properties": {},
            }
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": parameters,
                },
            })
        return formatted

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Translate our internal Message objects to the OpenAI wire format.

        Key differences from Anthropic format:
          - System message is just role="system" in the message list (not a
            separate API parameter like Anthropic's system= field).
          - Tool results use role="tool" with tool_call_id (not role="user"
            with embedded tool_result blocks like Anthropic).
          - Each tool result is a separate message (vs. Anthropic batching them).
          - Assistant tool calls are in a tool_calls array on the message.
        """
        openai_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                openai_messages.append({
                    "role": "system",
                    "content": msg.content or "",
                })

            elif msg.role == Role.USER:
                openai_messages.append({
                    "role": "user",
                    "content": msg.content or "",
                })

            elif msg.role == Role.ASSISTANT:
                assistant_msg: dict[str, Any] = {"role": "assistant"}

                if msg.content:
                    assistant_msg["content"] = msg.content
                else:
                    # OpenAI requires content to be null (not missing) when there are tool_calls
                    assistant_msg["content"] = None

                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                openai_messages.append(assistant_msg)

            elif msg.role == Role.TOOL_RESULT:
                # OpenAI represents each tool result as a separate message
                # with role="tool", unlike Anthropic which batches them in
                # a single role="user" message.
                if msg.tool_results:
                    for tr in msg.tool_results:
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": tr.tool_call_id,
                            "content": tr.output,
                        })

        return openai_messages

    async def send(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a non-streaming request and return the complete response."""
        openai_messages = self._prepare_messages(messages)
        formatted_tools = self.format_tools(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": openai_messages,
        }
        if formatted_tools:
            kwargs["tools"] = formatted_tools
            kwargs["tool_choice"] = "auto"

        retries = 0
        while True:
            try:
                response = await self.client.chat.completions.create(**kwargs)
                break
            except (APIConnectionError, RateLimitError, APIError) as e:
                retries += 1
                if retries > self.max_retries:
                    raise
                import asyncio
                wait = 2 ** retries
                logger.warning(f"OpenAI-compatible API error: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)

        choice = response.choices[0]
        message_obj = choice.message

        # Extract text content
        content = message_obj.content or None

        # Extract tool calls
        tool_calls: list[ToolCall] = []
        if message_obj.tool_calls:
            for tc in message_obj.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                ))

        msg = Message.assistant(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )

        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        return LLMResponse(
            message=msg,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason="tool_use" if tool_calls else choice.finish_reason,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response, yielding StreamEvents for real-time display.

        Uses the low-level create(stream=True) approach rather than the
        high-level .stream() context manager. The context manager is an OpenAI
        Python SDK helper that adds extra abstraction but isn't available on
        all OpenAI-compatible endpoints. The raw SSE iterator is universally
        supported.

        Tool call arguments arrive in chunks via delta.tool_calls[].function.arguments.
        We accumulate partial JSON across chunks, then emit tool_call_end with
        the fully-parsed arguments dict.
        """
        openai_messages = self._prepare_messages(messages)
        formatted_tools = self.format_tools(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if formatted_tools:
            kwargs["tools"] = formatted_tools
            kwargs["tool_choice"] = "auto"

        # Accumulate tool call arguments across chunks.
        # OpenAI sends partial JSON in multiple deltas keyed by index.
        # Structure: {index: {"id": str, "name": str, "arguments": str}}
        tool_call_accumulators: dict[int, dict[str, str]] = {}

        try:
            # create() with stream=True returns an async iterator of raw chunks.
            stream_iter = await self.client.chat.completions.create(**kwargs)

            async for chunk in stream_iter:
                # --- Usage (may appear in any chunk, typically the last one) ---
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    yield StreamEvent(
                        type="usage",
                        usage={
                            "input_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                            "output_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                        },
                    )

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # --- Text delta ---
                if delta.content:
                    yield StreamEvent(type="text_delta", text=delta.content)

                # --- Tool call delta ---
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index

                        if idx not in tool_call_accumulators:
                            # First chunk for this tool call — emit start event.
                            tool_call_accumulators[idx] = {
                                "id": tc_delta.id or "",
                                "name": (tc_delta.function.name if tc_delta.function else "") or "",
                                "arguments": "",
                            }
                            yield StreamEvent(
                                type="tool_call_start",
                                tool_call=ToolCall(
                                    id=tool_call_accumulators[idx]["id"],
                                    name=tool_call_accumulators[idx]["name"],
                                    arguments={},
                                ),
                            )
                        else:
                            # Update id/name if they arrive in later chunks
                            if tc_delta.id:
                                tool_call_accumulators[idx]["id"] = tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                tool_call_accumulators[idx]["name"] = tc_delta.function.name

                        # Accumulate partial arguments JSON.
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_call_accumulators[idx]["arguments"] += tc_delta.function.arguments
                            yield StreamEvent(
                                type="tool_call_delta",
                                text=tc_delta.function.arguments,
                            )

                # --- Stream end ---
                if choice.finish_reason is not None:
                    # Finalize all accumulated tool calls
                    for acc in tool_call_accumulators.values():
                        try:
                            arguments = json.loads(acc["arguments"]) if acc["arguments"] else {}
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse tool call arguments for "
                                f"{acc['name']!r}: {acc['arguments']!r}"
                            )
                            arguments = {}

                        yield StreamEvent(
                            type="tool_call_end",
                            tool_call=ToolCall(
                                id=acc["id"],
                                name=acc["name"],
                                arguments=arguments,
                            ),
                        )

                    tool_call_accumulators.clear()
                    yield StreamEvent(type="message_end")

        except (APIConnectionError, RateLimitError, APIError) as e:
            logger.error(f"OpenAI-compatible stream error: {e}")
            raise
