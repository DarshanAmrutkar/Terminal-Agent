"""Tests for LLM FallbackProvider and automated circuit breaker."""

import pytest
from typing import AsyncGenerator

from terminal_agent.llm.base import LLMProvider
from terminal_agent.llm.fallback import FallbackProvider, CircuitState, is_failover_trigger
from terminal_agent.llm.message import Message, StreamEvent, LLMResponse


class MockProvider(LLMProvider):
    def __init__(self, name: str, should_fail: bool = False, fail_exception: Exception | None = None):
        super().__init__(model=name, api_key="test-key")
        self.should_fail = should_fail
        self.fail_exception = fail_exception or RuntimeError("429 Too Many Requests: Rate limit exceeded")
        self.call_count = 0

    async def send(self, messages, tools=None):
        self.call_count += 1
        if self.should_fail:
            raise self.fail_exception
        return LLMResponse(message=Message.assistant(content=f"Response from {self.model}"))

    async def stream(self, messages, tools=None) -> AsyncGenerator[StreamEvent, None]:
        self.call_count += 1
        if self.should_fail:
            raise self.fail_exception
        yield StreamEvent(type="text_delta", text=f"Stream from {self.model}")
        yield StreamEvent(type="message_end")

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_is_failover_trigger():
    """Verify that rate limits, timeouts, and outages trigger failover."""
    assert is_failover_trigger(RuntimeError("429 Rate limit exceeded")) is True
    assert is_failover_trigger(RuntimeError("503 Service Unavailable")) is True
    assert is_failover_trigger(RuntimeError("529 Overloaded")) is True
    assert is_failover_trigger(RuntimeError("Connection error")) is True
    assert is_failover_trigger(ValueError("Invalid syntax")) is False


@pytest.mark.asyncio
async def test_fallback_provider_normal_execution():
    """Verify that when primary succeeds, circuit stays closed and fallback is not called."""
    primary = MockProvider(name="primary-sonnet", should_fail=False)
    secondary = MockProvider(name="secondary-gpt4o", should_fail=False)
    fallback_p = FallbackProvider(primary=primary, fallbacks=[secondary])

    assert fallback_p.circuit_state == CircuitState.CLOSED
    assert fallback_p.active_provider == primary

    resp = await fallback_p.send([Message.user("Hello")])
    assert "primary-sonnet" in resp.message.content
    assert primary.call_count == 1
    assert secondary.call_count == 0
    assert fallback_p.circuit_state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_fallback_provider_auto_failover_on_429():
    """Verify automated failover to secondary provider when primary encounters 429 rate limit."""
    primary = MockProvider(name="primary-anthropic", should_fail=True)
    secondary = MockProvider(name="secondary-openrouter", should_fail=False)
    fallback_p = FallbackProvider(primary=primary, fallbacks=[secondary], failure_threshold=1)

    resp = await fallback_p.send([Message.user("Hello")])
    assert "secondary-openrouter" in resp.message.content
    assert primary.call_count == 1
    assert secondary.call_count == 1
    assert fallback_p.circuit_state == CircuitState.OPEN
    assert fallback_p.total_failovers == 1
    assert fallback_p.active_provider == secondary

    # Next call should go straight to fallback without re-attempting failed primary
    resp2 = await fallback_p.send([Message.user("Second question")])
    assert "secondary-openrouter" in resp2.message.content
    assert primary.call_count == 1  # Unchanged
    assert secondary.call_count == 2


@pytest.mark.asyncio
async def test_fallback_provider_stream_failover():
    """Verify automated stream failover when primary fails on stream initiation."""
    primary = MockProvider(name="primary-stream", should_fail=True)
    secondary = MockProvider(name="secondary-stream", should_fail=False)
    fallback_p = FallbackProvider(primary=primary, fallbacks=[secondary])

    events = []
    async for event in fallback_p.stream([Message.user("Stream test")]):
        events.append(event)

    text = "".join(e.text for e in events if e.type == "text_delta")
    assert "secondary-stream" in text
    assert fallback_p.circuit_state == CircuitState.OPEN


def test_fallback_provider_manual_reset():
    """Verify manual circuit reset restores primary provider as active."""
    primary = MockProvider(name="primary", should_fail=False)
    secondary = MockProvider(name="secondary", should_fail=False)
    fallback_p = FallbackProvider(primary=primary, fallbacks=[secondary])

    fallback_p._trip(RuntimeError("Simulated trip"))
    assert fallback_p.circuit_state == CircuitState.OPEN
    assert fallback_p.active_provider == secondary

    fallback_p.reset_circuit()
    assert fallback_p.circuit_state == CircuitState.CLOSED
    assert fallback_p.active_provider == primary
