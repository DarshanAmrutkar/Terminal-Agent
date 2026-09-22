"""Provider Failover and Circuit Breaker implementation.

Wraps a primary LLM provider and fallback provider(s) with an automated
circuit breaker that diverts traffic upon rate limits (429) or service outages (5xx/529).
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, AsyncGenerator, Sequence

from terminal_agent.llm.base import LLMProvider
from terminal_agent.llm.message import Message, StreamEvent, LLMResponse

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"        # Normal operation: routing to primary
    OPEN = "open"            # Tripped: routing to fallback provider
    HALF_OPEN = "half_open"  # Probing: testing primary after cooldown


# Error strings and exception patterns that trigger automated failover
RETRYABLE_ERROR_INDICATORS = (
    "429",
    "rate limit",
    "ratelimit",
    "quota exceeded",
    "500",
    "502",
    "503",
    "504",
    "529",
    "overloaded",
    "service unavailable",
    "internal server error",
    "connection error",
    "timeout",
)


def is_failover_trigger(error: Exception) -> bool:
    """Determine if an exception warrants tripping the circuit breaker and failing over."""
    err_msg = str(error).lower()
    err_type = type(error).__name__.lower()
    
    # Check error message strings
    if any(indicator in err_msg for indicator in RETRYABLE_ERROR_INDICATORS):
        return True
        
    # Check exception class names (e.g. RateLimitError, APIConnectionError, InternalServerError)
    if any(indicator.replace(" ", "") in err_type for indicator in ("ratelimit", "apiconnection", "internalserver", "serviceunavailable")):
        return True

    # Check for HTTP status codes on the exception object if present
    status_code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status_code in (429, 500, 502, 503, 504, 529):
        return True

    return False


class FallbackProvider(LLMProvider):
    """An LLM provider proxy implementing the Circuit Breaker pattern.

    Routes requests to a primary provider, automatically switching to a list of
    fallback providers when encountering rate limits or server errors.
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: Sequence[LLMProvider],
        failure_threshold: int = 1,
        cooldown_seconds: float = 60.0,
    ) -> None:
        super().__init__(
            model=primary.model,
            api_key=primary.api_key,
            max_tokens=primary.max_tokens,
        )
        if not fallbacks:
            raise ValueError("FallbackProvider requires at least one fallback provider")

        self.primary = primary
        self.fallbacks = list(fallbacks)
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self._state = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._last_state_change: float = 0.0
        self._active_fallback_index: int = 0
        self._total_failovers: int = 0

    @property
    def circuit_state(self) -> CircuitState:
        """Current state of the circuit breaker."""
        if self._state == CircuitState.OPEN:
            # Check if cooldown has elapsed to enter HALF_OPEN
            if time.time() - self._last_state_change >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def active_provider(self) -> LLMProvider:
        """The provider currently receiving traffic."""
        state = self.circuit_state
        if state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
            return self.primary
        return self.fallbacks[self._active_fallback_index]

    @property
    def total_failovers(self) -> int:
        """Total number of failover events recorded."""
        return self._total_failovers

    def reset_circuit(self) -> None:
        """Manually reset the circuit breaker back to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._active_fallback_index = 0
        self._last_state_change = time.time()
        logger.info(f"Circuit breaker manually reset to CLOSED. Primary: {self.primary.model}")

    def _record_success(self) -> None:
        """Record a successful response."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Primary probe succeeded. Circuit breaker reset to CLOSED.")
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._active_fallback_index = 0

    def _record_failure(self, error: Exception) -> None:
        """Record an error and evaluate whether to trip circuit."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._trip(error)

    def _trip(self, reason: Exception) -> None:
        """Trip circuit breaker to OPEN state and switch to fallback."""
        self._state = CircuitState.OPEN
        self._last_state_change = time.time()
        self._total_failovers += 1
        active_fb = self.fallbacks[self._active_fallback_index]
        logger.warning(
            f"Circuit breaker TRIPPED! Reason: {reason}. Switching to fallback provider: {active_fb.model}"
        )

    async def send(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        """Send messages with circuit-breaker protection and automated fallback."""
        state = self.circuit_state

        if state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
            try:
                response = await self.primary.send(messages, tools=tools)
                self._record_success()
                return response
            except Exception as e:
                if is_failover_trigger(e):
                    self._record_failure(e)
                    # Proceed to fallback attempt below
                else:
                    raise

        # Try fallbacks in order
        for idx in range(self._active_fallback_index, len(self.fallbacks)):
            fallback = self.fallbacks[idx]
            try:
                logger.info(f"Executing request with fallback provider: {fallback.model}")
                return await fallback.send(messages, tools=tools)
            except Exception as e:
                logger.error(f"Fallback provider {fallback.model} failed: {e}")
                if idx < len(self.fallbacks) - 1:
                    self._active_fallback_index = idx + 1
                    continue
                raise

        raise RuntimeError("All providers (primary and fallbacks) failed")

    async def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream response events with circuit-breaker protection and automated fallback."""
        state = self.circuit_state

        if state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
            try:
                yielded_any = False
                async for event in self.primary.stream(messages, tools=tools):
                    yielded_any = True
                    yield event
                self._record_success()
                return
            except Exception as e:
                # If stream already started yielding tokens to user, we cannot seamlessly restart
                if yielded_any or not is_failover_trigger(e):
                    raise
                self._record_failure(e)

        # Fallback streaming
        for idx in range(self._active_fallback_index, len(self.fallbacks)):
            fallback = self.fallbacks[idx]
            try:
                logger.info(f"Streaming from fallback provider: {fallback.model}")
                async for event in fallback.stream(messages, tools=tools):
                    yield event
                return
            except Exception as e:
                logger.error(f"Fallback stream from {fallback.model} failed: {e}")
                if idx < len(self.fallbacks) - 1:
                    self._active_fallback_index = idx + 1
                    continue
                raise

    def count_tokens(self, text: str) -> int:
        return self.active_provider.count_tokens(text)

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.active_provider.format_tools(tools)
