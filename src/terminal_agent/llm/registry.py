"""Provider Registry — extensible factory for LLM providers.

Resolves the Open-Closed Principle (OCP) violation by decoupling provider creation
from the core Agent class. New providers (e.g. Gemini, Ollama, Bedrock) can be
registered without modifying Agent internals.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from terminal_agent.llm.base import LLMProvider

if TYPE_CHECKING:
    from terminal_agent.core.config import AgentConfig

ProviderFactory = Callable[["AgentConfig"], LLMProvider]


class ProviderRegistry:
    """Registry and factory for LLM providers."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        """Register a factory for a given provider name."""
        self._factories[name.lower()] = factory

    def create(self, provider_name: str, config: AgentConfig) -> LLMProvider:
        """Instantiate the LLM provider using the registered factory."""
        key = provider_name.lower()
        factory = self._factories.get(key)
        if factory is None:
            supported = ", ".join(sorted(self._factories.keys()))
            raise ValueError(
                f"Unsupported provider: {provider_name!r}. "
                f"Supported providers: {supported}"
            )
        return factory(config)

    def is_registered(self, name: str) -> bool:
        """Check if a provider name is registered."""
        return name.lower() in self._factories

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return sorted(self._factories.keys())


# Global default provider registry
default_provider_registry = ProviderRegistry()


# Default factories
def _create_anthropic_provider(config: AgentConfig) -> LLMProvider:
    from terminal_agent.llm.anthropic import AnthropicProvider
    return AnthropicProvider(
        model=config.model_name,
        api_key=config.get_api_key("anthropic"),
        max_tokens=config.max_tokens,
    )


def _create_openai_compatible_provider(config: AgentConfig) -> LLMProvider:
    from terminal_agent.llm.openai_compatible import OpenAICompatibleProvider
    return OpenAICompatibleProvider(
        model=config.model_name,
        api_key=config.get_api_key(config.provider),
        base_url=config.get_base_url(),
        max_tokens=config.max_tokens,
    )


default_provider_registry.register("anthropic", _create_anthropic_provider)
default_provider_registry.register("openai", _create_openai_compatible_provider)
default_provider_registry.register("nvidia", _create_openai_compatible_provider)
default_provider_registry.register("openrouter", _create_openai_compatible_provider)
