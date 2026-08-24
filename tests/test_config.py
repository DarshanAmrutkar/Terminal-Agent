"""Unit tests for configuration and provider routing."""

from __future__ import annotations

import pytest
from terminal_agent.core.config import AgentConfig
from terminal_agent.core.agent import Agent
from terminal_agent.llm.openai_compatible import OpenAICompatibleProvider


def test_openrouter_config_defaults():
    config = AgentConfig(
        provider="openrouter",
        openrouter_api_key="sk-or-test-key",
    )
    assert config.provider == "openrouter"
    assert config.get_api_key() == "sk-or-test-key"
    assert config.get_base_url() == "https://openrouter.ai/api/v1"
    config.validate_api_key()  # Should not raise


def test_openrouter_missing_api_key_raises():
    config = AgentConfig(
        provider="openrouter",
        openrouter_api_key=None,
    )
    config.openrouter_api_key = None
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        config.validate_api_key()


def test_openrouter_creates_openai_compatible_provider():
    config = AgentConfig(
        provider="openrouter",
        model_name="anthropic/claude-3.5-sonnet",
        openrouter_api_key="sk-or-test-key",
    )
    agent = Agent(config)
    assert isinstance(agent.provider, OpenAICompatibleProvider)
    assert agent.provider.model == "anthropic/claude-3.5-sonnet"
    assert agent.provider.base_url == "https://openrouter.ai/api/v1"
