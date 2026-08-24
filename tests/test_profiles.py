"""Unit tests for ModelProfile, ProfileRegistry, and dynamic profile switching."""

from __future__ import annotations

import pytest
from terminal_agent.core.config import AgentConfig
from terminal_agent.core.agent import Agent
from terminal_agent.llm.profiles import (
    ModelProfile,
    ProfileRegistry,
    DEFAULT_PROFILES,
    register_profile,
    get_profile,
    list_profiles,
    remove_profile,
)
from terminal_agent.llm.openai_compatible import OpenAICompatibleProvider
from terminal_agent.llm.anthropic import AnthropicProvider


def test_default_profiles_exist():
    assert "sonnet" in DEFAULT_PROFILES
    assert "nemotron" in DEFAULT_PROFILES
    assert "gpt4o" in DEFAULT_PROFILES
    assert "deepseek" in DEFAULT_PROFILES
    assert "deepseek-r1" in DEFAULT_PROFILES
    assert "openrouter-claude" in DEFAULT_PROFILES


def test_profile_resolution_from_preset():
    config = AgentConfig(
        profile="nemotron",
        nvidia_api_key="nvapi-test-key",
    )
    profile = config.get_model_profile()
    assert profile.name == "nemotron"
    assert profile.provider == "nvidia"
    assert profile.model_name == "nvidia/nemotron-3-ultra-550b-a55b"
    assert profile.max_tokens == 15000
    assert profile.context_window == 500000


def test_custom_profile_adhoc():
    config = AgentConfig(
        provider="openrouter",
        model_name="mistralai/mistral-large",
        openrouter_api_key="sk-or-test",
        max_tokens=4096,
        max_context_tokens=32000,
    )
    profile = config.get_model_profile()
    assert profile.provider == "openrouter"
    assert profile.model_name == "mistralai/mistral-large"
    assert profile.max_tokens == 4096
    assert profile.context_window == 32000


def test_register_and_remove_profile():
    # Register custom profile
    custom = register_profile(
        name="qwen-coder",
        provider="openrouter",
        model_name="qwen/qwen-2.5-coder-32b-instruct",
        max_tokens=8192,
        context_window=32768,
        description="Qwen 2.5 Coder 32B",
    )
    assert get_profile("qwen-coder").model_name == "qwen/qwen-2.5-coder-32b-instruct"
    assert "qwen-coder" in list_profiles()

    # Remove profile
    remove_profile("qwen-coder")
    assert "qwen-coder" not in list_profiles()
    with pytest.raises(KeyError, match="Unknown profile 'qwen-coder'"):
        get_profile("qwen-coder")


def test_agent_dynamic_profile_switching():
    config = AgentConfig(
        profile="deepseek",
        openrouter_api_key="sk-or-test",
        nvidia_api_key="nvapi-test",
        anthropic_api_key="sk-ant-test",
    )
    agent = Agent(config)
    assert isinstance(agent.provider, OpenAICompatibleProvider)
    assert agent.config.model_name == "deepseek/deepseek-chat"

    # Switch to nemotron
    new_profile = agent.switch_model_profile("nemotron")
    assert new_profile.name == "nemotron"
    assert agent.config.provider == "nvidia"
    assert agent.config.model_name == "nvidia/nemotron-3-ultra-550b-a55b"
    assert isinstance(agent.provider, OpenAICompatibleProvider)
    assert agent.provider.base_url == "https://integrate.api.nvidia.com/v1"
    assert agent.cost_tracker.model_name == "nvidia/nemotron-3-ultra-550b-a55b"

    # Switch to sonnet
    sonnet_profile = agent.switch_model_profile("sonnet")
    assert sonnet_profile.name == "sonnet"
    assert agent.config.provider == "anthropic"
    assert isinstance(agent.provider, AnthropicProvider)
    assert agent.provider.model == "claude-sonnet-4-20250514"


def test_agent_switch_profile_missing_key_raises():
    config = AgentConfig(
        profile="deepseek",
        openrouter_api_key="sk-or-test",
        # intentionally no anthropic key
        anthropic_api_key=None,
    )
    config.anthropic_api_key = None
    agent = Agent(config)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        agent.switch_model_profile("sonnet")


def test_agent_switch_unknown_profile_raises():
    config = AgentConfig(
        profile="deepseek",
        openrouter_api_key="sk-or-test",
    )
    agent = Agent(config)

    with pytest.raises(KeyError, match="Unknown profile 'unknown-xyz'"):
        agent.switch_model_profile("unknown-xyz")
