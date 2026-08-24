"""LLM provider and profile interfaces."""

from terminal_agent.llm.base import LLMProvider
from terminal_agent.llm.anthropic import AnthropicProvider
from terminal_agent.llm.openai_compatible import OpenAICompatibleProvider
from terminal_agent.llm.message import Message, Role, ToolCall, ToolResultContent, StreamEvent, LLMResponse
from terminal_agent.llm.profiles import (
    ModelProfile,
    ProfileRegistry,
    DEFAULT_PROFILES,
    profile_registry,
    register_profile,
    get_profile,
    list_profiles,
    remove_profile,
)

__all__ = [
    "LLMProvider",
    "AnthropicProvider",
    "OpenAICompatibleProvider",
    "Message",
    "Role",
    "ToolCall",
    "ToolResultContent",
    "StreamEvent",
    "LLMResponse",
    "ModelProfile",
    "ProfileRegistry",
    "DEFAULT_PROFILES",
    "profile_registry",
    "register_profile",
    "get_profile",
    "list_profiles",
    "remove_profile",
]
