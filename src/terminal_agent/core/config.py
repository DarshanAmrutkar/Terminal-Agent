"""Configuration management for Terminal Agent.

Supports loading from environment variables, CLI options, and .env file.
Responsible exclusively for configuration loading, API keys, and agent policies.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from terminal_agent.llm.profiles import ModelProfile, profile_registry


class PermissionMode(str, Enum):
    """Trust levels for command execution."""
    SAFE = "safe"
    AUTO_TEST = "auto-test"
    YOLO = "yolo"


class AgentConfig(BaseSettings):
    """Main configuration for Terminal Agent.
    
    Settings are loaded from (in priority order):
    1. Explicit constructor arguments
    2. Environment variables (AGENT_ prefix for most, standard names for API keys)
    3. .env file
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Active Profile & Model Settings ---
    profile: Optional[str] = Field(
        default=None,
        description="Active model profile preset (e.g. 'sonnet', 'nemotron', 'deepseek')",
    )
    provider: str = Field(
        default="anthropic",
        description="LLM provider: anthropic, nvidia, openai, openrouter",
    )
    model_name: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model identifier",
    )
    max_tokens: int = Field(
        default=8192,
        description="Max output tokens per LLM response",
    )

    # --- Agent Policy Settings ---
    max_iterations: int = Field(
        default=25,
        description="Max tool-use loops per user turn",
    )
    permission_mode: PermissionMode = Field(
        default=PermissionMode.SAFE,
        description="Command approval mode: safe, auto-test, yolo",
    )

    # --- Context Settings ---
    max_context_tokens: int = Field(
        default=100000,
        description="Max context window budget",
    )
    summarization_threshold: float = Field(
        default=0.75,
        description="Summarize when this fraction of context is used",
    )
    max_output_per_tool: int = Field(
        default=10000,
        description="Max chars per tool output before truncation",
    )

    # --- Command Settings ---
    timeout: int = Field(
        default=120,
        description="Shell command timeout in seconds",
    )
    safe_commands: list[str] = Field(
        default=[
            "ls", "dir", "cat", "type", "echo", "pwd", "cd", "grep", "find",
            "head", "tail", "wc", "sort", "uniq", "which", "where",
            "git status", "git diff", "git log", "git branch",
            "python --version", "python3 --version", "node --version",
            "pip list", "npm list",
        ],
        description="Commands that are always auto-approved",
    )
    blocked_patterns: list[str] = Field(
        default=[
            "rm -rf /", "rm -rf /*",
            "mkfs", "dd if=",
            "> /dev/sda", "> /dev/null",
            ":(){ :|:& };:",
            "sudo rm", "sudo mkfs",
            "format c:",
        ],
        description="Command patterns that are always blocked",
    )

    # --- API Keys (loaded from standard env var names, not AGENT_ prefix) ---
    anthropic_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)
    nvidia_api_key: Optional[str] = Field(default=None)
    openrouter_api_key: Optional[str] = Field(default=None)

    # --- Base URL Settings ---
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL override for OpenAI-compatible endpoints",
    )
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="Default base URL for NVIDIA NIM",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Default base URL for OpenAI",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Default base URL for OpenRouter",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Explicitly load .env into os.environ for non-prefixed vars
        from dotenv import load_dotenv
        load_dotenv(override=False)

        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.google_api_key:
            self.google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.nvidia_api_key:
            self.nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.openrouter_api_key:
            self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

        # Synchronize active profile if specified
        if self.profile and profile_registry.has(self.profile):
            preset = profile_registry.get(self.profile)
            if "provider" not in kwargs:
                self.provider = preset.provider
            if "model_name" not in kwargs:
                self.model_name = preset.model_name
            if "max_tokens" not in kwargs:
                self.max_tokens = preset.max_tokens
            if "max_context_tokens" not in kwargs:
                self.max_context_tokens = preset.context_window

    def get_model_profile(self) -> ModelProfile:
        """Return the active ModelProfile, resolving presets and ad-hoc overrides."""
        if self.profile and profile_registry.has(self.profile):
            preset = profile_registry.get(self.profile)
            return ModelProfile(
                name=preset.name,
                provider=self.provider,
                model_name=self.model_name,
                base_url=self.get_base_url(self.provider) or preset.base_url,
                max_tokens=self.max_tokens,
                context_window=self.max_context_tokens,
                description=preset.description,
            )
        return ModelProfile(
            name=self.provider,
            provider=self.provider,
            model_name=self.model_name,
            base_url=self.get_base_url(self.provider),
            max_tokens=self.max_tokens,
            context_window=self.max_context_tokens,
            description=f"{self.provider} / {self.model_name}",
        )

    def get_api_key(self, provider: Optional[str] = None) -> str | None:
        """Get the API key for the configured or specified provider."""
        p = provider or self.provider
        key_map = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "google": self.google_api_key,
            "nvidia": self.nvidia_api_key,
            "openrouter": self.openrouter_api_key,
        }
        return key_map.get(p)

    def get_base_url(self, provider: Optional[str] = None) -> str | None:
        """Return the API base URL for OpenAI-compatible providers."""
        if self.base_url:
            return self.base_url
        p = provider or self.provider
        url_map = {
            "nvidia": self.nvidia_base_url,
            "openai": self.openai_base_url,
            "openrouter": self.openrouter_base_url,
        }
        return url_map.get(p)

    def validate_api_key(self, provider: Optional[str] = None) -> None:
        """Raise ValueError if the API key for the provider is missing."""
        p = provider or self.provider
        key = self.get_api_key(p)
        if not key:
            env_var_map = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "google": "GOOGLE_API_KEY",
                "nvidia": "NVIDIA_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }
            env_var = env_var_map.get(p, f"{p.upper()}_API_KEY")
            raise ValueError(
                f"No API key found for provider '{p}'. "
                f"Set the {env_var} environment variable."
            )
