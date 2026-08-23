"""Configuration management for Terminal Agent.

Supports loading from environment variables and optional .terminal-agent.toml config file.
API keys use standard environment variable names (e.g., ANTHROPIC_API_KEY).
Agent-specific settings use the AGENT_ prefix.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    Supported providers:
      - "anthropic"  : Claude models via Anthropic API
      - "nvidia"     : NVIDIA NIM models (OpenAI-compatible)
      - "openai"     : OpenAI models (GPT-4o, o1, etc.)
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model Settings ---
    provider: str = Field(default="anthropic", description="LLM provider: anthropic, nvidia, openai")
    model_name: str = Field(default="claude-sonnet-4-20250514", description="Model identifier")
    max_tokens: int = Field(default=8192, description="Max output tokens per LLM response")

    # --- Agent Settings ---
    max_iterations: int = Field(default=25, description="Max tool-use loops per user turn")
    permission_mode: PermissionMode = Field(
        default=PermissionMode.SAFE,
        description="Command approval mode: safe, auto-test, yolo",
    )

    # --- Context Settings ---
    max_context_tokens: int = Field(default=100000, description="Max context window budget")
    summarization_threshold: float = Field(
        default=0.75, description="Summarize when this fraction of context is used"
    )
    max_output_per_tool: int = Field(
        default=10000, description="Max chars per tool output before truncation"
    )

    # --- Command Settings ---
    timeout: int = Field(default=120, description="Shell command timeout in seconds")
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

    # --- Provider-specific URLs ---
    # NVIDIA NIM and other OpenAI-compatible providers can be targeted by
    # changing this URL. Defaults to NVIDIA's cloud NIM endpoint.
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="Base URL for the NVIDIA NIM API (or any OpenAI-compatible endpoint)",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the OpenAI API",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # pydantic-settings automatically reads AGENT_* vars from the .env file.
        # However, non-prefixed vars like NVIDIA_API_KEY and ANTHROPIC_API_KEY
        # are NOT loaded into os.environ by pydantic-settings. We call
        # load_dotenv() explicitly so those vars become available via os.environ.
        # load_dotenv() is idempotent — calling it multiple times is safe.
        from dotenv import load_dotenv
        load_dotenv(override=False)  # override=False: env vars set before launch win

        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.google_api_key:
            self.google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.nvidia_api_key:
            self.nvidia_api_key = os.environ.get("NVIDIA_API_KEY")

    def get_api_key(self) -> str | None:
        """Get the API key for the configured provider."""
        key_map = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "google": self.google_api_key,
            "nvidia": self.nvidia_api_key,
        }
        return key_map.get(self.provider)

    def get_base_url(self) -> str | None:
        """Return the API base URL for OpenAI-compatible providers.

        Returns None for providers (like Anthropic) that use their own SDK
        and don't need a configurable base URL.
        """
        url_map = {
            "nvidia": self.nvidia_base_url,
            "openai": self.openai_base_url,
        }
        return url_map.get(self.provider)

    def validate_api_key(self) -> None:
        """Raise ValueError if the API key for the configured provider is missing."""
        key = self.get_api_key()
        if not key:
            env_var_map = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "google": "GOOGLE_API_KEY",
                "nvidia": "NVIDIA_API_KEY",
            }
            env_var = env_var_map.get(self.provider, f"{self.provider.upper()}_API_KEY")
            raise ValueError(
                f"No API key found for provider '{self.provider}'. "
                f"Set the {env_var} environment variable."
            )
