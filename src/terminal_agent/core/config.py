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
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model Settings ---
    provider: str = Field(default="anthropic", description="LLM provider: anthropic, openai, google, nvidia")
    model_name: str = Field(default="claude-sonnet-4-20250514", description="Model identifier")
    max_tokens: int = Field(default=8192, description="Max output tokens per LLM response")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA NIM API base URL",
    )

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

    def __init__(self, **kwargs):
        # Load .env into os.environ FIRST so API key fallbacks work
        from dotenv import load_dotenv
        load_dotenv(override=False)
        
        super().__init__(**kwargs)
        # API keys use standard env var names (no AGENT_ prefix)
        # Override from environment if not set via AGENT_ prefix
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

    def validate_api_key(self) -> None:
        """Raise ValueError if the API key for the configured provider is missing or a placeholder."""
        key = self.get_api_key()
        if not key or key.startswith("your-") or "your-api-key" in key or "your-nvidia-api-key" in key or len(key) < 15:
            raise ValueError(
                f"No valid API key found for provider '{self.provider}'.\n"
                f"Your environment or .env has a placeholder key: '{key}'.\n"
                f"Please set your real API key via environment variable: export {self.provider.upper()}_API_KEY=nvapi-...\n"
                f"or update your .env file with {self.provider.upper()}_API_KEY=nvapi-..."
            )
