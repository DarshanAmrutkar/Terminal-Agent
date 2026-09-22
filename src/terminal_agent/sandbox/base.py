"""Base interfaces and data structures for execution sandboxing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import fnmatch
import os
from pathlib import Path
from typing import Any, Sequence


@dataclass
class SandboxResult:
    """The outcome of executing a command inside a sandbox."""
    stdout: str
    stderr: str
    returncode: int
    duration_seconds: float
    timed_out: bool = False
    truncated: bool = False

    @property
    def is_error(self) -> bool:
        return self.timed_out or self.returncode != 0

    @property
    def formatted_output(self) -> str:
        """Render a formatted string suitable for tool results."""
        if self.timed_out:
            return f"Error: Command timed out after {self.duration_seconds:.1f}s."

        blocks = []
        if self.stdout:
            blocks.append(f"STDOUT:\n{self.stdout}")
        if self.stderr:
            blocks.append(f"STDERR:\n{self.stderr}")

        if not blocks:
            return "Command executed successfully with no output."
        return "\n\n".join(blocks)


# Safe environment variables permitted to pass through into sandboxed subprocesses
DEFAULT_ALLOWED_ENV_VARS: set[str] = {
    # System essentials
    "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "TMPDIR",
    "USER", "USERNAME", "HOME", "USERPROFILE",
    "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ",
    # Python & build environments
    "PYTHONPATH", "PYTHONHOME", "PYTHONUNBUFFERED", "PYTHONIOENCODING",
    "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    # Common runtime tools
    "NODE_PATH", "CARGO_HOME", "RUSTUP_HOME", "GO_PATH", "GOPATH",
}

# Glob patterns of environment variables that must ALWAYS be stripped (secrets, tokens, credentials)
DEFAULT_BLOCKED_ENV_PATTERNS: list[str] = [
    "*KEY*",
    "*SECRET*",
    "*TOKEN*",
    "*PASSWORD*",
    "*PASSWD*",
    "*CREDENTIAL*",
    "*AUTH*",
    "AWS_*",
    "AZURE_*",
    "GCP_*",
    "GOOGLE_*",
    "SSH_*",
    "GITHUB_*",
    "GITLAB_*",
    "ANTHROPIC_*",
    "OPENAI_*",
    "NVIDIA_*",
    "OPENROUTER_*",
    "AGENT_*_API_KEY",
]


@dataclass
class SandboxPolicy:
    """Policy rules governing sandboxed command execution."""
    working_dir: Path
    allowed_env_vars: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_ENV_VARS))
    blocked_env_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_BLOCKED_ENV_PATTERNS))
    allow_network: bool = True
    timeout_seconds: int = 120
    max_output_chars: int = 10000

    def sanitize_environment(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """Produce a scrubbed environment dictionary stripping secrets and unwhitelisted keys."""
        source = base_env if base_env is not None else dict(os.environ)
        sanitized: dict[str, str] = {}

        # Case-insensitive lookup map for Windows
        upper_allowed = {k.upper() for k in self.allowed_env_vars}

        for key, val in source.items():
            key_upper = key.upper()

            # 1. Reject if key matches any blocked wildcard pattern
            is_blocked = any(
                fnmatch.fnmatch(key_upper, pat.upper())
                for pat in self.blocked_env_patterns
            )
            if is_blocked:
                continue

            # 2. Allow if present in allowed whitelist
            if key_upper in upper_allowed:
                sanitized[key] = val

        # Ensure working_dir is set in environment if relevant
        sanitized["PWD"] = str(self.working_dir)
        return sanitized


class SandboxBackend(ABC):
    """Abstract Port for command execution sandboxes."""

    def __init__(self, policy: SandboxPolicy):
        self.policy = policy

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the sandbox backend implementation."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this sandbox backend is functional on the current system."""
        pass

    @abstractmethod
    async def execute(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: int | None = None,
    ) -> SandboxResult:
        """Execute command within the sandbox according to the configured policy."""
        pass
