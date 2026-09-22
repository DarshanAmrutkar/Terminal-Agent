"""Execution sandboxing subsystem for Terminal Agent."""

from __future__ import annotations

from terminal_agent.sandbox.base import (
    DEFAULT_ALLOWED_ENV_VARS,
    DEFAULT_BLOCKED_ENV_PATTERNS,
    SandboxBackend,
    SandboxPolicy,
    SandboxResult,
)
from terminal_agent.sandbox.docker import DockerSandbox
from terminal_agent.sandbox.factory import DisabledSandbox, create_sandbox
from terminal_agent.sandbox.local import LocalRestrictedSandbox

__all__ = [
    "DEFAULT_ALLOWED_ENV_VARS",
    "DEFAULT_BLOCKED_ENV_PATTERNS",
    "DisabledSandbox",
    "DockerSandbox",
    "LocalRestrictedSandbox",
    "SandboxBackend",
    "SandboxPolicy",
    "SandboxResult",
    "create_sandbox",
]
