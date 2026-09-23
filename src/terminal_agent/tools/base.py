import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ToolResult:
    """Represents the outcome of a tool execution."""
    output: str
    is_error: bool = False
    metadata: dict[str, Any] | None = None


import fnmatch

SENSITIVE_FILE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "*.pkcs12",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".git-credentials",
    ".netrc",
    "credentials.json",
    "*service_account*.json",
)

ALLOWED_SENSITIVE_EXCEPTIONS: tuple[str, ...] = (
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.defaults",
    "*.example",
    "*.sample",
)


def is_sensitive_path(path: Path | str) -> bool:
    """Check if a file path points to a sensitive file (secrets, credentials, keys).

    Explicit safe templates like '.env.example' are permitted.
    """
    p = Path(path)
    filename = p.name.lower()

    # 1. Check if filename is an explicit allowed exception
    for allowed in ALLOWED_SENSITIVE_EXCEPTIONS:
        if fnmatch.fnmatch(filename, allowed.lower()):
            return False

    # 2. Check filename against blocked patterns
    for pat in SENSITIVE_FILE_PATTERNS:
        if fnmatch.fnmatch(filename, pat.lower()):
            return True

    # 3. Check any parent directory component (e.g. .ssh/id_rsa or secrets/server.key)
    for part in p.parts:
        part_lower = part.lower()
        if part_lower in (".ssh", ".aws"):
            return True
        for pat in SENSITIVE_FILE_PATTERNS:
            if fnmatch.fnmatch(part_lower, pat.lower()):
                if not any(fnmatch.fnmatch(part_lower, a.lower()) for a in ALLOWED_SENSITIVE_EXCEPTIONS):
                    return True

    return False


def resolve_safe_path(
    path_str: str, 
    base_dir: Path | str | None = None,
    allow_sensitive: bool = False,
) -> tuple[Path | None, str | None]:
    """Resolve a path safely within base_dir (defaults to Path.cwd()).

    Guards against:
      1. Path traversal attacks outside the working directory (e.g., ../../etc/passwd)
      2. Direct access to sensitive secret files (.env, keys, credentials) unless allow_sensitive=True

    Returns:
        (resolved_path, None) if path is safe.
        (None, error_message) if path attempts to escape base_dir or touches sensitive files.
    """
    try:
        base = Path(base_dir or Path.cwd()).resolve()
        target = Path(path_str)
        if not target.is_absolute():
            resolved = (base / target).resolve()
        else:
            resolved = target.resolve()

        if resolved != base and base not in resolved.parents:
            return None, f"Error: Path traversal outside working directory is blocked: '{path_str}'"

        if not allow_sensitive and is_sensitive_path(resolved):
            return None, f"Security violation: Access to sensitive file is blocked: '{path_str}'"

        return resolved, None
    except Exception as e:
        return None, f"Error resolving path '{path_str}': {e}"


class Tool(abc.ABC):
    """Abstract base class for all tools."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The unique name of the tool."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """A description of what the tool does."""

    @property
    @abc.abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema defining the tool's parameters."""

    @property
    def requires_approval(self) -> bool:
        """Whether this tool requires user approval before execution."""
        return False

    @abc.abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the provided arguments.
        
        Returns:
            ToolResult containing the output, error status, and metadata.
        """
