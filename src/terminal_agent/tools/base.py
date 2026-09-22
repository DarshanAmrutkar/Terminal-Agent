import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional


from pathlib import Path


@dataclass
class ToolResult:
    """Represents the outcome of a tool execution."""
    output: str
    is_error: bool = False
    metadata: Optional[Dict[str, Any]] = None


def resolve_safe_path(path_str: str, base_dir: Path | str | None = None) -> tuple[Optional[Path], Optional[str]]:
    """Resolve a path safely within base_dir (defaults to Path.cwd()).

    Returns:
        (resolved_path, None) if path is within base_dir.
        (None, error_message) if path attempts to escape base_dir.
    """
    try:
        base = Path(base_dir or Path.cwd()).resolve()
        target = Path(path_str)
        if not target.is_absolute():
            resolved = (base / target).resolve()
        else:
            resolved = target.resolve()

        if resolved == base or base in resolved.parents:
            return resolved, None
        return None, f"Error: Path traversal outside working directory is blocked: '{path_str}'"
    except Exception as e:
        return None, f"Error resolving path '{path_str}': {e}"


class Tool(abc.ABC):
    """Abstract base class for all tools."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The unique name of the tool."""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """A description of what the tool does."""
        pass

    @property
    @abc.abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema defining the tool's parameters."""
        pass

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
        pass
