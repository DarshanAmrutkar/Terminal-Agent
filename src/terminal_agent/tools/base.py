import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """Represents the outcome of a tool execution."""
    output: str
    is_error: bool = False
    metadata: Optional[Dict[str, Any]] = None


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
