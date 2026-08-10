from typing import Dict, List, Type
from .base import Tool


class ToolRegistry:
    """Singleton registry for managing tools."""
    
    _instance = None
    _tools: Dict[str, Tool] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool_class: Type[Tool]) -> None:
        """Register a tool class in the registry."""
        tool_instance = tool_class()
        self._tools[tool_instance.name] = tool_instance

    def get(self, name: str) -> Tool:
        """Retrieve a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return self._tools[name]

    def get_all(self) -> List[Tool]:
        """Retrieve all registered tools."""
        return list(self._tools.values())

    def get_tool_definitions(self) -> List[Dict]:
        """
        Get tool schemas formatted for LLM APIs (e.g., Anthropic format).
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in self._tools.values()
        ]


def register_tool(cls: Type[Tool]) -> Type[Tool]:
    """Decorator to register a tool class with the global ToolRegistry."""
    registry = ToolRegistry()
    registry.register(cls)
    return cls
