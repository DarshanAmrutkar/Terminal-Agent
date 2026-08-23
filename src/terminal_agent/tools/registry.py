"""Tool registry — manages the collection of tools available to the agent.

Design: Dependency Injection over Singleton
--------------------------------------------
The previous design used a Singleton pattern with class-level state. That made
unit testing impossible: every test shared the same global tool registry, and
there was no way to create a fresh, isolated registry per test.

The new design separates two concerns:

  1. Tool *declaration*: The @register_tool decorator collects tool classes into
     a module-level list (_TOOL_CLASSES). This runs at import time and is
     intentionally side-effect-light — it doesn't instantiate anything.

  2. Tool *registry*: ToolRegistry is a plain class. Agent creates its own
     instance and populates it explicitly via load_defaults() or register().
     No global state. No shared instances between tests.

Why not just a plain dict in Agent?
  We want a typed, method-rich abstraction that can be extended (e.g., to support
  lazy loading, tool metadata, or remote tool registries) without changing Agent.
  A class gives us that extension point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from .base import Tool

# ---------------------------------------------------------------------------
# Module-level declaration registry
# ---------------------------------------------------------------------------
# The @register_tool decorator appends tool *classes* here at import time.
# Nothing is instantiated. Agent decides when and how to create instances.
_TOOL_CLASSES: list[Type[Tool]] = []


def register_tool(cls: Type[Tool]) -> Type[Tool]:
    """Decorator that marks a Tool subclass for automatic registration.

    Usage::

        @register_tool
        class ReadFileTool(Tool):
            ...

    The class is appended to _TOOL_CLASSES. When a ToolRegistry is created and
    load_defaults() is called, all decorated classes are instantiated and
    registered in that registry instance.
    """
    _TOOL_CLASSES.append(cls)
    return cls


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Manages the set of tools available to an agent instance.

    This is a plain class — not a Singleton. Every Agent creates its own
    ToolRegistry, giving each agent full control over which tools are available.
    This is critical for testing: a test can create a ToolRegistry, register
    only the tools it needs, and inject it into the component under test.

    Responsibilities:
      - Store tool instances keyed by name.
      - Provide lookup by name (used during tool execution).
      - Provide the full tool definition list (sent to the LLM as JSON schemas).
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Register a tool *instance* directly.

        Use this when you need precise control, e.g. in tests::

            registry = ToolRegistry()
            registry.register(MockReadFileTool())
        """
        self._tools[tool.name] = tool

    def register_class(self, tool_class: Type[Tool]) -> None:
        """Instantiate a tool class and register it."""
        self.register(tool_class())

    def load_defaults(self) -> None:
        """Instantiate and register all tools decorated with @register_tool.

        Call this once after importing all tool modules::

            import terminal_agent.tools.read_file      # triggers @register_tool
            import terminal_agent.tools.write_file
            ...
            registry = ToolRegistry()
            registry.load_defaults()
        """
        for tool_class in _TOOL_CLASSES:
            self.register(tool_class())

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool:
        """Return a registered tool by name.

        Raises:
            KeyError: if no tool with that name has been registered.
        """
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' is not registered. "
                f"Available tools: {sorted(self._tools)}"
            )
        return self._tools[name]

    def get_all(self) -> list[Tool]:
        """Return all registered tool instances."""
        return list(self._tools.values())

    def get_tool_definitions(self) -> list[dict]:
        """Return tool schemas in the format expected by LLM APIs.

        The schema format follows Anthropic's tool definition structure.
        Other providers override format_tools() in their LLMProvider subclass
        to adapt this structure to their API's expectations.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={sorted(self._tools)})"
