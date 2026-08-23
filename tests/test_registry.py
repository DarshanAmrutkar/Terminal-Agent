"""Tests for the ToolRegistry.

These tests demonstrate why the Singleton was a problem and why the new
injectable design is correct. Each test creates its own registry instance
with exactly the tools it needs.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from terminal_agent.tools.registry import ToolRegistry, register_tool, _TOOL_CLASSES
from terminal_agent.tools.base import Tool, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTool(Tool):
    """A minimal tool for testing. Does not register via @register_tool."""

    @property
    def name(self) -> str:
        return "fake_tool"

    @property
    def description(self) -> str:
        return "A fake tool for unit tests."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output="fake output")


class AnotherFakeTool(Tool):
    @property
    def name(self) -> str:
        return "another_fake"

    @property
    def description(self) -> str:
        return "Another fake tool."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output="another output")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToolRegistryIsolation:
    """Each test creates a fresh ToolRegistry. No shared state."""

    def test_empty_registry_has_no_tools(self):
        registry = ToolRegistry()
        assert len(registry) == 0
        assert registry.get_all() == []

    def test_register_tool_instance(self):
        registry = ToolRegistry()
        registry.register(FakeTool())
        assert len(registry) == 1
        assert "fake_tool" in registry

    def test_two_registries_are_independent(self):
        """This is the key test that would have FAILED with the old Singleton."""
        registry_a = ToolRegistry()
        registry_b = ToolRegistry()

        registry_a.register(FakeTool())

        # registry_b must NOT see registry_a's tools
        assert len(registry_b) == 0
        assert "fake_tool" not in registry_b

    def test_get_unknown_tool_raises_keyerror(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent_tool")

    def test_get_known_tool_returns_instance(self):
        registry = ToolRegistry()
        tool = FakeTool()
        registry.register(tool)
        assert registry.get("fake_tool") is tool

    def test_get_tool_definitions_format(self):
        """Tool definitions must match the format expected by LLM APIs."""
        registry = ToolRegistry()
        registry.register(FakeTool())

        defs = registry.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "fake_tool"
        assert "description" in defs[0]
        assert "input_schema" in defs[0]

    def test_register_class_instantiates_tool(self):
        registry = ToolRegistry()
        registry.register_class(FakeTool)
        assert "fake_tool" in registry

    def test_contains_operator(self):
        registry = ToolRegistry()
        registry.register(FakeTool())
        assert "fake_tool" in registry
        assert "nonexistent" not in registry

    def test_repr_shows_tool_names(self):
        registry = ToolRegistry()
        registry.register(FakeTool())
        r = repr(registry)
        assert "fake_tool" in r


class TestLoadDefaults:
    """Tests for load_defaults() integration with @register_tool."""

    def test_load_defaults_loads_real_tools(self):
        """load_defaults() should populate the registry with all @register_tool classes."""
        # Import tool modules to trigger @register_tool
        import terminal_agent.tools.read_file      # noqa: F401
        import terminal_agent.tools.write_file     # noqa: F401
        import terminal_agent.tools.run_command    # noqa: F401

        registry = ToolRegistry()
        registry.load_defaults()

        assert "read_file" in registry
        assert "write_file" in registry
        assert "run_command" in registry

    def test_load_defaults_twice_does_not_duplicate(self):
        """Calling load_defaults() twice should not create duplicate entries."""
        import terminal_agent.tools.read_file  # noqa: F401

        registry = ToolRegistry()
        registry.load_defaults()
        count_after_first = len(registry)

        registry.load_defaults()  # second call
        assert len(registry) == count_after_first
