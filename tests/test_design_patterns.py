import asyncio
import os
from pathlib import Path
import tempfile
import pytest

from terminal_agent.core.config import AgentConfig
from terminal_agent.core.session import Session
from terminal_agent.core.agent import Agent
from terminal_agent.core.events import (
    AgentEventListener,
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    TurnStartEvent,
    TurnEndEvent,
)
from terminal_agent.llm.base import LLMProvider
from terminal_agent.llm.message import Message, ToolCall, LLMResponse
from terminal_agent.llm.registry import ProviderRegistry
from terminal_agent.utils.approval import AutoApprovalHandler
from terminal_agent.tools.registry import discover_builtin_tools, ToolRegistry


class DummyProvider(LLMProvider):
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def send(self, messages, tools=None):
        return LLMResponse(message=Message.assistant(content="dummy response"))

    async def stream(self, messages, tools=None):
        yield None


def test_provider_registry_open_closed_principle():
    """Verify that new providers can be registered without editing core Agent code."""
    registry = ProviderRegistry()
    assert "dummy" not in registry.list_providers()

    registry.register("dummy", lambda cfg: DummyProvider(model="dummy-model", api_key="dummy-key"))
    assert "dummy" in registry.list_providers()

    config = AgentConfig(provider="dummy", anthropic_api_key="key")
    provider = registry.create("dummy", config)
    assert isinstance(provider, DummyProvider)

    with pytest.raises(ValueError, match="Unsupported provider"):
        registry.create("non_existent", config)


class EventRecordingListener(AgentEventListener):
    def __init__(self):
        self.events = []

    def on_turn_start(self, event: TurnStartEvent):
        self.events.append(("turn_start", event.iteration))

    def on_turn_end(self, event: TurnEndEvent):
        self.events.append(("turn_end", event.iteration))

    def on_text_delta(self, event: TextDeltaEvent):
        self.events.append(("text_delta", event.text))

    def on_tool_call_start(self, event: ToolCallStartEvent):
        self.events.append(("tool_call_start", event.tool_name))

    def on_tool_call_end(self, event: ToolCallEndEvent):
        self.events.append(("tool_call_end", event.tool_name, event.is_error))


@pytest.mark.asyncio
async def test_agent_event_observer_pattern():
    """Verify Agent emits events to observers without coupling to console rendering."""
    with tempfile.TemporaryDirectory() as temp_dir:
        orig_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            config = AgentConfig(
                provider="anthropic",
                anthropic_api_key="sk-test",
            )
            listener = EventRecordingListener()
            agent = Agent(
                config, 
                working_directory=temp_dir, 
                listeners=[listener],
                approval_handler=AutoApprovalHandler(approve=True),
            )

            # Emit synthetic tool call
            tc = ToolCall(id="call_1", name="write_file", arguments={"path": "test.txt", "content": "hello"})
            res = await agent._execute_single_tool(tc)

            assert not res.is_error
            # Verify listener caught both start and end events
            event_types = [e[0] for e in listener.events]
            assert "tool_call_start" in event_types
            assert "tool_call_end" in event_types
        finally:
            os.chdir(orig_cwd)


@pytest.mark.asyncio
async def test_approval_strategy_pattern():
    """Verify ApprovalHandler strategy allows deterministic auto-approval/denial."""
    with tempfile.TemporaryDirectory() as temp_dir:
        orig_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            config = AgentConfig(
                provider="anthropic",
                anthropic_api_key="sk-test",
                permission_mode="safe",  # In safe mode, write_file requires approval
            )

            # 1. Auto-deny strategy
            agent_deny = Agent(
                config,
                working_directory=temp_dir,
                approval_handler=AutoApprovalHandler(approve=False),
                listeners=[],  # Headless
            )
            tc = ToolCall(id="c1", name="write_file", arguments={"path": "denied.txt", "content": "evil"})
            res_denied = await agent_deny._execute_single_tool(tc)
            assert res_denied.is_error is True
            assert "User declined" in res_denied.output
            assert not (Path(temp_dir) / "denied.txt").exists()

            # 2. Auto-approve strategy
            agent_approve = Agent(
                config,
                working_directory=temp_dir,
                approval_handler=AutoApprovalHandler(approve=True),
                listeners=[],
            )
            tc2 = ToolCall(id="c2", name="write_file", arguments={"path": "approved.txt", "content": "good"})
            res_approved = await agent_approve._execute_single_tool(tc2)
            assert res_approved.is_error is False
            assert (Path(temp_dir) / "approved.txt").exists()
        finally:
            os.chdir(orig_cwd)


def test_session_token_encapsulation_tell_dont_ask():
    """Verify Session.apply_stream_usage encapsulates token arithmetic without leaking state."""
    session = Session(working_directory=".")
    assert session.total_input_tokens == 0
    assert session.total_output_tokens == 0

    # Stream token updates: turn 1
    session.apply_stream_usage(current_input=100, current_output=50, last_input=0, last_output=0)
    assert session.total_input_tokens == 100
    assert session.total_output_tokens == 50
    assert session.current_context_tokens == 150

    # Stream delta within turn 1
    session.apply_stream_usage(current_input=100, current_output=80, last_input=100, last_output=50)
    assert session.total_input_tokens == 100
    assert session.total_output_tokens == 80
    assert session.current_context_tokens == 180


def test_dynamic_tool_discovery_dependency_inversion():
    """Verify ToolRegistry discovers all builtin tools without hardcoded agent imports."""
    registry = ToolRegistry()
    registry.load_defaults()

    tool_names = [t.name for t in registry.get_all()]
    expected_tools = ["read_file", "write_file", "search_replace", "grep_search", "list_directory", "run_command", "get_repo_map"]

    for expected in expected_tools:
        assert expected in tool_names, f"Expected '{expected}' in dynamically discovered tools"
