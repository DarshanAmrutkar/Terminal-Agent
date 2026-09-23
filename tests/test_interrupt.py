"""Tests for graceful interruption and subprocess tree cleanup."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

from terminal_agent.core.agent import Agent
from terminal_agent.core.config import AgentConfig
from terminal_agent.core.session_store import SessionStore
from terminal_agent.llm.base import LLMProvider
from terminal_agent.llm.message import LLMResponse, Message, StreamEvent
from terminal_agent.sandbox.base import SandboxPolicy
from terminal_agent.sandbox.local import LocalRestrictedSandbox


class HangingProvider(LLMProvider):
    """A provider that simulates an ongoing long streaming call until cancelled."""
    def __init__(self):
        super().__init__(model="hanging-model", api_key="test-key")

    async def send(self, messages, tools=None):
        await asyncio.sleep(10.0)
        return LLMResponse(message=Message.assistant(content="done"))

    async def stream(self, messages, tools=None):
        yield StreamEvent(type="text_delta", text="Starting...")
        await asyncio.sleep(10.0)
        yield StreamEvent(type="message_end")

    def count_tokens(self, text: str) -> int:
        return len(text.split())


@pytest.mark.asyncio
async def test_sandbox_terminate_active_process():
    """Verify that terminate() terminates an active running process in LocalRestrictedSandbox."""
    with tempfile.TemporaryDirectory() as temp_dir:
        policy = SandboxPolicy(working_dir=Path(temp_dir), timeout_seconds=30)
        sandbox = LocalRestrictedSandbox(policy=policy)

        # Launch a long sleep command
        sleep_cmd = "ping 127.0.0.1 -n 10" if sys.platform == "win32" else "sleep 10"
        task = asyncio.create_task(sandbox.execute(sleep_cmd))

        # Give it a moment to spawn
        await asyncio.sleep(0.3)
        assert sandbox._current_process is not None

        # Terminate active process
        await sandbox.terminate()

        # The task should complete or return early
        result = await task
        assert sandbox._current_process is None


@pytest.mark.asyncio
async def test_agent_process_message_interruption_preserves_session():
    """Verify that when process_message is cancelled (e.g. Ctrl+C), session state is auto-saved."""
    with tempfile.TemporaryDirectory() as temp_dir:
        orig_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            store = SessionStore(storage_dir=temp_dir)
            config = AgentConfig(provider="anthropic", anthropic_api_key="sk-test")
            
            agent = Agent(
                config=config,
                working_directory=temp_dir,
                session_store=store,
                listeners=[],
            )
            # Swap provider for hanging provider
            agent.provider = HangingProvider()

            # Start message processing in task
            task = asyncio.create_task(agent.process_message("Count to a million"))
            await asyncio.sleep(0.2)

            # Cancel task (simulating Ctrl+C KeyboardInterrupt / CancelledError)
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

            # Verify session was preserved in session_store
            saved_session = store.load_session(agent.session.session_id)
            assert saved_session is not None
            user_msgs = [m for m in saved_session.messages if m.role.value == "user"]
            assert len(user_msgs) == 1
            assert user_msgs[0].content == "Count to a million"
        finally:
            os.chdir(orig_cwd)
