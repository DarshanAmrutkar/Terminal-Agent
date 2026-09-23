import json
import os
import tempfile
from pathlib import Path

import pytest

from terminal_agent.core.agent import Agent
from terminal_agent.core.config import AgentConfig, ExecutionMode
from terminal_agent.core.fast_path import AgentlessFastPath
from terminal_agent.llm.base import LLMProvider
from terminal_agent.llm.message import LLMResponse, Message


class MockFastPathProvider(LLMProvider):
    def __init__(self, search: str, replace: str):
        super().__init__(model="mock", api_key="test")
        self.search = search
        self.replace = replace

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def send(self, messages, tools=None):
        payload = json.dumps({"search": self.search, "replace": self.replace})
        return LLMResponse(message=Message.assistant(content=f"```json\n{payload}\n```"))

    async def stream(self, messages, tools=None):
        yield None


def test_fast_path_localization():
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        (dir_path / "user_auth.py").write_text("def authenticate(): pass")
        (dir_path / "payment_gateway.py").write_text("def process_payment(): pass")
        (dir_path / "order_processor.py").write_text("def create_order(): pass")

        provider = MockFastPathProvider("", "")
        fast_path = AgentlessFastPath(provider=provider, working_directory=dir_path)

        candidates = fast_path.localize("Fix authentication token validation in user_auth")
        assert len(candidates) > 0
        assert candidates[0] == "user_auth.py"


@pytest.mark.asyncio
async def test_fast_path_repair_and_validation():
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        target = dir_path / "math_utils.py"
        target.write_text("def add(a, b):\n    return a - b\n")

        # Provider provides fix: replace "return a - b" with "return a + b"
        provider = MockFastPathProvider(
            search="    return a - b",
            replace="    return a + b"
        )
        fast_path = AgentlessFastPath(provider=provider, working_directory=dir_path)

        result = await fast_path.execute(
            task_description="Fix subtraction bug in add function in math_utils.py"
        )

        assert result.success is True
        assert result.patch_applied is True
        assert "return a + b" in target.read_text()


@pytest.mark.asyncio
async def test_fast_path_syntax_error_rejection():
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        target = dir_path / "calc.py"
        target.write_text("def calculate():\n    return 42\n")

        # Provider returns broken syntax
        provider = MockFastPathProvider(
            search="    return 42",
            replace="    return 42 +"
        )
        fast_path = AgentlessFastPath(provider=provider, working_directory=dir_path)

        result = await fast_path.execute(
            task_description="Update return in calc.py"
        )

        assert result.success is False
        assert result.patch_applied is False
        assert result.phase_reached == "repair"
        assert "syntax error" in result.explanation.lower()
        # Original file unchanged
        assert target.read_text() == "def calculate():\n    return 42\n"


@pytest.mark.asyncio
async def test_agent_solve_task_fast_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        orig_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            dir_path = Path(temp_dir)
            target = dir_path / "greeting.py"
            target.write_text("def greet():\n    return 'bye'\n")

            config = AgentConfig(
                provider="anthropic",
                anthropic_api_key="sk-test",
                execution_mode=ExecutionMode.FAST_PATH,
            )
            agent = Agent(config=config, working_directory=temp_dir)
            # Inject mock provider
            agent.provider = MockFastPathProvider(
                search="    return 'bye'",
                replace="    return 'hello'"
            )

            res = await agent.solve_task("Fix greeting return in greeting.py")
            assert res["mode"] == "fast_path"
            assert res["success"] is True
            assert "return 'hello'" in target.read_text()
        finally:
            os.chdir(orig_cwd)
