import pytest
from terminal_agent.core.reflexion import ReflexionMemory, FailureEpisode
from terminal_agent.llm.message import ToolCall
from terminal_agent.core.config import AgentConfig
from terminal_agent.core.agent import Agent


def test_reflexion_record_failure_and_heuristics():
    memory = ReflexionMemory(max_episodes=5)

    # 1. Syntax Error
    ep1 = memory.record_failure(
        tool_name="write_file",
        args={"path": "src/main.py"},
        error_output="[Syntax Gate: Rejected Edit]\nSyntaxError at line 12: unclosed parenthesis"
    )
    assert ep1.tool_name == "write_file"
    assert "src/main.py" in ep1.action_summary
    assert "syntax" in ep1.reflection.lower()
    assert ep1.resolved is False

    # 2. Indentation Error
    ep2 = memory.record_failure(
        tool_name="search_replace",
        args={"path": "src/utils.py"},
        error_output="IndentationError: unexpected indent at line 55"
    )
    assert "indentation" in ep2.reflection.lower()

    # 3. Path Traversal
    ep3 = memory.record_failure(
        tool_name="write_file",
        args={"path": "../../etc/passwd"},
        error_output="Error: Path traversal outside working directory is blocked"
    )
    assert "sandbox" in ep3.reflection.lower() or "workspace" in ep3.reflection.lower()

    # 4. Pytest Failure
    ep4 = memory.record_failure(
        tool_name="run_command",
        args={"command": "pytest tests/test_core.py"},
        error_output="FAILED tests/test_core.py::test_auth - AssertionError: assert False"
    )
    assert "test" in ep4.reflection.lower()


def test_reflexion_deduplication():
    memory = ReflexionMemory(max_episodes=5)

    # Record first failure
    ep1 = memory.record_failure(
        tool_name="run_command",
        args={"command": "pytest tests/"},
        error_output="1 failed"
    )
    assert len(memory.episodes) == 1

    # Record second failure with same tool and command
    ep2 = memory.record_failure(
        tool_name="run_command",
        args={"command": "pytest tests/"},
        error_output="1 failed again with new trace"
    )
    assert len(memory.episodes) == 1
    assert memory.episodes[0].error_output == "1 failed again with new trace"


def test_reflexion_bounded_buffer():
    memory = ReflexionMemory(max_episodes=3)

    for i in range(5):
        memory.record_failure(
            tool_name="run_command",
            args={"command": f"cmd_{i}"},
            error_output=f"Error {i}"
        )

    # Should be capped at 3
    assert len(memory.episodes) == 3
    assert memory.episodes[-1].action_summary == "command='cmd_4'"


def test_reflexion_mark_resolved_and_context_block():
    memory = ReflexionMemory(max_episodes=5)

    memory.record_failure(
        tool_name="write_file",
        args={"path": "app.py"},
        error_output="SyntaxError: invalid syntax"
    )
    memory.record_failure(
        tool_name="run_command",
        args={"command": "pytest"},
        error_output="AssertionError: 2 != 3"
    )

    assert memory.has_active_reflections() is True
    context_block = memory.format_context_block()
    assert "Episodic Self-Reflections" in context_block
    assert "write_file" in context_block
    assert "run_command" in context_block

    # Resolve write_file
    resolved_count = memory.mark_resolved("write_file")
    assert resolved_count == 1
    assert memory.has_active_reflections() is True

    # Context block should now only contain run_command
    updated_block = memory.format_context_block()
    assert "write_file" not in updated_block
    assert "run_command" in updated_block

    # Resolve all
    memory.mark_resolved()
    assert memory.has_active_reflections() is False
    assert memory.format_context_block() == ""


@pytest.mark.asyncio
async def test_agent_integration_with_reflexion():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as temp_dir:
        orig_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            config = AgentConfig(
                provider="anthropic",
                anthropic_api_key="sk-ant-test",
                model_name="claude-3-5-sonnet-20241022",
                permission_mode="yolo",  # Auto-approve for test speed
            )
            agent = Agent(config=config, working_directory=temp_dir)

            # Trigger a tool error: write_file with invalid syntax
            tool_call = ToolCall(
                id="call_1",
                name="write_file",
                arguments={
                    "path": "broken.py",
                    "content": "def foo(\n   broken syntax"
                }
            )

            result = await agent._execute_single_tool(tool_call)
            assert result.is_error is True
            assert "[Reflexion Self-Correction Advice]" in result.output
            assert agent.reflexion_memory.has_active_reflections() is True

            # Now trigger a successful write
            tool_call_ok = ToolCall(
                id="call_2",
                name="write_file",
                arguments={
                    "path": "fixed.py",
                    "content": "def foo():\n    pass\n"
                }
            )
            result_ok = await agent._execute_single_tool(tool_call_ok)
            assert result_ok.is_error is False

            # write_file failures should now be marked resolved
            assert any(ep.resolved for ep in agent.reflexion_memory.episodes if ep.tool_name == "write_file")
        finally:
            os.chdir(orig_cwd)
