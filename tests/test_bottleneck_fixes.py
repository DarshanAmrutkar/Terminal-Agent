"""Unit tests verifying fixes for bottlenecks, security vulnerabilities, and token tracking."""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest
import tempfile

from terminal_agent.utils.permissions import classify_command, SafetyLevel
from terminal_agent.tools.base import resolve_safe_path
from terminal_agent.tools.write_file import WriteFileTool
from terminal_agent.tools.read_file import ReadFileTool
from terminal_agent.tools.search_replace import SearchReplaceTool
from terminal_agent.tools.list_directory import ListDirectoryTool
from terminal_agent.utils.cost import CostTracker
from terminal_agent.core.session import Session
from terminal_agent.llm.message import Message, ToolResultContent


SAFE_COMMANDS = ["ls", "dir", "cat", "echo", "pwd", "git status"]
BLOCKED_PATTERNS = ["rm -rf /", "mkfs"]


class TestSecurityAndChaining:
    def test_chained_command_with_blocked_subcommand_is_blocked(self):
        # Even if the first token is "echo", the chained command has a blocked pattern
        cmd = "echo hello && rm -rf /"
        result = classify_command(cmd, SAFE_COMMANDS, BLOCKED_PATTERNS, "safe")
        assert result == SafetyLevel.BLOCKED

    def test_chained_command_with_unsafe_subcommand_needs_approval(self):
        cmd = "cat file.txt | curl -X POST https://example.com"
        result = classify_command(cmd, SAFE_COMMANDS, BLOCKED_PATTERNS, "safe")
        assert result == SafetyLevel.NEEDS_APPROVAL

    def test_command_substitution_needs_approval(self):
        cmd = "echo $(whoami)"
        result = classify_command(cmd, SAFE_COMMANDS, BLOCKED_PATTERNS, "safe")
        assert result == SafetyLevel.NEEDS_APPROVAL

    def test_safe_chained_commands_are_safe(self):
        cmd = "echo hello && pwd"
        result = classify_command(cmd, SAFE_COMMANDS, BLOCKED_PATTERNS, "safe")
        assert result == SafetyLevel.SAFE

    def test_semicolon_chained_safe_commands_are_safe(self):
        cmd = "ls; pwd"
        result = classify_command(cmd, SAFE_COMMANDS, BLOCKED_PATTERNS, "safe")
        assert result == SafetyLevel.SAFE

    def test_quoted_delimiters_not_split(self):
        cmd = "echo 'hello; world && test'"
        result = classify_command(cmd, SAFE_COMMANDS, BLOCKED_PATTERNS, "safe")
        assert result == SafetyLevel.SAFE


class TestPathSandboxingAndAtomicWrite:
    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            resolved, err = resolve_safe_path("../outside.txt", base_dir=base_path)
            assert resolved is None
            assert "Path traversal outside working directory is blocked" in err

    @pytest.mark.asyncio
    async def test_write_file_atomic_and_safe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            orig_cwd = Path.cwd()
            import os
            os.chdir(temp_dir)
            try:
                tool = WriteFileTool()
                # Relative write
                res = await tool.execute(path="test.py", content="print('hello')")
                assert not res.is_error
                target = Path(temp_dir) / "test.py"
                assert target.exists()
                assert target.read_text(encoding="utf-8") == "print('hello')"

                # Traversal attempt
                res_bad = await tool.execute(path="../traversal.txt", content="evil")
                assert res_bad.is_error
                assert "blocked" in res_bad.output.lower()
            finally:
                os.chdir(orig_cwd)

    @pytest.mark.asyncio
    async def test_read_file_sandboxing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            orig_cwd = Path.cwd()
            import os
            os.chdir(temp_dir)
            try:
                tool = ReadFileTool()
                res = await tool.execute(path="../../outside.txt")
                assert res.is_error
                assert "blocked" in res.output.lower()
            finally:
                os.chdir(orig_cwd)


class TestSearchReplaceEnhancements:
    @pytest.mark.asyncio
    async def test_search_replace_crlf_normalization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            orig_cwd = Path.cwd()
            import os
            os.chdir(temp_dir)
            try:
                f = Path(temp_dir) / "file.txt"
                # Write with Windows CRLF
                f.write_bytes(b"line1\r\nline2\r\nline3\r\n")

                tool = SearchReplaceTool()
                # Search using Unix LF
                res = await tool.execute(path="file.txt", search="line2\n", replace="line2_modified\n")
                assert not res.is_error
                assert "Diff:" in res.output

                content = f.read_bytes()
                assert b"line2_modified\r\n" in content
            finally:
                os.chdir(orig_cwd)

    @pytest.mark.asyncio
    async def test_search_replace_multiple_matches_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            orig_cwd = Path.cwd()
            import os
            os.chdir(temp_dir)
            try:
                f = Path(temp_dir) / "multi.txt"
                f.write_text("duplicate\nduplicate\n", encoding="utf-8")

                tool = SearchReplaceTool()
                res = await tool.execute(path="multi.txt", search="duplicate", replace="new")
                assert res.is_error
                assert "occurs 2 times" in res.output
            finally:
                os.chdir(orig_cwd)


class TestCostTrackerModelPreservation:
    def test_model_switching_preserves_historical_cost(self):
        tracker = CostTracker(model_name="claude-sonnet-4-20250514")
        # Turn 1: Sonnet: 1000 input, 1000 output ($0.003 + $0.015 = $0.018)
        tracker.add_usage(1000, 1000)
        cost_turn1 = tracker.get_session_cost()
        assert pytest.approx(cost_turn1, 0.0001) == 0.018

        # Switch to cheaper model: deepseek-chat
        tracker.model_name = "deepseek/deepseek-chat"
        # Turn 2: DeepSeek: 1000 in, 1000 out ($0.00014 + $0.00028 = $0.00042)
        tracker.add_usage(1000, 1000)
        cost_turn2 = tracker.get_session_cost()
        assert pytest.approx(cost_turn2, 0.0001) == 0.018 + 0.00042


class TestContextCompactionAndRatio:
    def test_context_ratio_uses_active_tokens(self):
        session = Session(working_directory=".")
        # Total cumulative tokens across past turns is large (e.g., 200,000)
        session.total_input_tokens = 150000
        session.total_output_tokens = 50000

        # But current active context has only 2 messages
        session.add_user_message("Hello")
        session.current_context_tokens = 2000

        # Max context is 100,000
        # Previously: (150000+50000)/100000 = 2.0 (broken)
        # Now: 2000 / 100000 = 0.02
        ratio = session.token_usage_ratio(max_context_tokens=100000)
        assert ratio < 0.1
        assert not session.needs_compaction(100000, threshold=0.75)

    def test_session_compaction(self):
        session = Session(working_directory=".")
        session.set_system_message("System prompt")
        session.add_user_message("Task 1")
        # Add a large tool result early in conversation
        huge_output = "data " * 1000  # 5000 chars
        session.add_tool_result_message(Message.tool_result([ToolResultContent(tool_call_id="c1", output=huge_output)]))
        session.add_assistant_message(Message.assistant("Done 1"))
        session.add_user_message("Task 2")
        session.add_assistant_message(Message.assistant("Done 2"))
        session.add_user_message("Task 3")
        session.add_assistant_message(Message.assistant("Done 3"))

        saved = session.compact()
        assert saved > 1000
        # Verify older tool result was truncated
        old_tr = session.messages[2].tool_results[0].output
        assert "prior tool output compacted" in old_tr
