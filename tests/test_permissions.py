"""Tests for PermissionChecker.

These tests verify the permission decision logic in isolation, without
starting an Agent or making any LLM calls.
"""

from __future__ import annotations

import pytest

from terminal_agent.utils.permissions import (
    PermissionChecker,
    PermissionOutcome,
    SafetyLevel,
    classify_command,
)
from terminal_agent.tools.run_command import RunCommandTool
from terminal_agent.tools.read_file import ReadFileTool
from terminal_agent.llm.message import ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAFE_COMMANDS = ["ls", "git status", "git log", "cat"]
BLOCKED_PATTERNS = ["rm -rf /", "mkfs", ":(){ :|:& };:"]


def make_checker(mode: str = "safe") -> PermissionChecker:
    return PermissionChecker(
        permission_mode=mode,
        safe_commands=SAFE_COMMANDS,
        blocked_patterns=BLOCKED_PATTERNS,
    )


def make_run_call(command: str) -> ToolCall:
    return ToolCall(id="test-id", name="run_command", arguments={"command": command})


# ---------------------------------------------------------------------------
# classify_command tests
# ---------------------------------------------------------------------------

class TestClassifyCommand:
    def test_blocked_pattern_always_blocked(self):
        result = classify_command("rm -rf /", SAFE_COMMANDS, BLOCKED_PATTERNS, "safe")
        assert result == SafetyLevel.BLOCKED

    def test_blocked_pattern_in_yolo_mode_still_blocked(self):
        # Even yolo mode does not override BLOCKED patterns
        result = classify_command("rm -rf / --no-preserve-root", SAFE_COMMANDS, BLOCKED_PATTERNS, "yolo")
        assert result == SafetyLevel.BLOCKED

    def test_safe_command_exact_match(self):
        result = classify_command("git status", SAFE_COMMANDS, BLOCKED_PATTERNS, "safe")
        assert result == SafetyLevel.SAFE

    def test_safe_command_prefix_match(self):
        result = classify_command("ls -la", SAFE_COMMANDS, BLOCKED_PATTERNS, "safe")
        assert result == SafetyLevel.SAFE

    def test_read_only_command_is_safe(self):
        result = classify_command("grep -r 'import' src/", SAFE_COMMANDS, BLOCKED_PATTERNS, "safe")
        assert result == SafetyLevel.SAFE

    def test_unknown_command_needs_approval(self):
        result = classify_command("pip install requests", SAFE_COMMANDS, BLOCKED_PATTERNS, "safe")
        assert result == SafetyLevel.NEEDS_APPROVAL

    def test_yolo_mode_approves_unknown_commands(self):
        result = classify_command("pip install requests", SAFE_COMMANDS, BLOCKED_PATTERNS, "yolo")
        assert result == SafetyLevel.SAFE

    def test_auto_test_approves_pytest(self):
        result = classify_command("pytest tests/", SAFE_COMMANDS, BLOCKED_PATTERNS, "auto-test")
        assert result == SafetyLevel.SAFE

    def test_auto_test_still_needs_approval_for_other(self):
        result = classify_command("pip install requests", SAFE_COMMANDS, BLOCKED_PATTERNS, "auto-test")
        assert result == SafetyLevel.NEEDS_APPROVAL


# ---------------------------------------------------------------------------
# PermissionChecker.check() tests
# ---------------------------------------------------------------------------

class TestPermissionChecker:

    def test_read_only_tool_always_allowed(self):
        """ReadFileTool.requires_approval is False, so it bypasses all checks."""
        checker = make_checker("safe")
        tool = ReadFileTool()
        tc = ToolCall(id="1", name="read_file", arguments={"path": "/some/file.py"})
        decision = checker.check(tool, tc)
        assert decision.outcome == PermissionOutcome.ALLOW

    def test_blocked_command_is_denied(self):
        checker = make_checker("safe")
        tool = RunCommandTool()
        tc = make_run_call("rm -rf /")
        decision = checker.check(tool, tc)
        assert decision.outcome == PermissionOutcome.DENY

    def test_safe_command_is_allowed(self):
        checker = make_checker("safe")
        tool = RunCommandTool()
        tc = make_run_call("ls -la")
        decision = checker.check(tool, tc)
        assert decision.outcome == PermissionOutcome.ALLOW

    def test_unknown_command_requires_approval(self):
        checker = make_checker("safe")
        tool = RunCommandTool()
        tc = make_run_call("pip install numpy")
        decision = checker.check(tool, tc)
        assert decision.outcome == PermissionOutcome.REQUIRE_APPROVAL

    def test_yolo_mode_allows_everything_except_blocked(self):
        checker = make_checker("yolo")
        tool = RunCommandTool()
        tc = make_run_call("pip install numpy")
        decision = checker.check(tool, tc)
        assert decision.outcome == PermissionOutcome.ALLOW

    def test_yolo_mode_still_blocks_blocked_patterns(self):
        """Even yolo mode cannot override BLOCKED patterns."""
        checker = make_checker("yolo")
        tool = RunCommandTool()
        tc = make_run_call("rm -rf /")
        decision = checker.check(tool, tc)
        assert decision.outcome == PermissionOutcome.DENY

    def test_decision_includes_reason_string(self):
        """PermissionDecision must always include a human-readable reason."""
        checker = make_checker("safe")
        tool = RunCommandTool()

        # DENY
        tc = make_run_call("rm -rf /")
        d = checker.check(tool, tc)
        assert d.reason != ""

        # ALLOW
        tc2 = make_run_call("ls")
        d2 = checker.check(tool, tc2)
        assert d2.reason != ""

        # REQUIRE_APPROVAL
        tc3 = make_run_call("pip install numpy")
        d3 = checker.check(tool, tc3)
        assert d3.reason != ""
