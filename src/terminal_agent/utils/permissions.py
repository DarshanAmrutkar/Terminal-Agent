"""Permission system for tool execution.

Responsibilities:
  - Classify how safe a command or tool invocation is.
  - Decide whether to allow, require approval for, or block an invocation.
  - Keep the Agent loop free of tool-name-specific logic.

Design: Why a PermissionChecker class instead of functions?
-----------------------------------------------------------
The original code had `classify_command()` as a module-level function and
placed the approval logic directly in Agent._execute_single_tool(), including
a hard-coded `if tool_call.name == "run_command"` check.

Problems with that approach:
  1. Agent must know the *names* of tools that need special handling.
     Adding a new shell-like tool (e.g., "run_python") requires editing Agent.
     This violates the Open/Closed Principle.
  2. The approval decision (a policy) was interleaved with the execution
     flow in Agent, which is already responsible for orchestrating the loop.
     Mixing policy with orchestration makes both harder to test and reason about.

The new design:
  - PermissionChecker owns the decision: given a (Tool, ToolCall, config),
    what should happen?
  - Agent calls checker.check() and receives a PermissionDecision.
  - Agent never references specific tool names.
  - To add a new tool with command-like behavior, you extend PermissionChecker's
    logic (or give the tool a 'command_key' argument) \u2014 Agent stays unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from terminal_agent.tools.base import Tool, ToolResult
from terminal_agent.llm.message import ToolCall


class SafetyLevel(Enum):
    """Safety classification for a command or tool invocation."""
    SAFE = "SAFE"               # auto-approve
    NEEDS_APPROVAL = "NEEDS_APPROVAL"  # ask the user
    BLOCKED = "BLOCKED"         # reject without asking


class PermissionOutcome(Enum):
    """The result of a permission check."""
    ALLOW = "allow"       # proceed with execution
    DENY = "deny"         # reject \u2014 return error to LLM
    REQUIRE_APPROVAL = "require_approval"  # prompt user before proceeding


@dataclass
class PermissionDecision:
    """Encapsulates the outcome of a permission check.

    Attributes:
        outcome:  What the agent should do.
        reason:   Human-readable explanation (shown to user or logged).
    """
    outcome: PermissionOutcome
    reason: str = ""


def classify_command(command: str, safe_commands: list[str], blocked_patterns: list[str], permission_mode: str) -> SafetyLevel:
    """Classify a raw shell command string's safety level.

    Args:
        command:          The command string the agent wants to execute.
        safe_commands:    Commands/prefixes that are always auto-approved.
        blocked_patterns: Substrings that cause immediate rejection.
        permission_mode:  'safe', 'auto-test', or 'yolo'.

    Returns:
        SafetyLevel indicating how to handle the command.
    """
    cmd = command.strip()

    # Check blocked patterns first \u2014 these are always rejected regardless of mode.
    for pattern in blocked_patterns:
        if pattern in cmd:
            return SafetyLevel.BLOCKED

    # In yolo mode, everything that isn\u2019t blocked is auto-approved.
    if permission_mode == "yolo":
        return SafetyLevel.SAFE

    # Check explicitly safe commands / prefixes.
    cmd_base = cmd.split()[0] if cmd else ""
    if cmd_base in safe_commands or cmd in safe_commands:
        return SafetyLevel.SAFE

    # Read-only commands are generally safe.
    read_only_starts = (
        "ls", "dir", "cat", "type", "echo", "pwd", "grep", "find",
        "head", "tail", "less", "more", "wc", "sort", "uniq", "which", "where",
    )
    if cmd_base in read_only_starts:
        return SafetyLevel.SAFE

    # In auto-test mode, test runners are auto-approved.
    if permission_mode == "auto-test":
        test_runners = ("pytest", "python -m pytest", "npm test", "yarn test", "jest")
        if any(cmd.startswith(runner) for runner in test_runners):
            return SafetyLevel.SAFE

    return SafetyLevel.NEEDS_APPROVAL


class PermissionChecker:
    """Decides whether a tool invocation should proceed, be blocked, or need approval.

    The checker is the single place that understands permission policy. Agent
    calls check() and acts on the PermissionDecision without knowing any
    tool-name-specific rules.

    Why is this a class and not a function?
    State: the checker holds config (safe_commands, blocked_patterns, mode).
    Testability: you can instantiate PermissionChecker with controlled config in tests.
    Extensibility: future checks (e.g., per-user rate limits, file access auditing)
                   can be added as methods without touching Agent.
    """

    # The argument name that run_command-style tools use to pass the shell command.
    # This is the only coupling to tool implementation: we expect that any tool
    # with a 'command' argument in its tool call is shell-executing.
    COMMAND_ARGUMENT = "command"

    def __init__(
        self,
        permission_mode: str,
        safe_commands: list[str],
        blocked_patterns: list[str],
    ) -> None:
        self.permission_mode = permission_mode
        self.safe_commands = safe_commands
        self.blocked_patterns = blocked_patterns

    def check(self, tool: Tool, tool_call: ToolCall) -> PermissionDecision:
        """Evaluate whether a tool call should be allowed to execute.

        Decision tree:
          1. If the tool does not require approval \u2192 ALLOW immediately.
          2. If permission mode is 'yolo' \u2192 ALLOW immediately.
          3. If the tool call has a 'command' argument, classify that command:
               - BLOCKED  \u2192 DENY
               - SAFE     \u2192 ALLOW
               - NEEDS_APPROVAL \u2192 REQUIRE_APPROVAL
          4. Otherwise (non-command tool that requires_approval) \u2192 REQUIRE_APPROVAL.

        Args:
            tool:       The Tool instance about to execute.
            tool_call:  The ToolCall with name and arguments.

        Returns:
            PermissionDecision describing what the agent should do.
        """
        # Fast path: tool doesn't require approval at all.
        if not tool.requires_approval:
            return PermissionDecision(outcome=PermissionOutcome.ALLOW)

        # Check if this tool call is invoking a shell command.
        # We do this BEFORE checking yolo mode because BLOCKED patterns are a
        # hard safety floor — they cannot be overridden even in yolo mode.
        # The philosophy: yolo removes the "approval prompt" friction, but does
        # not disable safety rails that exist to prevent catastrophic mistakes.
        command = tool_call.arguments.get(self.COMMAND_ARGUMENT)
        if command is not None:
            safety = classify_command(
                command=command,
                safe_commands=self.safe_commands,
                blocked_patterns=self.blocked_patterns,
                permission_mode=self.permission_mode,
            )
            if safety == SafetyLevel.BLOCKED:
                return PermissionDecision(
                    outcome=PermissionOutcome.DENY,
                    reason=f"Command blocked by safety policy: {command!r}",
                )
            if safety == SafetyLevel.SAFE:
                return PermissionDecision(
                    outcome=PermissionOutcome.ALLOW,
                    reason=f"Command classified as safe: {command!r}",
                )
            # SafetyLevel.NEEDS_APPROVAL → fall through to check yolo/prompt

        # Fast path: user has disabled all approval prompts.
        # This only runs after BLOCKED patterns have been checked above.
        if self.permission_mode == "yolo":
            return PermissionDecision(
                outcome=PermissionOutcome.ALLOW,
                reason="yolo mode: command auto-approved",
            )

        # Non-command tool that still requires approval (e.g., write_file).
        if command is None:
            return PermissionDecision(
                outcome=PermissionOutcome.REQUIRE_APPROVAL,
                reason=f"Tool '{tool_call.name}' requires user approval before executing.",
            )

        # Command that needs approval (NEEDS_APPROVAL, not yolo mode)
        return PermissionDecision(
            outcome=PermissionOutcome.REQUIRE_APPROVAL,
            reason=f"Command requires user approval: {command!r}",
        )
