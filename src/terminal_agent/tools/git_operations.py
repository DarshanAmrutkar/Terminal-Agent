"""Git operations tool for Terminal Agent.

Provides the LLM with access to git operations: status, diff, log, branch, commit, blame.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict

from terminal_agent.tools.base import Tool, ToolResult
from terminal_agent.tools.registry import register_tool
from terminal_agent.repo.git import GitRepo


@register_tool
class GitOperationsTool(Tool):
    """Tool to perform git operations."""

    @property
    def name(self) -> str:
        return "git_operations"

    @property
    def description(self) -> str:
        return "Perform git operations: status, diff, log, branch, commit, blame"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["status", "diff", "log", "branch", "commit", "blame"],
                    "description": "The git operation to perform.",
                },
                "args": {
                    "type": "object",
                    "description": (
                        "Optional arguments for the operation. Examples: "
                        "{'staged': true} for diff, {'n': 5} for log, "
                        "{'message': 'fix: bug'} for commit, {'path': 'file.py'} for blame."
                    ),
                },
            },
            "required": ["operation"],
        }

    @property
    def requires_approval(self) -> bool:
        return False

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute a git operation."""
        operation = kwargs.get("operation")
        args = kwargs.get("args") or {}

        cwd = os.getcwd()
        repo = GitRepo(cwd)

        if not repo.is_git_repo():
            return ToolResult(output="Error: Not a git repository.", is_error=True)

        loop = asyncio.get_event_loop()

        def _run_op() -> str:
            try:
                if operation == "status":
                    return repo.status()
                elif operation == "diff":
                    staged = args.get("staged", False)
                    return repo.diff(staged=staged)
                elif operation == "log":
                    n = args.get("n", 10)
                    return repo.log(n=n)
                elif operation == "branch":
                    return repo.current_branch()
                elif operation == "commit":
                    message = args.get("message")
                    if not message:
                        return "Error: commit requires a 'message' argument."
                    add_all = args.get("add_all", True)
                    return repo.commit(message=message, add_all=add_all)
                elif operation == "blame":
                    path = args.get("path")
                    if not path:
                        return "Error: blame requires a 'path' argument."
                    return repo.blame(path=path)
                else:
                    return f"Error: unknown operation '{operation}'"
            except Exception as e:
                return f"Error: {str(e)}"

        output = await loop.run_in_executor(None, _run_op)

        is_error = output.startswith("Error")
        return ToolResult(output=output, is_error=is_error)
