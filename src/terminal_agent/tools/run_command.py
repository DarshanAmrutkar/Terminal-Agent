from pathlib import Path
from typing import Any

from terminal_agent.sandbox.base import SandboxBackend, SandboxPolicy, SandboxResult
from terminal_agent.sandbox.local import LocalRestrictedSandbox

from .base import Tool, ToolResult
from .registry import register_tool


@register_tool
class RunCommandTool(Tool):
    """Tool for executing shell commands inside a secured execution sandbox."""

    def __init__(self, sandbox: SandboxBackend | None = None, working_dir: str | None = None):
        self.sandbox = sandbox
        self.working_dir = Path(working_dir).resolve() if working_dir else Path.cwd()

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return "Execute a shell command in the system."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command string to execute"
                },
                "cwd": {
                    "type": "string",
                    "description": "Current working directory for the command"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120)"
                }
            },
            "required": ["command"]
        }

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(
        self, 
        command: str, 
        cwd: str | None = None, 
        timeout: int = 120, 
        **kwargs
    ) -> ToolResult:
        try:
            # Determine active sandbox backend
            sandbox = self.sandbox
            if sandbox is None:
                effective_dir = Path(cwd).resolve() if cwd else self.working_dir
                policy = SandboxPolicy(working_dir=effective_dir, timeout_seconds=timeout)
                sandbox = LocalRestrictedSandbox(policy=policy)

            # Execute command inside sandbox
            result: SandboxResult = await sandbox.execute(
                command=command,
                cwd=cwd,
                timeout=timeout,
            )

            return ToolResult(
                output=result.formatted_output,
                is_error=result.is_error,
                metadata={
                    "returncode": result.returncode,
                    "cwd": cwd or str(self.working_dir),
                    "timed_out": result.timed_out,
                    "truncated": result.truncated,
                    "sandbox": sandbox.name,
                }
            )

        except Exception as e:
            return ToolResult(
                output=f"Failed to execute command: {e!s}", 
                is_error=True
            )
