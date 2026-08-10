import asyncio
from typing import Any, Dict, Optional

from .base import Tool, ToolResult
from .registry import register_tool


@register_tool
class RunCommandTool(Tool):
    """Tool for executing shell commands."""

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return "Execute a shell command in the system."

    @property
    def parameters(self) -> Dict[str, Any]:
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
        cwd: Optional[str] = None, 
        timeout: int = 120, 
        **kwargs
    ) -> ToolResult:
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return ToolResult(
                    output=f"Error: Command timed out after {timeout} seconds.",
                    is_error=True
                )

            stdout = stdout_bytes.decode('utf-8', errors='replace').strip()
            stderr = stderr_bytes.decode('utf-8', errors='replace').strip()
            returncode = process.returncode

            # Truncate output if necessary (prevent context explosion)
            max_len = 10000
            if len(stdout) > max_len:
                stdout = stdout[:max_len] + f"\n... [stdout truncated from {len(stdout)} characters]"
            if len(stderr) > max_len:
                stderr = stderr[:max_len] + f"\n... [stderr truncated from {len(stderr)} characters]"

            output_blocks = []
            if stdout:
                output_blocks.append(f"STDOUT:\n{stdout}")
            if stderr:
                output_blocks.append(f"STDERR:\n{stderr}")
                
            if not output_blocks:
                result_text = "Command executed successfully with no output."
            else:
                result_text = "\n\n".join(output_blocks)

            return ToolResult(
                output=result_text,
                is_error=(returncode != 0),
                metadata={
                    "returncode": returncode,
                    "cwd": cwd
                }
            )

        except Exception as e:
            return ToolResult(
                output=f"Failed to execute command: {str(e)}", 
                is_error=True
            )
