import asyncio
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult
from .registry import register_tool


@register_tool
class GrepSearchTool(Tool):
    """Tool for searching codebases using ripgrep."""

    @property
    def name(self) -> str:
        return "grep_search"

    @property
    def description(self) -> str:
        return "Search for patterns in files using ripgrep (rg)."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (defaults to current directory)"
                },
                "includes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns to include (e.g., ['*.py'])"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case-insensitive"
                }
            },
            "required": ["pattern"]
        }

    @property
    def requires_approval(self) -> bool:
        return False

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        includes: Optional[List[str]] = None,
        case_insensitive: bool = False,
    ) -> ToolResult:
        search_path = Path(path)

        if not search_path.exists():
            return ToolResult(output=f"Error: Path '{path}' does not exist.", is_error=True)

        # Run ripgrep in a thread pool to avoid blocking the event loop
        return await asyncio.to_thread(
            self._run_ripgrep, pattern, search_path, includes, case_insensitive
        )

    def _run_ripgrep(
        self, pattern: str, path: Path, includes: Optional[List[str]], case_insensitive: bool
    ) -> ToolResult:
        cmd = ["rg", "--line-number", "--color", "never", "--max-count", "50"]

        if case_insensitive:
            cmd.append("-i")

        if includes:
            for glob in includes:
                cmd.extend(["-g", glob])

        cmd.extend(["--", pattern, str(path)])

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                # ripgrep already limits to 50 matches per file (--max-count 50),
                # but we also truncate total output lines to prevent overload
                lines = result.stdout.splitlines()[:50]
                output = "\n".join(lines)
                if len(result.stdout.splitlines()) > 50:
                    output += "\n... [Output truncated to 50 lines]"
                return ToolResult(output=output)
            elif result.returncode == 1:
                return ToolResult(output="No matches found.")
            else:
                return ToolResult(output=f"Ripgrep error: {result.stderr}", is_error=True)

        except FileNotFoundError:
            return ToolResult(
                output="Error: 'rg' (ripgrep) command not found. Please install ripgrep to use grep_search.",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(output=f"Error executing ripgrep: {str(e)}", is_error=True)