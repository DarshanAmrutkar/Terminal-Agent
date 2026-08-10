import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult
from .registry import register_tool


@register_tool
class GrepSearchTool(Tool):
    """Tool for searching codebases using ripgrep or a Python fallback."""

    @property
    def name(self) -> str:
        return "grep_search"

    @property
    def description(self) -> str:
        return "Search for patterns in files. Uses ripgrep (rg) if available, falls back to Python regex."

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
        **kwargs
    ) -> ToolResult:
        search_path = Path(path)
        
        if not search_path.exists():
            return ToolResult(output=f"Error: Path '{path}' does not exist.", is_error=True)

        if self._has_ripgrep():
            return self._run_ripgrep(pattern, search_path, includes, case_insensitive)
        else:
            return self._run_python_fallback(pattern, search_path, includes, case_insensitive)

    def _has_ripgrep(self) -> bool:
        try:
            subprocess.run(["rg", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

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
                check=False
            )
            
            if result.returncode == 0:
                # Limit to 50 lines to prevent overload
                lines = result.stdout.splitlines()[:50]
                output = "\n".join(lines)
                if len(result.stdout.splitlines()) > 50:
                    output += "\n... [Output truncated to 50 matches]"
                return ToolResult(output=output)
            elif result.returncode == 1:
                return ToolResult(output="No matches found.")
            else:
                return ToolResult(output=f"Ripgrep error: {result.stderr}", is_error=True)
                
        except Exception as e:
            return ToolResult(output=f"Error executing ripgrep: {str(e)}", is_error=True)

    def _run_python_fallback(
        self, pattern: str, path: Path, includes: Optional[List[str]], case_insensitive: bool
    ) -> ToolResult:
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(output=f"Invalid regex pattern: {str(e)}", is_error=True)

        matches = []
        
        # Extremely basic fallback directory walk.
        # Doesn't respect .gitignore automatically, but respects limits
        if path.is_file():
            self._search_file(path, regex, matches)
        else:
            for root, _, files in os.walk(path):
                # Basic skip for common ignored dirs
                if any(x in root for x in [".git", "node_modules", "__pycache__", ".venv"]):
                    continue
                    
                for file in files:
                    # Simple glob checking if includes provided
                    if includes:
                        if not any(Path(file).match(glob) for glob in includes):
                            continue
                            
                    file_path = Path(root) / file
                    self._search_file(file_path, regex, matches)
                    
                    if len(matches) >= 50:
                        break
                if len(matches) >= 50:
                    break

        if not matches:
            return ToolResult(output="No matches found.")

        output = "\n".join(matches)
        if len(matches) == 50:
             output += "\n... [Output truncated to 50 matches]"
             
        return ToolResult(output=output)

    def _search_file(self, file_path: Path, regex: re.Pattern, matches: List[str]) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append(f"{file_path}:{i}:{line.rstrip()}")
                        if len(matches) >= 50:
                            break
        except (UnicodeDecodeError, PermissionError):
            pass
