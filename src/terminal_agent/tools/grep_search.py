import asyncio
import subprocess
from pathlib import Path
from typing import Any

from .base import (
    ALLOWED_SENSITIVE_EXCEPTIONS,
    SENSITIVE_FILE_PATTERNS,
    Tool,
    ToolResult,
    is_sensitive_path,
    resolve_safe_path,
)
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
    def parameters(self) -> dict[str, Any]:
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
        includes: list[str] | None = None,
        case_insensitive: bool = False,
    ) -> ToolResult:
        search_path, err = resolve_safe_path(path)
        if err or search_path is None:
            return ToolResult(output=err or "Invalid path", is_error=True)

        if not search_path.exists():
            return ToolResult(output=f"Error: Path '{path}' does not exist.", is_error=True)

        # Run ripgrep (or python fallback) in a thread pool to avoid blocking the event loop
        return await asyncio.to_thread(
            _run_ripgrep_with_fallback, pattern, search_path, includes, case_insensitive
        )

import fnmatch
import os
import re


def _python_grep(
    pattern: str, path: Path, includes: list[str] | None, case_insensitive: bool
) -> ToolResult:
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return ToolResult(output=f"Invalid regex pattern '{pattern}': {e}", is_error=True)

    matches = []
    ignored_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode"}

    def should_include(p: Path) -> bool:
        if not includes:
            return True
        return any(fnmatch.fnmatch(p.name, pat) for pat in includes)

    files_to_search = []
    if path.is_file():
        files_to_search.append(path)
    else:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            for file in files:
                fp = Path(root) / file
                if should_include(fp):
                    files_to_search.append(fp)

    for fp in files_to_search:
        if is_sensitive_path(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if regex.search(line):
                        clean_line = line.rstrip("\r\n")
                        matches.append(f"{fp}:{line_num}:{clean_line}")
                        if len(matches) >= 50:
                            break
        except Exception:
            continue
        if len(matches) >= 50:
            break

    if not matches:
        return ToolResult(output="No matches found.")
    output = "\n".join(matches)
    if len(matches) >= 50:
        output += "\n... [Output truncated to 50 lines]"
    return ToolResult(output=output)


def _run_ripgrep_with_fallback(
    pattern: str, path: Path, includes: list[str] | None, case_insensitive: bool
) -> ToolResult:
    cmd = ["rg", "--line-number", "--color", "never", "--max-count", "50"]

    if case_insensitive:
        cmd.append("-i")

    # Exclude sensitive files from ripgrep
    for pat in SENSITIVE_FILE_PATTERNS:
        cmd.extend(["-g", f"!{pat}"])
    for allowed in ALLOWED_SENSITIVE_EXCEPTIONS:
        cmd.extend(["-g", allowed])

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
            raw_lines = result.stdout.splitlines()
            filtered_lines: list[str] = []
            for line in raw_lines:
                parts = line.split(":", 2)
                if parts and is_sensitive_path(parts[0]):
                    continue
                filtered_lines.append(line)
                if len(filtered_lines) >= 50:
                    break

            if not filtered_lines:
                return ToolResult(output="No matches found.")

            output = "\n".join(filtered_lines)
            if len(raw_lines) > 50:
                output += "\n... [Output truncated to 50 lines]"
            return ToolResult(output=output)
        elif result.returncode == 1:
            return ToolResult(output="No matches found.")
        else:
            return ToolResult(output=f"Ripgrep error: {result.stderr}", is_error=True)

    except FileNotFoundError:
        # Fall back to pure-Python grep when 'rg' is not installed
        return _python_grep(pattern, path, includes, case_insensitive)
    except Exception as e:
        return ToolResult(output=f"Error executing ripgrep: {e!s}", is_error=True)