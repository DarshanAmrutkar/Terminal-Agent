import asyncio
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult, resolve_safe_path
from .registry import register_tool


def _format_size(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:3.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PB"


def _list_directory_sync(target_path: Path, path_str: str, recursive: bool, max_depth: int) -> ToolResult:
    if not target_path.exists():
        return ToolResult(output=f"Error: Path '{path_str}' does not exist.", is_error=True)

    if not target_path.is_dir():
        return ToolResult(output=f"Error: Path '{path_str}' is not a directory.", is_error=True)

    ignored_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode"}
    output_lines = []
    visited: set[Path] = set()
    max_entries = 500
    truncated = False

    def walk_dir(current_path: Path, current_depth: int):
        nonlocal truncated
        if current_depth > max_depth or len(output_lines) >= max_entries:
            return

        resolved = current_path.resolve()
        if resolved in visited:
            return
        visited.add(resolved)

        try:
            entries = sorted(list(current_path.iterdir()), key=lambda e: (e.is_file(), e.name.lower()))
        except PermissionError:
            output_lines.append(f"{'  ' * current_depth}[Permission Denied] {current_path.name}")
            return

        for entry in entries:
            if len(output_lines) >= max_entries:
                truncated = True
                return

            indent = "  " * current_depth
            if entry.is_dir():
                if entry.name in ignored_dirs:
                    continue
                output_lines.append(f"{indent}📁 {entry.name}/")
                if recursive:
                    walk_dir(entry, current_depth + 1)
            else:
                try:
                    size_str = _format_size(entry.stat().st_size)
                    output_lines.append(f"{indent}📄 {entry.name} ({size_str})")
                except OSError:
                    output_lines.append(f"{indent}📄 {entry.name} (size unknown)")

    walk_dir(target_path, 0)

    if not output_lines:
        return ToolResult(output="Directory is empty or all contents are ignored.")

    result_text = "\n".join(output_lines)
    if truncated:
        result_text += f"\n... [Output truncated to {max_entries} entries to prevent context explosion]"

    return ToolResult(output=result_text)


@register_tool
class ListDirectoryTool(Tool):
    """Tool for listing the contents of a directory."""

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List files and directories in a given path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to list subdirectories recursively (default: False)"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth for recursive listing (default: 3)"
                }
            },
            "required": ["path"]
        }

    @property
    def requires_approval(self) -> bool:
        return False

    async def execute(
        self, 
        path: str, 
        recursive: bool = False, 
        max_depth: int = 3, 
        **kwargs
    ) -> ToolResult:
        target_path, err = resolve_safe_path(path)
        if err or target_path is None:
            return ToolResult(output=err or "Invalid path", is_error=True)

        try:
            return await asyncio.to_thread(_list_directory_sync, target_path, path, recursive, max_depth)
        except Exception as e:
            return ToolResult(output=f"Error listing directory '{path}': {e!s}", is_error=True)

