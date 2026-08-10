import os
from pathlib import Path
from typing import Any, Dict

from .base import Tool, ToolResult
from .registry import register_tool


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
    def parameters(self) -> Dict[str, Any]:
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

    def _format_size(self, size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:3.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}PB"

    async def execute(
        self, 
        path: str, 
        recursive: bool = False, 
        max_depth: int = 3, 
        **kwargs
    ) -> ToolResult:
        target_path = Path(path)

        if not target_path.exists():
            return ToolResult(output=f"Error: Path '{path}' does not exist.", is_error=True)

        if not target_path.is_dir():
            return ToolResult(output=f"Error: Path '{path}' is not a directory.", is_error=True)

        ignored_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode"}
        output_lines = []

        def walk_dir(current_path: Path, current_depth: int):
            if current_depth > max_depth:
                return

            try:
                entries = sorted(list(current_path.iterdir()), key=lambda e: (e.is_file(), e.name.lower()))
            except PermissionError:
                output_lines.append(f"{'  ' * current_depth}[Permission Denied] {current_path.name}")
                return

            for entry in entries:
                indent = "  " * current_depth
                if entry.is_dir():
                    if entry.name in ignored_dirs:
                        continue
                    output_lines.append(f"{indent}📁 {entry.name}/")
                    if recursive:
                        walk_dir(entry, current_depth + 1)
                else:
                    try:
                        size_str = self._format_size(entry.stat().st_size)
                        output_lines.append(f"{indent}📄 {entry.name} ({size_str})")
                    except OSError:
                        output_lines.append(f"{indent}📄 {entry.name} (size unknown)")

        walk_dir(target_path, 0)
        
        if not output_lines:
            return ToolResult(output="Directory is empty or all contents are ignored.")
            
        return ToolResult(output="\n".join(output_lines))
