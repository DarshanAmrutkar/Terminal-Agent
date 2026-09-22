import asyncio
import os
from pathlib import Path
from typing import Any, Dict
import uuid

from .base import Tool, ToolResult, resolve_safe_path
from .registry import register_tool


def _write_file_sync(file_path: Path, content: str, create_dirs: bool) -> tuple[int, bool]:
    is_new = not file_path.exists()
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write using a temporary file in the same directory
    tmp_file = file_path.parent / f".tmp_{uuid.uuid4().hex}_{file_path.name}"
    try:
        tmp_file.write_text(content, encoding="utf-8")
        tmp_file.replace(file_path)
    finally:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception:
                pass
    return len(content.encode("utf-8")), is_new


@register_tool
class WriteFileTool(Tool):
    """Tool for writing or overwriting files."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Create a new file or overwrite an existing file with the provided content."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path where the file should be written"
                },
                "content": {
                    "type": "string",
                    "description": "The full text content to write to the file"
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "Whether to create parent directories if they don't exist. Defaults to True."
                }
            },
            "required": ["path", "content"]
        }

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(
        self, 
        path: str, 
        content: str, 
        create_dirs: bool = True, 
        **kwargs
    ) -> ToolResult:
        file_path, err = resolve_safe_path(path)
        if err or file_path is None:
            return ToolResult(output=err or "Invalid path", is_error=True)

        try:
            bytes_written, is_new = await asyncio.to_thread(
                _write_file_sync, file_path, content, create_dirs
            )
            action = "Created new file" if is_new else "Overwrote existing file"
            return ToolResult(
                output=f"Successfully {action.lower()} at '{path}'. Wrote {bytes_written} bytes.",
                metadata={
                    "bytes_written": bytes_written,
                    "is_new": is_new,
                    "path": str(file_path)
                }
            )
        except PermissionError:
            return ToolResult(
                output=f"Error: Permission denied when writing to '{path}'.",
                is_error=True
            )
        except Exception as e:
            return ToolResult(
                output=f"An unexpected error occurred while writing to '{path}': {str(e)}", 
                is_error=True
            )

