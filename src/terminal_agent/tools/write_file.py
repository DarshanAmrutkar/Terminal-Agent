import os
from pathlib import Path
from typing import Any, Dict

from .base import Tool, ToolResult
from .registry import register_tool


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
        try:
            file_path = Path(path)
            
            is_new = not file_path.exists()
            
            if create_dirs:
                try:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    return ToolResult(
                        output=f"Error: Permission denied when creating directories for '{path}'.",
                        is_error=True
                    )
                except Exception as e:
                    return ToolResult(
                        output=f"Error creating directories for '{path}': {str(e)}",
                        is_error=True
                    )
            
            try:
                # Write the file, ensuring utf-8
                bytes_written = file_path.write_text(content, encoding="utf-8")
            except PermissionError:
                return ToolResult(
                    output=f"Error: Permission denied when writing to '{path}'.",
                    is_error=True
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

        except Exception as e:
            return ToolResult(
                output=f"An unexpected error occurred while writing to '{path}': {str(e)}", 
                is_error=True
            )
