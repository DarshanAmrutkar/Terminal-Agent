import os
from pathlib import Path
from typing import Any, Dict, Optional

from .base import Tool, ToolResult
from .registry import register_tool


@register_tool
class ReadFileTool(Tool):
    """Tool for reading files or specific line ranges from files."""
    
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file, optionally restricted to a line range."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed). Inclusive."
                },
                "end_line": {
                    "type": "integer",
                    "description": "Ending line number. Inclusive."
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
        start_line: Optional[int] = None, 
        end_line: Optional[int] = None, 
        **kwargs
    ) -> ToolResult:
        try:
            file_path = Path(path)
            
            if not file_path.exists():
                return ToolResult(
                    output=f"Error: File not found at '{path}'", 
                    is_error=True
                )
                
            if not file_path.is_file():
                return ToolResult(
                    output=f"Error: Path '{path}' is not a file.", 
                    is_error=True
                )
                
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                return ToolResult(
                    output=f"Error: File '{path}' appears to be binary or has an unsupported encoding.",
                    is_error=True
                )
            except PermissionError:
                return ToolResult(
                    output=f"Error: Permission denied when accessing '{path}'.",
                    is_error=True
                )

            total_lines = len(lines)
            start = max(1, start_line) if start_line is not None else 1
            end = min(total_lines, end_line) if end_line is not None else total_lines
            
            if start > end or start > total_lines:
                return ToolResult(
                    output=f"Error: Invalid line range {start}-{end} for file with {total_lines} lines.",
                    is_error=True
                )
                
            max_lines = 2000
            truncated = False
            if (end - start + 1) > max_lines:
                end = start + max_lines - 1
                truncated = True

            output_lines = []
            for i in range(start - 1, end):
                # Ensure we format the line number consistently
                # e.g., '  1 | import os'
                line_content = lines[i].rstrip('\r\n')
                output_lines.append(f"{i + 1:4d} | {line_content}")

            result_text = "\n".join(output_lines)
            if truncated:
                result_text += f"\n... [Warning: Output truncated to {max_lines} lines to prevent context explosion]"

            return ToolResult(
                output=result_text,
                metadata={"total_lines": total_lines, "lines_returned": end - start + 1}
            )

        except Exception as e:
            return ToolResult(
                output=f"An unexpected error occurred while reading '{path}': {str(e)}", 
                is_error=True
            )
