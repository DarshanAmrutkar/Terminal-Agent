import difflib
from pathlib import Path
from typing import Any, Dict

from .base import Tool, ToolResult
from .registry import register_tool


@register_tool
class SearchReplaceTool(Tool):
    """Tool for editing a file by searching for exact text and replacing it."""

    @property
    def name(self) -> str:
        return "search_replace"

    @property
    def description(self) -> str:
        return "Replace an exact block of text in a file with new text."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to modify"
                },
                "search": {
                    "type": "string",
                    "description": "The exact text to search for and replace"
                },
                "replace": {
                    "type": "string",
                    "description": "The new text to insert in place of the search text"
                }
            },
            "required": ["path", "search", "replace"]
        }

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(self, path: str, search: str, replace: str, **kwargs) -> ToolResult:
        try:
            file_path = Path(path)
            
            if not file_path.exists() or not file_path.is_file():
                return ToolResult(
                    output=f"Error: File '{path}' does not exist or is not a file.",
                    is_error=True
                )
                
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return ToolResult(output=f"Error: Could not read '{path}' as utf-8 text.", is_error=True)

            # Try exact match first
            if search in content:
                new_content = content.replace(search, replace, 1)
                file_path.write_text(new_content, encoding="utf-8")
                return self._build_success_result(content, search, replace)
                
            # If exact match fails, try fuzzy match (strip trailing spaces per line)
            search_lines = [line.rstrip() for line in search.splitlines()]
            content_lines = content.splitlines()
            content_lines_stripped = [line.rstrip() for line in content_lines]
            
            # Find the best match
            search_joined = "\n".join(search_lines)
            content_joined = "\n".join(content_lines_stripped)
            
            if search_joined in content_joined:
                # We found a match ignoring trailing whitespace.
                # However, a simple replace on stripped strings destroys original whitespace.
                # So we must find the index of the matching chunk in the original lines.
                match_idx = -1
                search_len = len(search_lines)
                
                for i in range(len(content_lines_stripped) - search_len + 1):
                    if content_lines_stripped[i:i+search_len] == search_lines:
                        match_idx = i
                        break
                        
                if match_idx != -1:
                    # Construct new content by substituting the lines
                    replace_lines = replace.splitlines()
                    new_lines = content_lines[:match_idx] + replace_lines + content_lines[match_idx + search_len:]
                    new_content = "\n".join(new_lines)
                    # Add back trailing newline if it originally existed
                    if content.endswith("\n") and not new_content.endswith("\n"):
                        new_content += "\n"
                        
                    file_path.write_text(new_content, encoding="utf-8")
                    return self._build_success_result(content, search, replace)
                    
            # If we still haven't found a match, provide a clear error with diff
            return self._build_error_result(content, search)

        except Exception as e:
            return ToolResult(output=f"Error executing search and replace: {str(e)}", is_error=True)

    def _build_success_result(self, original: str, search: str, replace: str) -> ToolResult:
        return ToolResult(
            output="Successfully replaced text block.\nNote: Showing a basic success indicator.",
            metadata={"replaced": True}
        )
        
    def _build_error_result(self, original: str, search: str) -> ToolResult:
        # Give closest partial match info
        matcher = difflib.SequenceMatcher(None, original, search)
        match = matcher.find_longest_match(0, len(original), 0, len(search))
        
        closest_match = ""
        if match.size > 10:  # Arbitrary threshold to show some partial context
            closest_match = original[match.a:match.a+match.size]
            
        error_msg = "Error: Could not find exact or fuzzy match for the search string.\n"
        error_msg += "Check indentation, whitespace, and line endings.\n"
        if closest_match:
            error_msg += f"Closest partial match found was:\n---\n{closest_match}\n---"
            
        return ToolResult(output=error_msg, is_error=True)
