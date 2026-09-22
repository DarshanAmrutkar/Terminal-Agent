import asyncio
import difflib
import os
from pathlib import Path
from typing import Any, Dict
import uuid

from .base import Tool, ToolResult, resolve_safe_path
from .registry import register_tool


def _search_replace_sync(file_path: Path, path_str: str, search: str, replace: str) -> ToolResult:
    if not file_path.exists() or not file_path.is_file():
        return ToolResult(
            output=f"Error: File '{path_str}' does not exist or is not a file.",
            is_error=True
        )

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ToolResult(output=f"Error: Could not read '{path_str}' as utf-8 text.", is_error=True)

    uses_crlf = "\r\n" in content
    # Work with normalized LF
    norm_content = content.replace("\r\n", "\n")
    norm_search = search.replace("\r\n", "\n")
    norm_replace = replace.replace("\r\n", "\n")

    # 1. Exact match check
    match_count = norm_content.count(norm_search)
    if match_count > 1:
        return ToolResult(
            output=f"Error: Search text occurs {match_count} times in '{path_str}'. "
                   f"Please provide more surrounding lines to uniquely identify the block to replace.",
            is_error=True
        )

    new_content_norm = None
    if match_count == 1:
        new_content_norm = norm_content.replace(norm_search, norm_replace, 1)
    else:
        # 2. Fuzzy match (ignoring trailing whitespace per line)
        search_lines = [line.rstrip() for line in norm_search.splitlines()]
        content_lines = norm_content.splitlines()
        content_lines_stripped = [line.rstrip() for line in content_lines]

        search_joined = "\n".join(search_lines)
        content_joined = "\n".join(content_lines_stripped)

        fuzzy_count = content_joined.count(search_joined)
        if fuzzy_count > 1:
            return ToolResult(
                output=f"Error: Search text occurs {fuzzy_count} times (fuzzy) in '{path_str}'. "
                       f"Please provide more surrounding lines.",
                is_error=True
            )

        if fuzzy_count == 1:
            search_len = len(search_lines)
            match_idx = -1
            for i in range(len(content_lines_stripped) - search_len + 1):
                if content_lines_stripped[i:i+search_len] == search_lines:
                    match_idx = i
                    break

            if match_idx != -1:
                replace_lines = norm_replace.splitlines()
                new_lines = content_lines[:match_idx] + replace_lines + content_lines[match_idx + search_len:]
                new_content_norm = "\n".join(new_lines)
                if norm_content.endswith("\n") and not new_content_norm.endswith("\n"):
                    new_content_norm += "\n"

    if new_content_norm is not None:
        # Convert back to original line ending style if needed
        final_content = new_content_norm.replace("\n", "\r\n") if uses_crlf else new_content_norm

        # Atomic write
        tmp_file = file_path.parent / f".tmp_{uuid.uuid4().hex}_{file_path.name}"
        try:
            tmp_file.write_text(final_content, encoding="utf-8")
            tmp_file.replace(file_path)
        finally:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass

        # Generate a unified diff for clarity
        diff = list(difflib.unified_diff(
            content.splitlines(keepends=True),
            final_content.splitlines(keepends=True),
            fromfile=f"a/{path_str}",
            tofile=f"b/{path_str}",
            n=3
        ))
        diff_str = "".join(diff[:100])
        return ToolResult(
            output=f"Successfully replaced text block in '{path_str}'.\nDiff:\n{diff_str}",
            metadata={"replaced": True}
        )

    # 3. No match found — compute partial match preview safely without O(N*M) freeze
    # Limit search space to 30,000 characters to prevent high latency
    sample = norm_content[:30000]
    matcher = difflib.SequenceMatcher(None, sample, norm_search[:1000])
    match = matcher.find_longest_match(0, len(sample), 0, min(len(norm_search), 1000))

    closest_match = ""
    if match.size > 10:
        closest_match = sample[match.a:match.a+match.size]

    error_msg = f"Error: Could not find match for the search string in '{path_str}'.\n"
    error_msg += "Check indentation, whitespace, and surrounding context.\n"
    if closest_match:
        error_msg += f"Closest partial match found was:\n---\n{closest_match}\n---"

    return ToolResult(output=error_msg, is_error=True)


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
        file_path, err = resolve_safe_path(path)
        if err or file_path is None:
            return ToolResult(output=err or "Invalid path", is_error=True)

        try:
            return await asyncio.to_thread(_search_replace_sync, file_path, path, search, replace)
        except Exception as e:
            return ToolResult(output=f"Error executing search and replace: {str(e)}", is_error=True)

