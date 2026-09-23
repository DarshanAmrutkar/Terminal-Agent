"""Tool for generating and querying the repository symbol map."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from terminal_agent.repo.map_builder import RepoMapBuilder
from terminal_agent.tools.base import resolve_safe_path

from .base import Tool, ToolResult
from .registry import register_tool


@register_tool
class RepoMapTool(Tool):
    """Tool that returns a structural symbol map of code in the repository."""

    def __init__(self, working_dir: str | None = None):
        self.working_dir = Path(working_dir).resolve() if working_dir else Path.cwd()

    @property
    def name(self) -> str:
        return "get_repo_map"

    @property
    def description(self) -> str:
        return (
            "Inspect the architectural symbol outline (classes, methods, function signatures, "
            "and line numbers) of files in the repository without reading entire files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Optional subdirectory to restrict the symbol map to (e.g. 'src/models')",
                },
                "keywords": {
                    "type": "string",
                    "description": "Optional comma-separated keywords to prioritize in the symbol ranking (e.g. 'auth, token')",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum token budget for the skeleton output (default: 1500)",
                },
            },
        }

    @property
    def requires_approval(self) -> bool:
        return False  # Read-only tool

    async def execute(
        self,
        directory: str | None = None,
        keywords: str | None = None,
        max_tokens: int = 1500,
        **kwargs,
    ) -> ToolResult:
        try:
            target_dir = None
            if directory:
                safe_dir, err = resolve_safe_path(directory, base_dir=self.working_dir)
                if safe_dir is None or not safe_dir.is_dir():
                    return ToolResult(
                        output=f"Error: Target directory '{directory}' does not exist or is outside workspace.",
                        is_error=True,
                    )
                target_dir = safe_dir

            seed_keywords = [k.strip() for k in keywords.split(",")] if keywords else None

            builder = RepoMapBuilder(working_directory=self.working_dir)
            outline = builder.build_map(
                max_tokens=max_tokens,
                seed_keywords=seed_keywords,
                target_dir=target_dir,
            )

            return ToolResult(output=outline, is_error=False)

        except Exception as e:
            return ToolResult(
                output=f"Failed to generate repository map: {e!s}",
                is_error=True,
            )
