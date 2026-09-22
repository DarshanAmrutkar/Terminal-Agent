"""Repository Map Builder: renders compact symbol outlines within token budgets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from terminal_agent.repo.ast_parser import FileSymbols, SymbolASTParser, SymbolInfo
from terminal_agent.repo.graph import SymbolDependencyGraph


DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist",
    "build", ".egg-info", ".tox", ".idea", ".vscode",
}

DEFAULT_IGNORE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib",
    ".exe", ".bin", ".tar", ".gz", ".zip", ".png", ".jpg", ".jpeg",
    ".gif", ".ico", ".svg", ".lock",
}


class RepoMapBuilder:
    """Builds a token-budgeted, PageRank-ordered symbol map of the repository."""

    def __init__(
        self,
        working_directory: Path | str = ".",
        ignore_dirs: set[str] | None = None,
        ignore_extensions: set[str] | None = None,
    ):
        self.working_dir = Path(working_directory).resolve()
        self.ignore_dirs = ignore_dirs if ignore_dirs is not None else set(DEFAULT_IGNORE_DIRS)
        self.ignore_extensions = ignore_extensions if ignore_extensions is not None else set(DEFAULT_IGNORE_EXTENSIONS)

    def build_map(
        self,
        max_tokens: int = 1500,
        seed_keywords: Sequence[str] | None = None,
        target_dir: Path | str | None = None,
    ) -> str:
        """Scan workspace and produce an elided symbol skeleton respecting token budget."""
        search_root = Path(target_dir).resolve() if target_dir else self.working_dir
        # Rough estimation: 1 token ~ 4 characters
        char_budget = max_tokens * 4

        # 1. Collect all valid files
        all_files: list[Path] = []
        for root, dirs, files in os.walk(search_root):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            for f in sorted(files):
                p = Path(root) / f
                if p.suffix.lower() not in self.ignore_extensions:
                    all_files.append(p)

        if not all_files:
            return "(empty repository)"

        # 2. Parse Python files for symbols
        parsed_files: list[FileSymbols] = []
        other_files: list[str] = []

        for p in all_files:
            try:
                rel = p.relative_to(self.working_dir).as_posix()
            except ValueError:
                rel = p.as_posix()

            if p.suffix == ".py":
                syms = SymbolASTParser.parse_file(p, rel_path=rel)
                if syms:
                    parsed_files.append(syms)
                else:
                    other_files.append(rel)
            else:
                other_files.append(rel)

        # 3. Rank files via SymbolDependencyGraph
        graph = SymbolDependencyGraph(parsed_files)
        ranks = graph.compute_ranks(seed_keywords=seed_keywords)

        # Sort parsed files descending by centrality score
        sorted_files = sorted(
            parsed_files,
            key=lambda f: ranks.get(f.rel_path, 0.0),
            reverse=True,
        )

        # 4. Greedily format symbol skeletons within character budget
        output_blocks: list[str] = []
        current_chars = 0
        included_paths: set[str] = set()

        for f in sorted_files:
            block = self._format_file_skeleton(f)
            block_len = len(block) + 2

            if current_chars + block_len > char_budget:
                break

            output_blocks.append(block)
            current_chars += block_len
            included_paths.add(f.rel_path)

        # 5. Append concise directory listing for files not expanded in outline
        remaining_files = [
            f.rel_path for f in sorted_files if f.rel_path not in included_paths
        ] + other_files

        if remaining_files:
            rem_listing = "\n".join(f"  {r}" for r in remaining_files[:100])
            if len(remaining_files) > 100:
                rem_listing += f"\n  ... and {len(remaining_files) - 100} more files"

            output_blocks.append(f"### Other Repository Files:\n{rem_listing}")

        return "\n\n".join(output_blocks)

    def _format_file_skeleton(self, f: FileSymbols) -> str:
        """Render a single file's symbols as an elided code skeleton."""
        lines = [f"# {f.rel_path} (lines 1-{f.total_lines})"]

        # Group symbols by class
        standalone_symbols = [s for s in f.symbols if not s.parent_class]
        class_symbols = [s for s in f.symbols if s.kind == "class"]
        methods_by_class: dict[str, list[SymbolInfo]] = {}
        for s in f.symbols:
            if s.parent_class:
                methods_by_class.setdefault(s.parent_class, []).append(s)

        # Render classes and their methods
        for c in class_symbols:
            for d in c.decorators:
                lines.append(d)
            lines.append(c.signature)
            if c.docstring:
                lines.append(f'    """{c.docstring}"""')

            methods = methods_by_class.get(c.name, [])
            if methods:
                for m in methods:
                    for d in m.decorators:
                        lines.append(f"    {d}")
                    lines.append(f"    {m.signature}")
                    if m.docstring:
                        lines.append(f'        """{m.docstring}"""')
                    lines.append("        ...")
            else:
                lines.append("    ...")

        # Render standalone functions
        for s in standalone_symbols:
            if s.kind != "class":
                for d in s.decorators:
                    lines.append(d)
                lines.append(s.signature)
                if s.docstring:
                    lines.append(f'    """{s.docstring}"""')
                lines.append("    ...")

        if len(lines) == 1:
            # No functions or classes in file (e.g. constant definitions or imports only)
            lines.append("    (module definitions only)")

        return "\n".join(lines)
