"""Dependency graph and centrality ranking for repository symbols."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from terminal_agent.repo.ast_parser import FileSymbols


class SymbolDependencyGraph:
    """Builds a dependency graph across repository files and ranks modules by architectural importance."""

    def __init__(self, files: Sequence[FileSymbols]):
        self.files = list(files)
        self.file_map = {f.rel_path: f for f in self.files}
        # In-degree graph: target_file -> list of files that import it
        self.in_edges: dict[str, set[str]] = defaultdict(set)
        # Out-degree graph: source_file -> list of files it imports
        self.out_edges: dict[str, set[str]] = defaultdict(set)
        self._build_graph()

    def _build_graph(self) -> None:
        """Connect files based on import statements."""
        # Index module names to relative file paths
        module_index: dict[str, str] = {}
        for f in self.files:
            p = Path(f.rel_path)
            # e.g. "foo.py" -> "foo", "bar/baz.py" -> "bar.baz"
            stem = p.stem
            mod_dotted = ".".join(p.with_suffix("").parts)
            module_index[stem] = f.rel_path
            module_index[mod_dotted] = f.rel_path

        for f in self.files:
            for imp in f.imports:
                parts = imp.split(".")
                # Check top-level and dotted module matches
                matched_path = None
                for i in range(len(parts), 0, -1):
                    prefix = ".".join(parts[:i])
                    if prefix in module_index and module_index[prefix] != f.rel_path:
                        matched_path = module_index[prefix]
                        break

                if matched_path:
                    self.out_edges[f.rel_path].add(matched_path)
                    self.in_edges[matched_path].add(f.rel_path)

    def compute_ranks(self, seed_keywords: Sequence[str] | None = None) -> dict[str, float]:
        """Compute Personalized PageRank scores for each file, boosted by seed keywords."""
        all_paths = [f.rel_path for f in self.files]
        if not all_paths:
            return {}

        n = len(all_paths)
        # 1. Compute personalized teleportation vector v
        v = {p: 1.0 / n for p in all_paths}
        if seed_keywords:
            clean_seeds = [s.lower().strip() for s in seed_keywords if s.strip()]
            for f in self.files:
                p_lower = f.rel_path.lower()
                matches = 0
                for seed in clean_seeds:
                    if seed in p_lower:
                        matches += 2
                    for sym in f.symbols:
                        if seed in sym.name.lower():
                            matches += 3
                if matches > 0:
                    v[f.rel_path] += matches * 2.0

            total_v = sum(v.values())
            if total_v > 0:
                v = {p: v[p] / total_v for p in all_paths}

        scores = dict(v)

        # 2. Iterative Personalized PageRank relaxation
        damping = 0.85
        iterations = 15

        for _ in range(iterations):
            new_scores = {}
            for p in all_paths:
                incoming_sum = 0.0
                for source in self.in_edges.get(p, set()):
                    out_degree = len(self.out_edges.get(source, set()))
                    if out_degree > 0:
                        incoming_sum += scores[source] / out_degree
                new_scores[p] = (1.0 - damping) * v[p] + damping * incoming_sum
            scores = new_scores

        # Normalize scores to [0.0, 1.0] relative to max
        max_score = max(scores.values()) if scores and max(scores.values()) > 0 else 1.0
        for k in scores:
            scores[k] = round(scores[k] / max_score, 4)

        return scores
