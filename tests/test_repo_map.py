"""Unit tests for AST repository map and symbol extraction engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from terminal_agent.repo.ast_parser import SymbolASTParser
from terminal_agent.repo.graph import SymbolDependencyGraph
from terminal_agent.repo.map_builder import RepoMapBuilder
from terminal_agent.tools.repo_map import RepoMapTool

SAMPLE_CODE = '''"""Sample module docstring."""

from typing import List, Optional
import os

class DatabaseClient:
    """Manages connections to database."""

    def __init__(self, host: str, port: int = 5432) -> None:
        self.host = host
        self.port = port

    async def query(self, sql: str, params: Optional[dict] = None) -> List[dict]:
        """Execute query."""
        return []

def calculate_checksum(data: bytes) -> str:
    """Compute sha256 checksum."""
    return "checksum"
'''


def test_symbol_ast_parser():
    file_syms = SymbolASTParser.parse_source(SAMPLE_CODE, rel_path="db.py")

    assert file_syms.rel_path == "db.py"
    assert "os" in file_syms.imports
    assert "typing.List" in file_syms.imports or "List" in file_syms.imports

    symbols = file_syms.symbols
    assert len(symbols) == 4  # Class, __init__, query, calculate_checksum

    # Verify Class
    cls_sym = next(s for s in symbols if s.kind == "class")
    assert cls_sym.name == "DatabaseClient"
    assert cls_sym.docstring == "Manages connections to database."
    assert "class DatabaseClient:" in cls_sym.signature

    # Verify Method with type annotations
    query_sym = next(s for s in symbols if s.name == "query")
    assert query_sym.kind == "method"
    assert query_sym.parent_class == "DatabaseClient"
    assert "async def query(" in query_sym.signature
    assert "sql: str" in query_sym.signature
    assert "-> List[dict]:" in query_sym.signature

    # Verify standalone function
    func_sym = next(s for s in symbols if s.name == "calculate_checksum")
    assert func_sym.kind == "function"
    assert func_sym.docstring == "Compute sha256 checksum."


def test_symbol_ast_parser_syntax_error():
    broken_code = "def incomplete_func(\n"
    res = SymbolASTParser.parse_source(broken_code, rel_path="bad.py")
    assert res.rel_path == "bad.py"
    assert len(res.symbols) == 0


def test_symbol_dependency_graph_ranking():
    # File A: utils.py (no imports)
    f_utils = SymbolASTParser.parse_source(
        "def helper(): pass\n", rel_path="utils.py"
    )
    # File B: service.py (imports utils)
    f_service = SymbolASTParser.parse_source(
        "from utils import helper\ndef run(): helper()\n", rel_path="service.py"
    )
    # File C: app.py (imports service and utils)
    f_app = SymbolASTParser.parse_source(
        "import service\nimport utils\n", rel_path="app.py"
    )

    graph = SymbolDependencyGraph([f_utils, f_service, f_app])
    ranks = graph.compute_ranks()

    # utils is imported by both service and app, so it should rank highly
    assert ranks["utils.py"] >= ranks["app.py"]

    # Keyword boosting
    boosted = graph.compute_ranks(seed_keywords=["service"])
    assert boosted["service.py"] > ranks["service.py"]


def test_repo_map_builder():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "math_utils.py").write_text(
            "def add(a: int, b: int) -> int:\n    '''Add two integers.'''\n    return a + b\n",
            encoding="utf-8",
        )
        (root / "calculator.py").write_text(
            "from math_utils import add\nclass Calc:\n    def compute(self, x: int) -> int:\n        return add(x, 10)\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text("# Project Docs\n", encoding="utf-8")

        builder = RepoMapBuilder(working_directory=root)
        repo_map = builder.build_map(max_tokens=500)

        # Verify skeletons are present
        assert "math_utils.py" in repo_map
        assert "def add(a: int, b: int) -> int:" in repo_map
        assert "class Calc:" in repo_map
        assert "def compute(self, x: int) -> int:" in repo_map
        assert "README.md" in repo_map


@pytest.mark.asyncio
async def test_repo_map_tool():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "main.py").write_text(
            "def start() -> None:\n    '''Start application.'''\n    pass\n",
            encoding="utf-8",
        )

        tool = RepoMapTool(working_dir=str(root))
        result = await tool.execute()

        assert not result.is_error
        assert "main.py" in result.output
        assert "def start() -> None:" in result.output
        assert '"""Start application."""' in result.output
