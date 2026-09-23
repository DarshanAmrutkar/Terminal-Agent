"""AST parsing and symbol extraction for repository intelligence."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SymbolInfo:
    """Represents a code symbol (class, function, or method)."""
    name: str
    kind: str  # "class", "function", "async_function", "method"
    line_number: int
    end_line: int
    signature: str
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    parent_class: str | None = None


@dataclass
class FileSymbols:
    """All extracted symbols and imports for a single source file."""
    rel_path: str
    symbols: list[SymbolInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    total_lines: int = 0


class SymbolASTParser:
    """Parses Python source files using standard library AST to extract signatures and structure."""

    @classmethod
    def parse_file(cls, file_path: Path, rel_path: str = "") -> FileSymbols | None:
        """Read and parse a file from disk."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return cls.parse_source(content, rel_path=rel_path or str(file_path))
        except Exception:
            return None

    @classmethod
    def parse_source(cls, source: str, rel_path: str = "") -> FileSymbols:
        """Parse source code string into FileSymbols."""
        lines = source.splitlines()
        total_lines = len(lines)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            # If invalid syntax, return empty structure with line count
            return FileSymbols(rel_path=rel_path, total_lines=total_lines)

        symbols: list[SymbolInfo] = []
        imports: list[str] = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.extend(cls._extract_imports(node))

            elif isinstance(node, ast.ClassDef):
                symbols.extend(cls._extract_class(node))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(cls._extract_function(node))

        return FileSymbols(
            rel_path=rel_path,
            symbols=symbols,
            imports=imports,
            total_lines=total_lines,
        )

    @classmethod
    def _extract_imports(cls, node: ast.Import | ast.ImportFrom) -> list[str]:
        """Extract imported module and symbol names."""
        results: list[str] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                results.append(f"{mod}.{alias.name}" if mod else alias.name)
        return results

    @classmethod
    def _extract_function(
        cls,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_class: str | None = None,
    ) -> SymbolInfo:
        """Extract function signature, return type, and docstring."""
        kind = "method" if parent_class else ("async_function" if isinstance(node, ast.AsyncFunctionDef) else "function")
        prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "

        # Format arguments with type annotations
        params = []
        for arg in node.args.posonlyargs + node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            params.append(arg_str)

        if node.args.vararg:
            v = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                try:
                    v += f": {ast.unparse(node.args.vararg.annotation)}"
                except Exception:
                    pass
            params.append(v)

        for arg in node.args.kwonlyargs:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            params.append(arg_str)

        if node.args.kwarg:
            kw = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                try:
                    kw += f": {ast.unparse(node.args.kwarg.annotation)}"
                except Exception:
                    pass
            params.append(kw)

        param_str = ", ".join(params)
        ret_str = ""
        if node.returns:
            try:
                ret_str = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass

        signature = f"{prefix}{node.name}({param_str}){ret_str}:"
        doc = ast.get_docstring(node)
        decorators = [cls._unparse_decorator(d) for d in node.decorator_list]

        return SymbolInfo(
            name=node.name,
            kind=kind,
            line_number=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            signature=signature,
            docstring=doc.strip().split("\n")[0] if doc else None,
            decorators=decorators,
            parent_class=parent_class,
        )

    @classmethod
    def _extract_class(cls, node: ast.ClassDef) -> list[SymbolInfo]:
        """Extract class definition and its member methods."""
        bases = []
        for b in node.bases:
            try:
                bases.append(ast.unparse(b))
            except Exception:
                pass

        base_str = f"({', '.join(bases)})" if bases else ""
        class_sig = f"class {node.name}{base_str}:"
        doc = ast.get_docstring(node)
        decorators = [cls._unparse_decorator(d) for d in node.decorator_list]

        class_symbol = SymbolInfo(
            name=node.name,
            kind="class",
            line_number=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            signature=class_sig,
            docstring=doc.strip().split("\n")[0] if doc else None,
            decorators=decorators,
        )

        symbols = [class_symbol]

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(cls._extract_function(item, parent_class=node.name))

        return symbols

    @staticmethod
    def _unparse_decorator(d: ast.expr) -> str:
        try:
            return f"@{ast.unparse(d)}"
        except Exception:
            return "@decorator"
