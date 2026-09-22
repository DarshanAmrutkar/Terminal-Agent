import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from typing import List, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    tomllib = None  # type: ignore


@dataclass
class SyntaxCheckResult:
    """Outcome of pre-commit syntax validation."""
    is_valid: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    line_number: Optional[int] = None
    column: Optional[int] = None
    snippet: Optional[str] = None
    formatted_feedback: str = ""
    warnings: List[str] = field(default_factory=list)


class SyntaxGate:
    """
    Pre-commit syntax and integrity gate inspired by SWE-agent (NeurIPS 2024).
    Prevents syntax degradation spirals by validating code in-memory before it touches disk.
    """

    @classmethod
    def validate(
        cls, 
        path: str | Path, 
        content: str, 
        check_linter: bool = False
    ) -> SyntaxCheckResult:
        """
        Validates content for known file types (.py, .json, .toml).
        Unknown file types pass through as valid.
        """
        file_path = Path(path)
        ext = file_path.suffix.lower()

        if ext == ".py":
            return cls._validate_python(file_path.name, content, check_linter=check_linter)
        elif ext == ".json":
            return cls._validate_json(file_path.name, content)
        elif ext == ".toml":
            return cls._validate_toml(file_path.name, content)

        # Unrecognized file types pass without blocking
        return SyntaxCheckResult(is_valid=True)

    @classmethod
    def _validate_python(
        cls, 
        filename: str, 
        content: str, 
        check_linter: bool = False
    ) -> SyntaxCheckResult:
        try:
            ast.parse(content, filename=filename)
        except SyntaxError as e:
            err_type = type(e).__name__
            lineno = e.lineno or 1
            col = e.offset or 1
            raw_line = e.text.rstrip("\r\n") if e.text else ""
            
            # Build visual caret pointer
            caret_line = ""
            if raw_line and col > 0:
                indent = " " * (col - 1)
                caret_line = f"\n  {raw_line}\n  {indent}^"

            feedback = (
                f"[Syntax Gate: Rejected Edit]\n"
                f"{err_type} at line {lineno}, column {col} in '{filename}':\n"
                f"{caret_line}\n"
                f"Detail: {e.msg}\n\n"
                f"The proposed change was NOT written to disk to protect codebase integrity.\n"
                f"Please correct the syntax/indentation and re-apply."
            )

            return SyntaxCheckResult(
                is_valid=False,
                error_type=err_type,
                error_message=e.msg,
                line_number=lineno,
                column=col,
                snippet=raw_line,
                formatted_feedback=feedback.strip()
            )

        warnings: List[str] = []
        if check_linter:
            warnings = cls._run_ruff_check(filename, content)

        return SyntaxCheckResult(
            is_valid=True,
            warnings=warnings
        )

    @classmethod
    def _validate_json(cls, filename: str, content: str) -> SyntaxCheckResult:
        try:
            json.loads(content)
            return SyntaxCheckResult(is_valid=True)
        except json.JSONDecodeError as e:
            lines = content.splitlines()
            snippet = ""
            if 0 < e.lineno <= len(lines):
                line_str = lines[e.lineno - 1]
                indent = " " * max(0, e.colno - 1)
                snippet = f"\n  {line_str}\n  {indent}^"

            feedback = (
                f"[Syntax Gate: Rejected Edit]\n"
                f"JSONDecodeError in '{filename}' at line {e.lineno}, column {e.colno}:\n"
                f"{snippet}\n"
                f"Detail: {e.msg}\n\n"
                f"The proposed JSON was NOT written to disk because it is malformed.\n"
                f"Please fix formatting (e.g., missing quotes, commas, brackets) and re-apply."
            )
            return SyntaxCheckResult(
                is_valid=False,
                error_type="JSONDecodeError",
                error_message=e.msg,
                line_number=e.lineno,
                column=e.colno,
                snippet=lines[e.lineno - 1] if 0 < e.lineno <= len(lines) else None,
                formatted_feedback=feedback.strip()
            )

    @classmethod
    def _validate_toml(cls, filename: str, content: str) -> SyntaxCheckResult:
        if tomllib is None:
            return SyntaxCheckResult(is_valid=True)

        try:
            tomllib.loads(content)
            return SyntaxCheckResult(is_valid=True)
        except Exception as e:
            feedback = (
                f"[Syntax Gate: Rejected Edit]\n"
                f"TOMLDecodeError in '{filename}': {str(e)}\n\n"
                f"The proposed TOML was NOT written to disk because it is malformed.\n"
                f"Please correct the TOML syntax and re-apply."
            )
            return SyntaxCheckResult(
                is_valid=False,
                error_type=type(e).__name__,
                error_message=str(e),
                formatted_feedback=feedback.strip()
            )

    @classmethod
    def _run_ruff_check(cls, filename: str, content: str) -> List[str]:
        """Runs ruff on stdin buffer if available to gather diagnostics."""
        try:
            res = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "--stdin-filename", filename, "-"],
                input=content,
                capture_output=True,
                text=True,
                timeout=5
            )
            if res.stdout:
                return [line for line in res.stdout.splitlines() if line.strip()]
        except Exception:
            pass
        return []
