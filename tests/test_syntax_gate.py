import os
from pathlib import Path
import tempfile
import pytest

from terminal_agent.utils.syntax_gate import SyntaxGate, SyntaxCheckResult
from terminal_agent.tools.write_file import WriteFileTool
from terminal_agent.tools.search_replace import SearchReplaceTool


def test_syntax_gate_valid_python():
    code = """
def greet(name: str) -> str:
    \"\"\"Greeting function.\"\"\"
    return f"Hello, {name}!"

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
"""
    res = SyntaxGate.validate("test.py", code)
    assert res.is_valid is True
    assert res.error_type is None
    assert res.formatted_feedback == ""


def test_syntax_gate_python_syntax_error():
    broken_code = "def foo(\n    print('unclosed')"
    res = SyntaxGate.validate("broken.py", broken_code)
    assert res.is_valid is False
    assert res.error_type == "SyntaxError"
    assert "was never closed" in res.error_message or "SyntaxError" in res.formatted_feedback
    assert res.line_number is not None
    assert "[Syntax Gate: Rejected Edit]" in res.formatted_feedback


def test_syntax_gate_python_indentation_error():
    broken_code = "def foo():\nprint('bad indent')"
    res = SyntaxGate.validate("indent.py", broken_code)
    assert res.is_valid is False
    assert "IndentationError" in res.error_type
    assert "[Syntax Gate: Rejected Edit]" in res.formatted_feedback
    assert "IndentationError" in res.formatted_feedback


def test_syntax_gate_json_valid_and_invalid():
    valid_json = '{"name": "agent", "version": 2, "tools": ["write", "read"]}'
    res_valid = SyntaxGate.validate("config.json", valid_json)
    assert res_valid.is_valid is True

    invalid_json = '{"name": "agent", "version": 2, missing_quotes: true}'
    res_invalid = SyntaxGate.validate("config.json", invalid_json)
    assert res_invalid.is_valid is False
    assert res_invalid.error_type == "JSONDecodeError"
    assert "[Syntax Gate: Rejected Edit]" in res_invalid.formatted_feedback


def test_syntax_gate_toml_valid_and_invalid():
    valid_toml = """
[project]
name = "terminal-agent"
version = "0.1.0"
dependencies = ["pytest"]
"""
    res_valid = SyntaxGate.validate("pyproject.toml", valid_toml)
    assert res_valid.is_valid is True

    invalid_toml = """
[project
name = "broken
"""
    res_invalid = SyntaxGate.validate("pyproject.toml", invalid_toml)
    assert res_invalid.is_valid is False
    assert "[Syntax Gate: Rejected Edit]" in res_invalid.formatted_feedback


def test_syntax_gate_unrecognized_extensions_pass():
    text = "Here is some plain markdown: # Title\nRandom unclosed (bracket"
    res_md = SyntaxGate.validate("README.md", text)
    assert res_md.is_valid is True

    res_txt = SyntaxGate.validate("notes.txt", text)
    assert res_txt.is_valid is True


@pytest.mark.asyncio
async def test_write_file_rejects_syntax_error_and_preserves_disk():
    with tempfile.TemporaryDirectory() as temp_dir:
        orig_cwd = Path.cwd()
        os.chdir(temp_dir)
        try:
            tool = WriteFileTool()
            broken_code = "def compute(x):\n  return x +\n"
            res = await tool.execute(path="broken_script.py", content=broken_code)

            assert res.is_error is True
            assert "[Syntax Gate: Rejected Edit]" in res.output
            assert not (Path(temp_dir) / "broken_script.py").exists()
        finally:
            os.chdir(orig_cwd)


@pytest.mark.asyncio
async def test_write_file_allows_bypass():
    with tempfile.TemporaryDirectory() as temp_dir:
        orig_cwd = Path.cwd()
        os.chdir(temp_dir)
        try:
            tool = WriteFileTool()
            template_code = "def compute(x):\n  return {{ x }} +\n"
            res = await tool.execute(
                path="template.py", 
                content=template_code, 
                bypass_syntax_check=True
            )

            assert res.is_error is False
            target_file = Path(temp_dir) / "template.py"
            assert target_file.exists()
            assert target_file.read_text(encoding="utf-8") == template_code
        finally:
            os.chdir(orig_cwd)


@pytest.mark.asyncio
async def test_search_replace_rejects_broken_syntax_and_preserves_disk():
    with tempfile.TemporaryDirectory() as temp_dir:
        orig_cwd = Path.cwd()
        os.chdir(temp_dir)
        try:
            target_file = Path(temp_dir) / "app.py"
            initial_code = """def calculate(a, b):
    return a + b

def main():
    print(calculate(2, 3))
"""
            target_file.write_text(initial_code, encoding="utf-8")

            tool = SearchReplaceTool()
            # Replace a valid line with an unclosed statement
            res = await tool.execute(
                path="app.py",
                search="    return a + b",
                replace="    return a +"
            )

            assert res.is_error is True
            assert "Proposed search and replace would introduce a syntax error" in res.output
            assert "[Syntax Gate: Rejected Edit]" in res.output

            # File on disk must remain unchanged!
            assert target_file.read_text(encoding="utf-8") == initial_code
        finally:
            os.chdir(orig_cwd)


@pytest.mark.asyncio
async def test_search_replace_successful_edit():
    with tempfile.TemporaryDirectory() as temp_dir:
        orig_cwd = Path.cwd()
        os.chdir(temp_dir)
        try:
            target_file = Path(temp_dir) / "service.py"
            initial_code = """def run():
    print("starting")
"""
            target_file.write_text(initial_code, encoding="utf-8")

            tool = SearchReplaceTool()
            res = await tool.execute(
                path="service.py",
                search='    print("starting")',
                replace='    print("running service")'
            )

            assert res.is_error is False
            assert 'print("running service")' in target_file.read_text(encoding="utf-8")
        finally:
            os.chdir(orig_cwd)
