"""Tests for sensitive file protections across tools and permissions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from terminal_agent.llm.message import ToolCall
from terminal_agent.tools.base import (
    ALLOWED_SENSITIVE_EXCEPTIONS,
    SENSITIVE_FILE_PATTERNS,
    is_sensitive_path,
    resolve_safe_path,
)
from terminal_agent.tools.grep_search import GrepSearchTool
from terminal_agent.tools.read_file import ReadFileTool
from terminal_agent.tools.search_replace import SearchReplaceTool
from terminal_agent.tools.write_file import WriteFileTool
from terminal_agent.utils.permissions import (
    PermissionChecker,
    PermissionOutcome,
    SafetyLevel,
    classify_command,
    references_sensitive_file,
)


class TestSensitivePathDetection:
    """Test is_sensitive_path detection logic."""

    @pytest.mark.parametrize(
        "filename",
        [
            ".env",
            ".env.local",
            ".env.production",
            ".env.development.local",
            "server.pem",
            "cert.key",
            "id_rsa",
            "id_rsa.pub",
            "id_ed25519",
            ".git-credentials",
            ".netrc",
            "credentials.json",
            "service_account_credentials.json",
            "subdir/.env",
            "config/secrets/prod.key",
        ],
    )
    def test_sensitive_files_detected(self, filename: str):
        assert is_sensitive_path(filename) is True

    @pytest.mark.parametrize(
        "filename",
        [
            ".env.example",
            ".env.sample",
            ".env.template",
            ".env.defaults",
            "main.py",
            "config.toml",
            "README.md",
            "setup.cfg",
            "subdir/.env.example",
        ],
    )
    def test_allowed_templates_permitted(self, filename: str):
        assert is_sensitive_path(filename) is False


class TestResolveSafePathSensitiveGate:
    """Test resolve_safe_path blocks sensitive files."""

    def test_blocks_env_file(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            env_file = base / ".env"
            env_file.write_text("SECRET=123", encoding="utf-8")

            resolved, err = resolve_safe_path(".env", base_dir=base)
            assert resolved is None
            assert "Security violation" in err
            assert ".env" in err

    def test_allows_env_example_file(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            example_file = base / ".env.example"
            example_file.write_text("SECRET=YOUR_KEY", encoding="utf-8")

            resolved, err = resolve_safe_path(".env.example", base_dir=base)
            assert resolved is not None
            assert err is None
            assert resolved == example_file

    def test_blocks_nested_secret_key(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            key_file = base / "certs" / "private.key"
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.write_text("PRIVATE_KEY_DATA", encoding="utf-8")

            resolved, err = resolve_safe_path("certs/private.key", base_dir=base)
            assert resolved is None
            assert "Security violation" in err


class TestFileToolsSensitiveProtection:
    """Test read_file, write_file, search_replace cannot touch sensitive files."""

    @pytest.mark.asyncio
    async def test_read_file_blocks_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET_API_KEY=sk-secret-12345", encoding="utf-8")

        tool = ReadFileTool()
        res = await tool.execute(path=".env")
        assert res.is_error
        assert "Security violation" in res.output
        assert "sk-secret-12345" not in res.output

    @pytest.mark.asyncio
    async def test_read_file_allows_env_example(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        example_file = tmp_path / ".env.example"
        example_file.write_text("SECRET_API_KEY=YOUR_KEY_HERE", encoding="utf-8")

        tool = ReadFileTool()
        res = await tool.execute(path=".env.example")
        assert not res.is_error
        assert "YOUR_KEY_HERE" in res.output

    @pytest.mark.asyncio
    async def test_write_file_blocks_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env_file = tmp_path / ".env"

        tool = WriteFileTool()
        res = await tool.execute(path=".env", content="ATTACK_SECRET=hacked")
        assert res.is_error
        assert "Security violation" in res.output
        assert not env_file.exists()

    @pytest.mark.asyncio
    async def test_search_replace_blocks_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env_file = tmp_path / ".env"
        env_file.write_text("OLD_KEY=123", encoding="utf-8")

        tool = SearchReplaceTool()
        res = await tool.execute(path=".env", search="OLD_KEY=123", replace="NEW_KEY=456")
        assert res.is_error
        assert "Security violation" in res.output
        assert env_file.read_text(encoding="utf-8") == "OLD_KEY=123"


class TestGrepSearchSensitiveProtection:
    """Test grep_search skips sensitive files."""

    @pytest.mark.asyncio
    async def test_grep_search_skips_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env_file = tmp_path / ".env"
        env_file.write_text("SUPER_SECRET_TOKEN_XYZ=secret_val", encoding="utf-8")

        code_file = tmp_path / "app.py"
        code_file.write_text("# Regular code file", encoding="utf-8")

        tool = GrepSearchTool()
        res = await tool.execute(pattern="SUPER_SECRET_TOKEN_XYZ", path=".")
        assert "secret_val" not in res.output
        assert "No matches found." in res.output

    @pytest.mark.asyncio
    async def test_grep_search_allows_env_example(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        example_file = tmp_path / ".env.example"
        example_file.write_text("TEMPLATE_VAR=fill_me_in", encoding="utf-8")

        tool = GrepSearchTool()
        res = await tool.execute(pattern="TEMPLATE_VAR", path=".")
        assert "TEMPLATE_VAR" in res.output
        assert ".env.example" in res.output


class TestCommandPermissionsSensitiveProtection:
    """Test classify_command and PermissionChecker guard against sensitive files in shell commands."""

    def test_cat_env_requires_approval_even_if_cat_is_safe(self):
        safe_cmds = ["cat", "type", "ls", "grep"]
        blocked = ["rm -rf /"]

        res = classify_command("cat .env", safe_cmds, blocked, "safe")
        assert res == SafetyLevel.NEEDS_APPROVAL

    def test_type_env_requires_approval_on_windows(self):
        safe_cmds = ["type", "dir"]
        blocked = ["rm -rf /"]

        res = classify_command("type .env", safe_cmds, blocked, "safe")
        assert res == SafetyLevel.NEEDS_APPROVAL

    def test_cat_env_requires_approval_in_yolo_mode(self):
        # Yolo mode must NOT auto-approve sensitive file access
        safe_cmds = ["cat"]
        blocked = ["rm -rf /"]

        res = classify_command("cat .env", safe_cmds, blocked, "yolo")
        assert res == SafetyLevel.NEEDS_APPROVAL

    def test_cat_env_example_is_safe(self):
        safe_cmds = ["cat"]
        blocked = ["rm -rf /"]

        res = classify_command("cat .env.example", safe_cmds, blocked, "safe")
        assert res == SafetyLevel.SAFE

    def test_permission_checker_requires_approval_for_cat_env(self):
        checker = PermissionChecker(
            permission_mode="yolo",
            safe_commands=["cat"],
            blocked_patterns=["rm -rf /"],
        )
        tool = WriteFileTool()  # any tool
        tc = ToolCall(id="c1", name="run_command", arguments={"command": "cat .env"})
        decision = checker.check(tool, tc)
        assert decision.outcome == PermissionOutcome.REQUIRE_APPROVAL
