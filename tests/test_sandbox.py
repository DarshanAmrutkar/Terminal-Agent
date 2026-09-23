"""Unit tests for the Execution Sandboxing subsystem."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from terminal_agent.sandbox import (
    DisabledSandbox,
    DockerSandbox,
    LocalRestrictedSandbox,
    SandboxPolicy,
    create_sandbox,
)
from terminal_agent.tools.run_command import RunCommandTool


def test_sandbox_policy_environment_scrubbing():
    with tempfile.TemporaryDirectory() as temp_dir:
        policy = SandboxPolicy(working_dir=Path(temp_dir))

        dirty_env = {
            "PATH": "C:\\Windows\\system32;/usr/bin",
            "TEMP": "/tmp",
            "USER": "developer",
            "PYTHONPATH": "/workspace/src",
            # Secrets to scrub
            "ANTHROPIC_API_KEY": "sk-ant-1234567890",
            "OPENAI_API_KEY": "sk-openai-secret",
            "NVIDIA_API_KEY": "nvapi-secret",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "DATABASE_PASSWORD": "supersecretpassword",
            "GITHUB_TOKEN": "ghp_xxxxxxxxxxxx",
            "PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----",
            "RANDOM_UNLISTED_VAR": "should_be_dropped",
        }

        clean_env = policy.sanitize_environment(base_env=dirty_env)

        # 1. Verify secrets are stripped
        for secret_key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "NVIDIA_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "DATABASE_PASSWORD",
            "GITHUB_TOKEN",
            "PRIVATE_KEY",
            "RANDOM_UNLISTED_VAR",
        ]:
            assert secret_key not in clean_env

        # 2. Verify safe variables are retained
        assert clean_env.get("PATH") == dirty_env["PATH"]
        assert clean_env.get("TEMP") == dirty_env["TEMP"]
        assert clean_env.get("USER") == dirty_env["USER"]
        assert clean_env.get("PYTHONPATH") == dirty_env["PYTHONPATH"]


@pytest.mark.asyncio
async def test_local_restricted_sandbox_execution():
    with tempfile.TemporaryDirectory() as temp_dir:
        work_path = Path(temp_dir)
        policy = SandboxPolicy(working_dir=work_path)
        sandbox = LocalRestrictedSandbox(policy=policy)

        # Simple Python execution
        cmd = f'"{sys.executable}" -c "import os; print(\'SANDBOX_OK\')"'
        result = await sandbox.execute(command=cmd)

        assert not result.is_error
        assert "SANDBOX_OK" in result.stdout
        assert result.returncode == 0
        assert not result.timed_out


@pytest.mark.asyncio
async def test_local_restricted_sandbox_secret_isolation():
    """Verify that child processes inside the sandbox cannot read sensitive host env vars."""
    with tempfile.TemporaryDirectory() as temp_dir:
        work_path = Path(temp_dir)
        policy = SandboxPolicy(working_dir=work_path)
        sandbox = LocalRestrictedSandbox(policy=policy)

        # Temporarily inject a secret into host os.environ
        os.environ["SUPER_SECRET_TOKEN"] = "injected_token_xyz"
        try:
            cmd = f'"{sys.executable}" -c "import os; print(\'TOKEN=\' + os.environ.get(\'SUPER_SECRET_TOKEN\', \'NOT_FOUND\'))"'
            result = await sandbox.execute(command=cmd)
            assert not result.is_error
            assert "TOKEN=NOT_FOUND" in result.stdout
        finally:
            os.environ.pop("SUPER_SECRET_TOKEN", None)


@pytest.mark.asyncio
async def test_local_restricted_sandbox_cwd_jail():
    with tempfile.TemporaryDirectory() as temp_dir:
        work_path = Path(temp_dir)
        policy = SandboxPolicy(working_dir=work_path)
        sandbox = LocalRestrictedSandbox(policy=policy)

        # Attempt to run command outside workspace
        outside_path = work_path.parent
        result = await sandbox.execute(
            command="echo test",
            cwd=str(outside_path),
        )

        assert result.is_error
        assert "Security violation" in result.stderr
        assert result.returncode != 0


@pytest.mark.asyncio
async def test_local_restricted_sandbox_timeout():
    with tempfile.TemporaryDirectory() as temp_dir:
        work_path = Path(temp_dir)
        policy = SandboxPolicy(working_dir=work_path, timeout_seconds=1)
        sandbox = LocalRestrictedSandbox(policy=policy)

        # Run command that exceeds 1s timeout
        cmd = f'"{sys.executable}" -c "import time; time.sleep(5)"'
        result = await sandbox.execute(command=cmd, timeout=1)

        assert result.timed_out
        assert result.is_error
        assert "timed out" in result.stderr.lower()


@pytest.mark.asyncio
async def test_run_command_tool_with_sandbox():
    with tempfile.TemporaryDirectory() as temp_dir:
        work_path = Path(temp_dir)
        sandbox = create_sandbox(mode="local", working_dir=work_path)
        tool = RunCommandTool(sandbox=sandbox, working_dir=str(work_path))

        res = await tool.execute(command=f'"{sys.executable}" -c "print(\'TOOL_SANDBOX_SUCCESS\')"')
        assert not res.is_error
        assert "TOOL_SANDBOX_SUCCESS" in res.output
        assert res.metadata["sandbox"] == "local_restricted"
        assert res.metadata["returncode"] == 0


def test_create_sandbox_factory():
    with tempfile.TemporaryDirectory() as temp_dir:
        work_path = Path(temp_dir)

        # 1. Local mode
        sb_local = create_sandbox(mode="local", working_dir=work_path)
        assert isinstance(sb_local, LocalRestrictedSandbox)
        assert sb_local.name == "local_restricted"

        # 2. Disabled mode
        sb_disabled = create_sandbox(mode="disabled", working_dir=work_path)
        assert isinstance(sb_disabled, DisabledSandbox)
        assert sb_disabled.name == "disabled"

        # 3. Docker mode (falls back to local if docker is not running on host)
        sb_docker = create_sandbox(mode="docker", working_dir=work_path)
        assert isinstance(sb_docker, (DockerSandbox, LocalRestrictedSandbox))
