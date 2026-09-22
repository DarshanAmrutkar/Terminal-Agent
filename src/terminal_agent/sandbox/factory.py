"""Factory for instantiating configured Sandbox backends."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from typing import Optional

from terminal_agent.sandbox.base import SandboxBackend, SandboxPolicy, SandboxResult
from terminal_agent.sandbox.docker import DockerSandbox
from terminal_agent.sandbox.local import LocalRestrictedSandbox, _kill_process_tree


class DisabledSandbox(SandboxBackend):
    """Unrestricted host execution (sandbox disabled)."""

    @property
    def name(self) -> str:
        return "disabled"

    def is_available(self) -> bool:
        return True

    async def execute(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: int | None = None,
    ) -> SandboxResult:
        start_time = time.perf_counter()
        effective_timeout = timeout or self.policy.timeout_seconds
        target_cwd = cwd or self.policy.working_dir

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(target_cwd),
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=effective_timeout
                )
                timed_out = False
            except asyncio.TimeoutError:
                await _kill_process_tree(process)
                duration = time.perf_counter() - start_time
                return SandboxResult(
                    stdout="",
                    stderr=f"Error: Command timed out after {effective_timeout}s.",
                    returncode=124,
                    duration_seconds=round(duration, 2),
                    timed_out=True,
                )

            duration = time.perf_counter() - start_time
            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            returncode = process.returncode if process.returncode is not None else 0

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                duration_seconds=round(duration, 2),
                timed_out=timed_out,
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=str(e),
                returncode=1,
                duration_seconds=round(time.perf_counter() - start_time, 2),
            )


def create_sandbox(
    mode: str = "local",
    working_dir: Path | str = ".",
    policy: Optional[SandboxPolicy] = None,
    docker_image: str = "python:3.12-slim",
) -> SandboxBackend:
    """Instantiate and return the appropriate sandbox backend based on configuration."""
    work_path = Path(working_dir).resolve()
    active_policy = policy or SandboxPolicy(working_dir=work_path)

    mode_lower = mode.lower().strip()

    if mode_lower == "docker":
        docker_sb = DockerSandbox(policy=active_policy, image_name=docker_image)
        if docker_sb.is_available():
            return docker_sb
        # Fallback to local restricted sandbox if docker is unavailable
        return LocalRestrictedSandbox(policy=active_policy)

    elif mode_lower == "disabled":
        return DisabledSandbox(policy=active_policy)

    else:
        # Default to local restricted sandbox
        return LocalRestrictedSandbox(policy=active_policy)
