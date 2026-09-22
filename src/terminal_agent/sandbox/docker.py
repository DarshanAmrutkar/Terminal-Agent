"""Docker container sandbox backend for hardened command isolation."""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import subprocess
import time

from terminal_agent.sandbox.base import SandboxBackend, SandboxPolicy, SandboxResult
from terminal_agent.tools.base import resolve_safe_path


class DockerSandbox(SandboxBackend):
    """Executes commands inside an isolated Docker container with mounted workspace."""

    def __init__(
        self,
        policy: SandboxPolicy,
        image_name: str = "python:3.12-slim",
    ):
        super().__init__(policy)
        self.image_name = image_name

    @property
    def name(self) -> str:
        return "docker"

    def is_available(self) -> bool:
        """Check if Docker CLI is installed and the Docker daemon is responding."""
        docker_bin = shutil.which("docker")
        if not docker_bin:
            return False
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            )
            return res.returncode == 0
        except Exception:
            return False

    async def execute(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: int | None = None,
    ) -> SandboxResult:
        if not self.is_available():
            return SandboxResult(
                stdout="",
                stderr="Error: Docker is not available or Docker daemon is not running.",
                returncode=1,
                duration_seconds=0.0,
            )

        start_time = time.perf_counter()
        effective_timeout = timeout or self.policy.timeout_seconds

        # 1. Enforce CWD boundary
        target_cwd = cwd if cwd is not None else self.policy.working_dir
        safe_cwd, err = resolve_safe_path(target_cwd, base_dir=self.policy.working_dir)
        if safe_cwd is None:
            return SandboxResult(
                stdout="",
                stderr=f"Security violation: {err}",
                returncode=1,
                duration_seconds=0.0,
            )

        # 2. Build Docker CLI command
        docker_args = [
            "docker", "run", "--rm",
            "-v", f"{str(self.policy.working_dir)}:/workspace:rw",
            "-w", "/workspace",
            "--memory=1g",
            "--cpus=1.0",
        ]

        if not self.policy.allow_network:
            docker_args.extend(["--network", "none"])

        # Inject sanitized environment variables
        clean_env = self.policy.sanitize_environment()
        for k, v in clean_env.items():
            if k not in ("PATH", "SYSTEMROOT", "COMSPEC", "TEMP", "TMP"):
                docker_args.extend(["-e", f"{k}={v}"])

        docker_args.extend([
            self.image_name,
            "sh", "-c", command,
        ])

        try:
            process = await asyncio.create_subprocess_exec(
                *docker_args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=effective_timeout
                )
                timed_out = False
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                duration = time.perf_counter() - start_time
                return SandboxResult(
                    stdout="",
                    stderr=f"Error: Docker command timed out after {effective_timeout}s.",
                    returncode=124,
                    duration_seconds=round(duration, 2),
                    timed_out=True,
                )

            duration = time.perf_counter() - start_time
            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            returncode = process.returncode if process.returncode is not None else 0

            # Output truncation
            truncated = False
            max_chars = self.policy.max_output_chars
            if len(stdout) > max_chars:
                stdout = stdout[:max_chars] + f"\n... [stdout truncated from {len(stdout)} chars]"
                truncated = True
            if len(stderr) > max_chars:
                stderr = stderr[:max_chars] + f"\n... [stderr truncated from {len(stderr)} chars]"
                truncated = True

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                duration_seconds=round(duration, 2),
                timed_out=timed_out,
                truncated=truncated,
            )

        except Exception as e:
            duration = time.perf_counter() - start_time
            return SandboxResult(
                stdout="",
                stderr=f"Docker execution failed: {str(e)}",
                returncode=1,
                duration_seconds=round(duration, 2),
            )
