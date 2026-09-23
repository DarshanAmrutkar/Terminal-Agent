"""Docker container sandbox backend for hardened command isolation."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

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

    @staticmethod
    def _find_docker_executable() -> str | None:
        """Find docker executable from PATH or common install directories."""
        found = shutil.which("docker")
        if found:
            return found

        import os
        candidates = [
            # Windows Docker Desktop per-user and system installs
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "DockerDesktop" / "resources" / "bin" / "docker.exe",
            Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "Docker" / "Docker" / "resources" / "bin" / "docker.exe",
            Path("C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker.exe"),
            # Unix standard paths
            Path("/usr/bin/docker"),
            Path("/usr/local/bin/docker"),
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        return None

    def is_available(self) -> bool:
        """Check if Docker CLI is installed and the Docker daemon is responding."""
        docker_bin = self._find_docker_executable()
        if not docker_bin:
            return False
        try:
            res = subprocess.run(
                [docker_bin, "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
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

        # 2. Build workspace mount: if ephemeral, copy to tempdir so host files are never mutated/deleted
        temp_clone_dir: tempfile.TemporaryDirectory | None = None
        effective_mount_dir = self.policy.working_dir

        if self.policy.ephemeral:
            temp_clone_dir = tempfile.TemporaryDirectory(prefix="agent_ephemeral_")
            effective_mount_dir = Path(temp_clone_dir.name)
            try:
                shutil.copytree(
                    self.policy.working_dir,
                    effective_mount_dir,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", "node_modules"),
                )
            except Exception:
                pass

        # 3. Build Docker CLI command
        docker_bin = self._find_docker_executable() or "docker"
        docker_args = [
            docker_bin, "run", "--rm",
            "-v", f"{effective_mount_dir!s}:/workspace:rw",
            "-w", "/workspace",
            "--memory=1g",
            "--cpus=1.0",
            "--pids-limit=100",
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

        # Prepare host execution environment so docker finds helper tools (docker-credential-desktop, etc.)
        import os
        host_env = dict(os.environ)
        docker_parent = str(Path(docker_bin).parent)
        current_path = host_env.get("PATH", "")
        if docker_parent not in current_path:
            host_env["PATH"] = f"{docker_parent};{current_path}" if os.name == "nt" else f"{docker_parent}:{current_path}"

        try:
            process = await asyncio.create_subprocess_exec(
                *docker_args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=host_env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=effective_timeout
                )
                timed_out = False
            except TimeoutError:
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
                stderr=f"Docker execution failed: {e!s}",
                returncode=1,
                duration_seconds=round(duration, 2),
            )
        finally:
            if temp_clone_dir is not None:
                try:
                    temp_clone_dir.cleanup()
                except Exception:
                    pass
