"""Local in-process restricted sandbox backend."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
import sys
import time

from terminal_agent.sandbox.base import SandboxBackend, SandboxPolicy, SandboxResult
from terminal_agent.tools.base import resolve_safe_path


async def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    """Terminate the process and all its children across platforms."""
    try:
        if sys.platform == "win32":
            await asyncio.to_thread(
                subprocess.run,
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.kill()
    except Exception:
        pass
    try:
        await process.wait()
    except Exception:
        pass


class LocalRestrictedSandbox(SandboxBackend):
    """Executes commands on host system under strict environment and path containment."""

    def __init__(self, policy: SandboxPolicy):
        super().__init__(policy)
        self._current_process: asyncio.subprocess.Process | None = None

    @property
    def name(self) -> str:
        return "local_restricted"

    def is_available(self) -> bool:
        return True

    async def terminate(self) -> None:
        """Terminate any currently executing subprocess and process tree."""
        if self._current_process is not None and self._current_process.returncode is None:
            await _kill_process_tree(self._current_process)
            self._current_process = None

    async def execute(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: int | None = None,
    ) -> SandboxResult:
        start_time = time.perf_counter()
        effective_timeout = timeout or self.policy.timeout_seconds

        # 1. Enforce CWD boundary: must resolve inside working_dir
        target_cwd = cwd if cwd is not None else self.policy.working_dir
        safe_cwd, err = resolve_safe_path(target_cwd, base_dir=self.policy.working_dir)
        if safe_cwd is None:
            return SandboxResult(
                stdout="",
                stderr=f"Security violation: {err}",
                returncode=1,
                duration_seconds=0.0,
                timed_out=False,
            )

        if not safe_cwd.is_dir():
            return SandboxResult(
                stdout="",
                stderr=f"Error: Working directory '{safe_cwd}' does not exist.",
                returncode=1,
                duration_seconds=0.0,
                timed_out=False,
            )

        # 2. Scrub environment variables: strip all API keys and secrets
        clean_env = self.policy.sanitize_environment()

        # 3. Launch subprocess asynchronously with DEVNULL stdin
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(safe_cwd),
                env=clean_env,
            )
            self._current_process = process

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
                    stderr=f"Error: Command timed out after {effective_timeout} seconds.",
                    returncode=124,
                    duration_seconds=round(duration, 2),
                    timed_out=True,
                )
            except (asyncio.CancelledError, KeyboardInterrupt):
                await _kill_process_tree(process)
                raise
            finally:
                self._current_process = None

            duration = time.perf_counter() - start_time
            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            returncode = process.returncode if process.returncode is not None else 0

            # 4. Truncate outputs exceeding max character budget
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
                stderr=f"Execution error: {str(e)}",
                returncode=1,
                duration_seconds=round(duration, 2),
                timed_out=False,
            )
