from __future__ import annotations
import subprocess
import os

class GitRepo:
    """
    A wrapper around git subprocess commands.
    """
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def _run(self, cmd: list[str]) -> str:
        """Helper to run a git command and return its output."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0:
                return f"Error ({result.returncode}): {result.stderr.strip()}"
            return result.stdout.strip()
        except Exception as e:
            return f"Error executing {' '.join(cmd)}: {str(e)}"

    def is_git_repo(self) -> bool:
        """Check if the path is a git repository."""
        if not os.path.exists(self.repo_path):
            return False
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False

    def status(self) -> str:
        """Run git status --short."""
        return self._run(["git", "status", "--short"])

    def diff(self, staged: bool = False) -> str:
        """Run git diff or git diff --staged."""
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        return self._run(cmd)

    def log(self, n: int = 10) -> str:
        """Run git log --oneline -n N."""
        return self._run(["git", "log", "--oneline", "-n", str(n)])

    def current_branch(self) -> str:
        """Run git rev-parse --abbrev-ref HEAD."""
        return self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    def commit(self, message: str, add_all: bool = True) -> str:
        """Run git add -A and git commit -m message."""
        if add_all:
            add_res = self._run(["git", "add", "-A"])
            if add_res.startswith("Error"):
                return add_res
        return self._run(["git", "commit", "-m", message])

    def blame(self, path: str) -> str:
        """Run git blame path."""
        return self._run(["git", "blame", path])
