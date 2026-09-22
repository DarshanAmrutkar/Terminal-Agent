"""Session persistence and management store.

Provides durable local storage for sessions under ~/.terminal-agent/sessions/,
enabling session resumption (--resume), historical audits, and inspection.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from terminal_agent.core.session import Session

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_DIR = Path.home() / ".terminal-agent" / "sessions"


class SessionStore:
    """Manages disk persistence and retrieval of terminal agent sessions."""

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, session_id: str) -> Path:
        """Resolve session ID to file path."""
        # Sanitize session_id to prevent path traversal
        clean_id = Path(session_id).name
        if not clean_id.endswith(".json"):
            clean_id = f"{clean_id}.json"
        return self.storage_dir / clean_id

    def save_session(self, session: Session) -> Path:
        """Save a session to disk. Returns the saved file path."""
        target_path = self._get_path(session.session_id)
        session.save(target_path)
        logger.debug(f"Saved session {session.session_id} to {target_path}")
        return target_path

    def load_session(self, session_id: str) -> Session | None:
        """Load a session by session ID. Returns None if not found."""
        target_path = self._get_path(session_id)
        if not target_path.exists():
            # Try searching prefix if user passed partial ID
            matches = list(self.storage_dir.glob(f"{session_id}*.json"))
            if matches:
                target_path = matches[0]
            else:
                return None

        try:
            return Session.load(target_path)
        except Exception as e:
            logger.error(f"Failed to load session {session_id} from {target_path}: {e}")
            return None

    def get_latest_session(self) -> Session | None:
        """Retrieve the most recently updated session, or None if no sessions exist."""
        files = sorted(
            self.storage_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for file_path in files:
            try:
                return Session.load(file_path)
            except Exception as e:
                logger.warning(f"Error reading session file {file_path}: {e}")
                continue
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session by ID. Returns True if deleted."""
        target_path = self._get_path(session_id)
        if target_path.exists():
            target_path.unlink()
            return True
        return False

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List metadata for recent sessions, ordered by most recently modified first."""
        files = sorted(
            self.storage_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        results: list[dict[str, Any]] = []
        for p in files[:limit]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                messages = data.get("messages", [])
                user_prompts = [
                    m.get("content") for m in messages 
                    if m.get("role") == "user" and m.get("content")
                ]
                preview = user_prompts[0][:60] if user_prompts else "(empty)"
                if user_prompts and len(user_prompts[0]) > 60:
                    preview += "..."

                mtime = datetime.fromtimestamp(p.stat().st_mtime)

                results.append({
                    "session_id": data.get("session_id", p.stem),
                    "created_at": data.get("created_at", mtime.isoformat()),
                    "updated_at": mtime.strftime("%Y-%m-%d %H:%M:%S"),
                    "working_directory": data.get("working_directory", "unknown"),
                    "message_count": len(messages),
                    "total_tokens": (
                        data.get("total_input_tokens", 0) + data.get("total_output_tokens", 0)
                    ),
                    "preview": preview,
                    "path": str(p),
                })
            except Exception as e:
                logger.warning(f"Failed to read session metadata from {p}: {e}")
                continue

        return results
