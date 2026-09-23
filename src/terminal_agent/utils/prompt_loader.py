"""Prompt loader utility for Terminal Agent.

All LLM system prompts and task prompts are stored as plain Markdown files
in the top-level ``prompts/`` directory.  This module provides a single
point-of-access so that:

  * Prompt text never lives inline in Python source.
  * Prompts can be edited without touching any ``.py`` files.
  * The resolved path is computed relative to the package root so the
    module works regardless of the current working directory.

Usage::

    from terminal_agent.utils.prompt_loader import load_prompt

    # Simple prompt with no placeholders:
    text = load_prompt("intent_classifier")

    # Prompt with named placeholders filled via .format():
    text = load_prompt(
        "fast_path_repair",
        task_description="Fix the off-by-one error",
        rel_path="src/utils.py",
        file_content="...",
    )
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# The ``prompts/`` directory sits four levels above this file:
#   src/terminal_agent/utils/prompt_loader.py  →  ../../../../prompts/
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@lru_cache(maxsize=None)
def _read_prompt_file(name: str) -> str:
    """Read and cache the raw text of a prompt file.

    Args:
        name: Prompt filename without the ``.md`` extension.

    Returns:
        Raw template string from the file.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: '{path}'. "
            f"Available prompts: {[p.stem for p in _PROMPTS_DIR.glob('*.md')]}"
        )
    return path.read_text(encoding="utf-8")


def load_prompt(name: str, **kwargs: object) -> str:
    """Load a prompt template by name and optionally fill placeholders.

    Args:
        name:    Prompt filename without the ``.md`` extension
                 (e.g. ``"system"``, ``"intent_classifier"``).
        **kwargs: Key-value pairs substituted into ``{placeholder}`` slots
                 inside the template using ``str.format()``.  If no kwargs
                 are given the raw template is returned as-is.

    Returns:
        The prompt string, with any placeholders resolved.

    Raises:
        FileNotFoundError: If ``prompts/<name>.md`` does not exist.
        KeyError:          If a required placeholder is missing from kwargs.
    """
    template = _read_prompt_file(name)
    if kwargs:
        return template.format(**kwargs)
    return template


def list_prompts() -> list[str]:
    """Return the names of all available prompt files (without extension)."""
    return sorted(p.stem for p in _PROMPTS_DIR.glob("*.md"))
