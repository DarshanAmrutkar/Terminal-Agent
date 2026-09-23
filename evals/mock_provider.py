"""Deterministic Mock LLM Provider for offline evaluation and CI testing."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from terminal_agent.llm.base import LLMProvider
from terminal_agent.llm.message import (
    LLMResponse,
    Message,
    StreamEvent,
    ToolCall,
)

# Known automated solutions for benchmark fixtures
MOCK_SOLUTIONS: dict[str, dict[str, str]] = {
    "task_01_off_by_one": {
        "pagination.py": (
            "def paginate(items: list, page: int, page_size: int) -> list:\n"
            "    \"\"\"Return a slice of items for the given 1-indexed page.\"\"\"\n"
            "    if page < 1:\n"
            "        raise ValueError('Page must be >= 1')\n"
            "    start = (page - 1) * page_size\n"
            "    end = start + page_size\n"
            "    return items[start:end]\n"
        )
    },
    "task_02_key_error": {
        "config_loader.py": (
            "from typing import Any\n\n"
            "def get_nested(data: dict, key_path: str, default: Any = None) -> Any:\n"
            "    \"\"\"Extract nested value from dict using dot notation, e.g. 'db.host'.\"\"\"\n"
            "    curr = data\n"
            "    for part in key_path.split('.'):\n"
            "        if not isinstance(curr, dict) or part not in curr:\n"
            "            return default\n"
            "        curr = curr[part]\n"
            "    return curr\n"
        )
    },
    "task_03_list_dedup": {
        "utils.py": (
            "def deduplicate_list(items: list) -> list:\n"
            "    \"\"\"Remove duplicates while preserving original order.\"\"\"\n"
            "    seen = set()\n"
            "    result = []\n"
            "    for item in items:\n"
            "        if item not in seen:\n"
            "            seen.add(item)\n"
            "            result.append(item)\n"
            "    return result\n"
        )
    },
    "task_04_datetime_serializer": {
        "serializer.py": (
            "import json\n"
            "from datetime import datetime\n\n"
            "def serialize_to_json(obj: dict) -> str:\n"
            "    def default_handler(o):\n"
            "        if isinstance(o, datetime):\n"
            "            return o.isoformat()\n"
            "        raise TypeError(f'Object of type {type(o)} is not JSON serializable')\n"
            "    return json.dumps(obj, default=default_handler)\n"
        )
    },
    "task_05_type_conversion": {
        "pricing.py": (
            "def compute_total(items: list[dict]) -> float:\n"
            "    \"\"\"Compute sum of 'price' across all items.\"\"\"\n"
            "    total = 0.0\n"
            "    for item in items:\n"
            "        total += float(item['price'])\n"
            "    return round(total, 2)\n"
        )
    },
    "task_06_multi_file_refactor": {
        "stats_service.py": (
            "from math_utils import calculate_mean\n\n"
            "def get_summary_stats(values: list[float]) -> dict:\n"
            "    return {'mean': calculate_mean(values), 'count': len(values)}\n"
        )
    },
    "task_07_safe_truncate": {
        "strings.py": (
            "def truncate(text: str, max_length: int) -> str:\n"
            "    if len(text) <= max_length:\n"
            "        return text\n"
            "    if max_length < 3:\n"
            "        return text[:max_length]\n"
            "    return text[:max_length - 3] + '...'\n"
        )
    },
    "task_08_cache_ttl": {
        "cache.py": (
            "import time\n\n"
            "class SimpleCache:\n"
            "    def __init__(self):\n"
            "        self._store = {}\n\n"
            "    def set(self, key: str, value: any, ttl_seconds: float) -> None:\n"
            "        expires_at = time.time() + ttl_seconds\n"
            "        self._store[key] = (value, expires_at)\n\n"
            "    def get(self, key: str, default: any = None) -> any:\n"
            "        if key not in self._store:\n"
            "            return default\n"
            "        val, expires_at = self._store[key]\n"
            "        if time.time() >= expires_at:\n"
            "            del self._store[key]\n"
            "            return default\n"
            "        return val\n"
        )
    },
    "task_09_slugify": {
        "slug.py": (
            "import re\n\n"
            "def slugify(text: str) -> str:\n"
            "    text = text.lower()\n"
            "    text = re.sub(r'[^a-z0-9]+', '-', text)\n"
            "    return text.strip('-')\n"
        )
    },
    "sec_01_indirect_prompt_injection": {
        "auth_service.py": (
            "def validate_token(token: str) -> bool:\n"
            "    return token.startswith('valid_')\n"
        )
    },
    "sec_02_destructive_command_blocking": {
        "cleaner.py": (
            "def clean_workspace() -> bool:\n"
            "    return True\n"
        )
    },
    "sec_03_path_traversal_defense": {
        "safe_reader.py": (
            "def get_status() -> str:\n"
            "    return 'SAFE'\n"
        )
    },
    "sec_04_environment_secret_exfiltration": {
        "env_validator.py": (
            "def is_configured() -> bool:\n"
            "    return True\n"
        )
    },
    # Task 10 can remain intentionally un-mocked or simulated fail
    # to realistically reflect a ~80-90% benchmark pass rate in mock test mode!
}


MOCK_QA_RESPONSES: dict[str, str] = {
    "task_qa_01_sandboxing": (
        "# Codebase Architecture & Tool Sandboxing\n\n"
        "The project follows a Hexagonal architecture. Tool execution sandboxing is based on the Strategy Pattern:\n"
        "- `SandboxBackend`: Abstract base interface in `src/terminal_agent/sandbox/base.py`.\n"
        "- `LocalRestrictedSandbox`: In-process host execution with process containment.\n"
        "- `DockerSandbox`: Full container isolation.\n"
        "Security is governed by `SandboxPolicy` which invokes `sanitize_environment` to strip secrets and "
        "`resolve_safe_path` to prevent path traversal outside working directory. Subprocesses run with stdin=DEVNULL."
    ),
    "task_qa_02_permissions": (
        "# Command Permissions & Guardrails\n\n"
        "Command execution permissions are enforced by `PermissionChecker` in `src/terminal_agent/utils/permissions.py`:\n"
        "- `PermissionMode`: Configures safe, auto-test, and `yolo` modes.\n"
        "- `safe_commands`: Whitelist of harmless read-only shell commands.\n"
        "- `blocked_patterns`: Strictly blocked dangerous operations such as `rm -rf /` and format commands."
    ),
    "task_qa_03_prompt_caching": (
        "# Prompt Caching Optimization\n\n"
        "Prompt caching leverages Anthropic ephemeral cache breakpoints:\n"
        "- `cache_control`: Marker `{\"type\": \"ephemeral\"}` placed on system prompt and tool definitions.\n"
        "- `cost_tracker`: Computes cache read discount using `cache_read_input_tokens` at $0.30/MTok."
    ),
    "task_qa_04_session_persistence": (
        "# Session Persistence Lifecycle\n\n"
        "Sessions are managed through `SessionStore` in `src/terminal_agent/core/session_store.py`:\n"
        "- Saved to disk via `save_session` using `to_dict` serialization.\n"
        "- Restored via `load_session` and `from_dict`.\n"
        "- Resumed from CLI using `agent --resume <id>` or `agent sessions`."
    ),
}


class MockEvalProvider(LLMProvider):
    """Mock LLM provider that simulates realistic agent tool interactions for evaluations."""

    def __init__(
        self,
        model: str = "mock-eval-model",
        api_key: str = "mock-key",
        current_task_id: str | None = None,
    ):
        super().__init__(model=model, api_key=api_key)
        self.current_task_id = current_task_id
        self._turn_count = 0

    def set_task(self, task_id: str) -> None:
        self.current_task_id = task_id
        self._turn_count = 0

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def send(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        events = [e async for e in self.stream(messages, tools)]
        text_parts = [e.text for e in events if e.type == "text_delta" and e.text]
        content = "".join(text_parts) or None
        tool_calls = [e.tool_call for e in events if e.type == "tool_call_end" and e.tool_call]
        
        return LLMResponse(
            message=Message.assistant(content=content, tool_calls=tool_calls or None),
            input_tokens=150,
            output_tokens=75,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    async def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        self._turn_count += 1

        # RepoQA / Comprehension tasks
        if self.current_task_id in MOCK_QA_RESPONSES:
            if self._turn_count == 1:
                tc = ToolCall(
                    id=f"call_{self._turn_count}",
                    name="get_repo_map",
                    arguments={"max_tokens": 1500},
                )
                yield StreamEvent(type="text_delta", text="I will inspect the repository map to analyze the architecture.\n")
                yield StreamEvent(type="tool_call_start", tool_call=tc)
                yield StreamEvent(type="tool_call_end", tool_call=tc)
                yield StreamEvent(type="usage", usage={"input_tokens": 150, "output_tokens": 60})
                yield StreamEvent(type="message_end")
                return

            response_text = MOCK_QA_RESPONSES[self.current_task_id]
            yield StreamEvent(type="text_delta", text=response_text)
            yield StreamEvent(type="usage", usage={"input_tokens": 300, "output_tokens": 150})
            yield StreamEvent(type="message_end")
            return

        # Turn 1: If there's a known solution for the task, emit write_file tool call
        if self._turn_count == 1 and self.current_task_id in MOCK_SOLUTIONS:
            solutions = MOCK_SOLUTIONS[self.current_task_id]
            filename, content = next(iter(solutions.items()))

            tc = ToolCall(
                id=f"call_{self._turn_count}",
                name="write_file",
                arguments={"path": filename, "content": content},
            )

            thought = f"I will inspect the problem and fix the issue in `{filename}`.\n"
            yield StreamEvent(type="text_delta", text=thought)
            yield StreamEvent(type="tool_call_start", tool_call=tc)
            yield StreamEvent(type="tool_call_end", tool_call=tc)
            yield StreamEvent(type="usage", usage={"input_tokens": 180, "output_tokens": 85})
            yield StreamEvent(type="message_end")
            return

        # Turn 2 or unsolved task: Return completion message
        if self.current_task_id in MOCK_SOLUTIONS:
            final_text = "I have updated the code to fix the issue and all tests now pass."
        else:
            final_text = "Analysis completed. Could not automatically determine the fix for this task."

        yield StreamEvent(type="text_delta", text=final_text)
        yield StreamEvent(type="usage", usage={"input_tokens": 220, "output_tokens": 40})
        yield StreamEvent(type="message_end")

