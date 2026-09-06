"""Curated benchmark tasks for evaluating Terminal Agent."""

from __future__ import annotations

from evals.models import EvalTask, TaskCategory, TaskDifficulty

BENCHMARK_TASKS: list[EvalTask] = [
    # -------------------------------------------------------------------------
    # Task 1: Off-by-one error in pagination
    # -------------------------------------------------------------------------
    EvalTask(
        task_id="task_01_off_by_one",
        name="Fix pagination offset calculation",
        category=TaskCategory.BUG_FIX,
        difficulty=TaskDifficulty.EASY,
        prompt=(
            "The pagination function in `pagination.py` has an off-by-one bug when calculating "
            "the start index. Page numbers are 1-indexed (page 1 is items 0..page_size-1), but "
            "it currently calculates `start = page * page_size`. Fix `paginate` so tests in `test_pagination.py` pass."
        ),
        initial_files={
            "pagination.py": (
                "def paginate(items: list, page: int, page_size: int) -> list:\n"
                "    \"\"\"Return a slice of items for the given 1-indexed page.\"\"\"\n"
                "    if page < 1:\n"
                "        raise ValueError('Page must be >= 1')\n"
                "    start = page * page_size  # BUG: should be (page - 1) * page_size\n"
                "    end = start + page_size\n"
                "    return items[start:end]\n"
            ),
        },
        test_files={
            "test_pagination.py": (
                "import pytest\n"
                "from pagination import paginate\n\n"
                "def test_paginate_first_page():\n"
                "    items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n"
                "    assert paginate(items, page=1, page_size=3) == [1, 2, 3]\n\n"
                "def test_paginate_second_page():\n"
                "    items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n"
                "    assert paginate(items, page=2, page_size=3) == [4, 5, 6]\n\n"
                "def test_paginate_invalid_page():\n"
                "    with pytest.raises(ValueError):\n"
                "        paginate([1, 2], page=0, page_size=10)\n"
            ),
        },
        expected_files_modified=["pagination.py"],
    ),

    # -------------------------------------------------------------------------
    # Task 2: KeyError in nested config dictionary
    # -------------------------------------------------------------------------
    EvalTask(
        task_id="task_02_key_error",
        name="Handle missing keys in nested config extractor",
        category=TaskCategory.BUG_FIX,
        difficulty=TaskDifficulty.EASY,
        prompt=(
            "In `config_loader.py`, `get_nested` crashes with a KeyError when a key path is missing. "
            "Update it to return `default` instead of raising KeyError. Run `pytest test_config_loader.py` to verify."
        ),
        initial_files={
            "config_loader.py": (
                "from typing import Any\n\n"
                "def get_nested(data: dict, key_path: str, default: Any = None) -> Any:\n"
                "    \"\"\"Extract nested value from dict using dot notation, e.g. 'db.host'.\"\"\"\n"
                "    curr = data\n"
                "    for part in key_path.split('.'):\n"
                "        # BUG: Crashes on missing key or non-dict current value\n"
                "        curr = curr[part]\n"
                "    return curr\n"
            ),
        },
        test_files={
            "test_config_loader.py": (
                "from config_loader import get_nested\n\n"
                "def test_get_nested_success():\n"
                "    cfg = {'db': {'credentials': {'user': 'admin'}}}\n"
                "    assert get_nested(cfg, 'db.credentials.user') == 'admin'\n\n"
                "def test_get_nested_missing_key():\n"
                "    cfg = {'db': {'port': 5432}}\n"
                "    assert get_nested(cfg, 'db.host', default='localhost') == 'localhost'\n\n"
                "def test_get_nested_non_dict_traversal():\n"
                "    cfg = {'db': 'postgres://...'}\n"
                "    assert get_nested(cfg, 'db.host.name', default=None) is None\n"
            ),
        },
        expected_files_modified=["config_loader.py"],
    ),

    # -------------------------------------------------------------------------
    # Task 3: Order-preserving list deduplication
    # -------------------------------------------------------------------------
    EvalTask(
        task_id="task_03_list_dedup",
        name="Implement order-preserving deduplication",
        category=TaskCategory.REFACTOR,
        difficulty=TaskDifficulty.EASY,
        prompt=(
            "`utils.py` contains `deduplicate_list`, which currently casts to a set and scrambles "
            "element order. Modify it to return unique items while preserving their first-seen order."
        ),
        initial_files={
            "utils.py": (
                "def deduplicate_list(items: list) -> list:\n"
                "    \"\"\"Remove duplicates while preserving original order.\"\"\"\n"
                "    return list(set(items))  # BUG: destroys order\n"
            ),
        },
        test_files={
            "test_utils.py": (
                "from utils import deduplicate_list\n\n"
                "def test_order_preserved():\n"
                "    assert deduplicate_list([3, 1, 2, 3, 2, 4, 1]) == [3, 1, 2, 4]\n\n"
                "def test_empty_and_single():\n"
                "    assert deduplicate_list([]) == []\n"
                "    assert deduplicate_list(['a']) == ['a']\n\n"
                "def test_strings_order():\n"
                "    assert deduplicate_list(['banana', 'apple', 'banana', 'cherry']) == ['banana', 'apple', 'cherry']\n"
            ),
        },
        expected_files_modified=["utils.py"],
    ),

    # -------------------------------------------------------------------------
    # Task 4: Datetime ISO serializer
    # -------------------------------------------------------------------------
    EvalTask(
        task_id="task_04_datetime_serializer",
        name="Support datetime serialization in custom json dumper",
        category=TaskCategory.FEATURE,
        difficulty=TaskDifficulty.MEDIUM,
        prompt=(
            "`serializer.py` contains `serialize_to_json`. It currently fails with TypeError "
            "when serializing objects containing `datetime` instances. Update it using a custom JSONEncoder "
            "or default handler so `datetime` objects are serialized into ISO 8601 strings."
        ),
        initial_files={
            "serializer.py": (
                "import json\n\n"
                "def serialize_to_json(obj: dict) -> str:\n"
                "    # BUG: fails on datetime objects\n"
                "    return json.dumps(obj)\n"
            ),
        },
        test_files={
            "test_serializer.py": (
                "import json\n"
                "from datetime import datetime\n"
                "from serializer import serialize_to_json\n\n"
                "def test_serialize_basic():\n"
                "    assert json.loads(serialize_to_json({'a': 1})) == {'a': 1}\n\n"
                "def test_serialize_datetime():\n"
                "    now = datetime(2026, 9, 6, 12, 0, 0)\n"
                "    res = serialize_to_json({'created_at': now})\n"
                "    data = json.loads(res)\n"
                "    assert data['created_at'] == '2026-09-06T12:00:00'\n"
            ),
        },
        expected_files_modified=["serializer.py"],
    ),

    # -------------------------------------------------------------------------
    # Task 5: Type conversion bug in financial / stats calculation
    # -------------------------------------------------------------------------
    EvalTask(
        task_id="task_05_type_conversion",
        name="Fix string concatenation in price aggregator",
        category=TaskCategory.BUG_FIX,
        difficulty=TaskDifficulty.EASY,
        prompt=(
            "`pricing.py` has a bug in `compute_total`. Item prices can be passed as strings "
            "(e.g. '19.99'), causing string concatenation or errors instead of arithmetic addition. "
            "Convert prices properly and round the final sum to 2 decimal places."
        ),
        initial_files={
            "pricing.py": (
                "def compute_total(items: list[dict]) -> float:\n"
                "    \"\"\"Compute sum of 'price' across all items.\"\"\"\n"
                "    total = 0.0\n"
                "    for item in items:\n"
                "        total += item['price']  # BUG: fails if price is str or raises TypeError\n"
                "    return round(total, 2)\n"
            ),
        },
        test_files={
            "test_pricing.py": (
                "from pricing import compute_total\n\n"
                "def test_numeric_prices():\n"
                "    assert compute_total([{'price': 10.5}, {'price': 5.25}]) == 15.75\n\n"
                "def test_string_prices():\n"
                "    assert compute_total([{'price': '10.50'}, {'price': 4.5}]) == 15.0\n\n"
                "def test_empty_list():\n"
                "    assert compute_total([]) == 0.0\n"
            ),
        },
        expected_files_modified=["pricing.py"],
    ),

    # -------------------------------------------------------------------------
    # Task 6: Multi-file refactor and import update
    # -------------------------------------------------------------------------
    EvalTask(
        task_id="task_06_multi_file_refactor",
        name="Synchronize renamed function across modules",
        category=TaskCategory.MULTI_FILE,
        difficulty=TaskDifficulty.MEDIUM,
        prompt=(
            "`math_utils.py` contains `calculate_mean`, but `stats_service.py` is still attempting "
            "to import the obsolete name `calc_average`. Update `stats_service.py` so it imports "
            "and uses `calculate_mean`, resolving the ImportError."
        ),
        initial_files={
            "math_utils.py": (
                "def calculate_mean(numbers: list[float]) -> float:\n"
                "    if not numbers:\n"
                "        return 0.0\n"
                "    return sum(numbers) / len(numbers)\n"
            ),
            "stats_service.py": (
                "# BUG: obsolete import\n"
                "from math_utils import calc_average\n\n"
                "def get_summary_stats(values: list[float]) -> dict:\n"
                "    return {'mean': calc_average(values), 'count': len(values)}\n"
            ),
        },
        test_files={
            "test_stats.py": (
                "from stats_service import get_summary_stats\n\n"
                "def test_summary_stats():\n"
                "    res = get_summary_stats([10.0, 20.0, 30.0])\n"
                "    assert res == {'mean': 20.0, 'count': 3}\n\n"
                "def test_empty_stats():\n"
                "    assert get_summary_stats([]) == {'mean': 0.0, 'count': 0}\n"
            ),
        },
        expected_files_modified=["stats_service.py"],
    ),

    # -------------------------------------------------------------------------
    # Task 7: String truncation with ellipsis
    # -------------------------------------------------------------------------
    EvalTask(
        task_id="task_07_safe_truncate",
        name="Fix safe string truncate with ellipsis",
        category=TaskCategory.BUG_FIX,
        difficulty=TaskDifficulty.EASY,
        prompt=(
            "In `strings.py`, `truncate` should truncate a string to `max_length` characters. "
            "If the string exceeds `max_length`, it should append '...' such that the total length "
            "(including '...') is exactly `max_length`. If `max_length < 3`, it should simply slice to `max_length`."
        ),
        initial_files={
            "strings.py": (
                "def truncate(text: str, max_length: int) -> str:\n"
                "    if len(text) <= max_length:\n"
                "        return text\n"
                "    # BUG: len is max_length + 3\n"
                "    return text[:max_length] + '...'\n"
            ),
        },
        test_files={
            "test_strings.py": (
                "from strings import truncate\n\n"
                "def test_no_truncate_needed():\n"
                "    assert truncate('hello', 10) == 'hello'\n\n"
                "def test_truncate_with_ellipsis():\n"
                "    res = truncate('hello world', 8)\n"
                "    assert res == 'hello...'\n"
                "    assert len(res) == 8\n\n"
                "def test_short_max_length():\n"
                "    assert truncate('hello', 2) == 'he'\n"
            ),
        },
        expected_files_modified=["strings.py"],
    ),

    # -------------------------------------------------------------------------
    # Task 8: Cache TTL expiry boundary condition
    # -------------------------------------------------------------------------
    EvalTask(
        task_id="task_08_cache_ttl",
        name="Fix cache TTL expiry condition",
        category=TaskCategory.BUG_FIX,
        difficulty=TaskDifficulty.MEDIUM,
        prompt=(
            "In `cache.py`, the `SimpleCache` get method has a bug where expired entries are returned "
            "due to an incorrect time comparison logic. Fix `get` and `is_expired` so tests in `test_cache.py` pass."
        ),
        initial_files={
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
                "        # BUG: Inverted logic, returns None only if time.time() < expires_at\n"
                "        if time.time() < expires_at:\n"
                "            del self._store[key]\n"
                "            return default\n"
                "        return val\n"
            ),
        },
        test_files={
            "test_cache.py": (
                "import time\n"
                "from cache import SimpleCache\n\n"
                "def test_cache_hits_before_expiry():\n"
                "    c = SimpleCache()\n"
                "    c.set('foo', 'bar', ttl_seconds=10.0)\n"
                "    assert c.get('foo') == 'bar'\n\n"
                "def test_cache_miss_after_expiry():\n"
                "    c = SimpleCache()\n"
                "    c.set('temp', 123, ttl_seconds=0.01)\n"
                "    time.sleep(0.02)\n"
                "    assert c.get('temp') is None\n"
            ),
        },
        expected_files_modified=["cache.py"],
    ),

    # -------------------------------------------------------------------------
    # Task 9: URL Slugifier
    # -------------------------------------------------------------------------
    EvalTask(
        task_id="task_09_slugify",
        name="Implement safe URL slugify function",
        category=TaskCategory.FEATURE,
        difficulty=TaskDifficulty.MEDIUM,
        prompt=(
            "`slug.py` needs a `slugify(text: str) -> str` function that:\n"
            "1. Lowercases the string.\n"
            "2. Replaces any sequence of non-alphanumeric characters with a single hyphen '-'.\n"
            "3. Strips leading and trailing hyphens.\n"
            "Fix `slugify` so `pytest test_slug.py` passes."
        ),
        initial_files={
            "slug.py": (
                "import re\n\n"
                "def slugify(text: str) -> str:\n"
                "    # BUG: doesn't lower, leaves multiple hyphens, doesn't strip\n"
                "    return re.sub(r'[^a-zA-Z0-9]', '-', text)\n"
            ),
        },
        test_files={
            "test_slug.py": (
                "from slug import slugify\n\n"
                "def test_slugify_basic():\n"
                "    assert slugify('Hello World!') == 'hello-world'\n\n"
                "def test_slugify_multiple_special_chars():\n"
                "    assert slugify('AI & Machine Learning: 2026') == 'ai-machine-learning-2026'\n\n"
                "def test_slugify_leading_trailing():\n"
                "    assert slugify('---Leading and Trailing---') == 'leading-and-trailing'\n"
            ),
        },
        expected_files_modified=["slug.py"],
    ),

    # -------------------------------------------------------------------------
    # Task 10: Deep merge dictionaries
    # -------------------------------------------------------------------------
    EvalTask(
        task_id="task_10_deep_merge",
        name="Implement recursive deep dictionary merge",
        category=TaskCategory.REFACTOR,
        difficulty=TaskDifficulty.MEDIUM,
        prompt=(
            "In `merger.py`, `deep_merge(base: dict, update: dict) -> dict` currently does a shallow "
            "`base.update(update)`, which overwrites nested dictionaries rather than recursively merging them. "
            "Update it so nested dicts are merged, without mutating the original input dictionaries."
        ),
        initial_files={
            "merger.py": (
                "def deep_merge(base: dict, update: dict) -> dict:\n"
                "    # BUG: shallow update mutates base and overwrites nested dictionaries\n"
                "    res = dict(base)\n"
                "    res.update(update)\n"
                "    return res\n"
            ),
        },
        test_files={
            "test_merger.py": (
                "from merger import deep_merge\n\n"
                "def test_shallow_keys():\n"
                "    assert deep_merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}\n\n"
                "def test_nested_merge():\n"
                "    base = {'db': {'host': 'localhost', 'port': 5432}}\n"
                "    override = {'db': {'port': 5433, 'user': 'postgres'}, 'app': 'test'}\n"
                "    merged = deep_merge(base, override)\n"
                "    assert merged == {\n"
                "        'db': {'host': 'localhost', 'port': 5433, 'user': 'postgres'},\n"
                "        'app': 'test'\n"
                "    }\n"
                "    # Ensure base was not mutated\n"
                "    assert base['db']['port'] == 5432\n"
            ),
        },
        expected_files_modified=["merger.py"],
    ),

    # -------------------------------------------------------------------------
    # Task 11: Circular dependency resolution
    # -------------------------------------------------------------------------
    EvalTask(
        task_id="task_11_circular_dependency",
        name="Resolve circular import between model and serializer",
        category=TaskCategory.MULTI_FILE,
        difficulty=TaskDifficulty.HARD,
        prompt=(
            "Modules `user_model.py` and `user_serializer.py` have a circular import "
            "causing an ImportError at module import time. Refactor or use lazy import to break the cycle."
        ),
        initial_files={
            "user_model.py": (
                "from user_serializer import serialize_user\n\n"
                "class User:\n"
                "    def __init__(self, username: str):\n"
                "        self.username = username\n\n"
                "    def to_json(self):\n"
                "        return serialize_user(self)\n"
            ),
            "user_serializer.py": (
                "from user_model import User\n\n"
                "def serialize_user(u: User) -> dict:\n"
                "    return {'user': u.username}\n"
            ),
        },
        test_files={
            "test_circular.py": (
                "def test_import_and_serialize():\n"
                "    from user_model import User\n"
                "    u = User('alice')\n"
                "    assert u.to_json() == {'user': 'alice'}\n"
            ),
        },
        expected_files_modified=["user_model.py"],
    ),
]


def get_task(task_id: str) -> EvalTask | None:
    for t in BENCHMARK_TASKS:
        if t.task_id == task_id:
            return t
    return None


def get_all_tasks() -> list[EvalTask]:
    return list(BENCHMARK_TASKS)
