"""Unit tests for the evaluation framework."""

from __future__ import annotations

import pytest

from evals.models import EvalReport, EvalResult, EvalTask, TaskCategory, TaskDifficulty
from evals.mock_provider import MockEvalProvider
from evals.runner import EvalRunner
from evals.tasks.fixtures import BENCHMARK_TASKS, get_all_tasks, get_task
from terminal_agent.llm.message import Message


def test_benchmark_tasks_loaded():
    tasks = get_all_tasks()
    assert len(tasks) >= 10
    task1 = get_task("task_01_off_by_one")
    assert task1 is not None
    assert task1.name == "Fix pagination offset calculation"
    assert "pagination.py" in task1.initial_files
    assert "test_pagination.py" in task1.test_files


def test_eval_report_metric_calculation():
    results = [
        EvalResult(
            task_id="t1",
            task_name="Task 1",
            passed=True,
            category="bug_fix",
            difficulty="easy",
            duration_seconds=1.5,
            iterations=2,
            input_tokens=100,
            output_tokens=50,
            cost=0.002,
        ),
        EvalResult(
            task_id="t2",
            task_name="Task 2",
            passed=False,
            category="feature",
            difficulty="medium",
            duration_seconds=2.0,
            iterations=3,
            input_tokens=200,
            output_tokens=100,
            cost=0.004,
            error_message="Test failed",
        ),
    ]

    report = EvalReport.from_results(results, provider="test-provider", model_name="test-model")
    assert report.total_tasks == 2
    assert report.passed_tasks == 1
    assert report.failed_tasks == 1
    assert report.pass_rate_pct == 50.0
    assert report.total_duration_seconds == 3.5
    assert report.total_input_tokens == 300
    assert report.total_output_tokens == 150
    assert report.total_cost == 0.006

    md = report.to_markdown()
    assert "50.0%" in md
    assert "Task 1" in md


@pytest.mark.asyncio
async def test_mock_eval_provider():
    provider = MockEvalProvider(current_task_id="task_01_off_by_one")
    messages = [Message.user("Please fix the bug in pagination.py")]
    
    # First turn should emit write_file tool call
    events = [e async for e in provider.stream(messages)]
    tool_starts = [e for e in events if e.type == "tool_call_start"]
    assert len(tool_starts) == 1
    assert tool_starts[0].tool_call.name == "write_file"


@pytest.mark.asyncio
async def test_eval_runner_single_task(tmp_path):
    runner = EvalRunner(provider="mock", output_dir=tmp_path / "results")
    task = get_task("task_01_off_by_one")
    assert task is not None

    result = await runner.run_task(task)
    assert result.passed is True
    assert result.task_id == "task_01_off_by_one"
    assert result.iterations >= 1
    assert result.duration_seconds > 0
