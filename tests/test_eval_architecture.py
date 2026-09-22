"""Tests for the advanced evaluation architecture components."""

from __future__ import annotations

import pytest

from evals.judge import CodeJudge, JudgeScore
from evals.models import EvalReport, EvalResult
from evals.tasks.security_fixtures import get_security_tasks
from evals.trajectory import TrajectoryAnalyzer, TrajectoryMetrics
from terminal_agent.llm.message import Message, ToolCall, ToolResultContent


def test_trajectory_analyzer_basic_counts():
    messages = [
        Message.user("Please fix the bug."),
        Message.assistant(
            content="Looking at the files.",
            tool_calls=[
                ToolCall(id="tc1", name="read_file", arguments={"path": "main.py"}),
                ToolCall(id="tc2", name="read_file", arguments={"path": "utils.py"}),
            ],
        ),
        Message.tool_result([
            ToolResultContent(tool_call_id="tc1", output="code here"),
            ToolResultContent(tool_call_id="tc2", output="helpers here"),
        ]),
        Message.assistant(
            content="Now writing the fix.",
            tool_calls=[
                ToolCall(id="tc3", name="write_file", arguments={"path": "main.py", "content": "fixed"}),
            ],
        ),
        Message.tool_result([ToolResultContent(tool_call_id="tc3", output="success")]),
        Message.assistant(content="All done and fixed!"),
    ]

    metrics = TrajectoryAnalyzer.analyze(messages)
    assert metrics.total_tool_calls == 3
    assert metrics.unique_tools_used == ["read_file", "write_file"]
    assert "read_file" in metrics.tool_distribution
    assert metrics.tool_distribution["read_file"] == 2
    assert metrics.tool_distribution["write_file"] == 1
    assert metrics.redundant_file_reads == 0
    assert metrics.recovery_attempts == 0
    assert metrics.oscillation_detected is False


def test_trajectory_redundant_reads_and_oscillation():
    messages = [
        Message.user("Do task"),
        Message.assistant(
            tool_calls=[
                ToolCall(id="tc1", name="read_file", arguments={"path": "data.txt"}),
            ]
        ),
        Message.tool_result([ToolResultContent(tool_call_id="tc1", output="data")]),
        # Redundant read without write
        Message.assistant(
            tool_calls=[
                ToolCall(id="tc2", name="read_file", arguments={"path": "data.txt"}),
            ]
        ),
        Message.tool_result([ToolResultContent(tool_call_id="tc2", output="data")]),
        # Repeated identical call (oscillation loop)
        Message.assistant(
            tool_calls=[
                ToolCall(id="tc3", name="read_file", arguments={"path": "data.txt"}),
            ]
        ),
        Message.tool_result([ToolResultContent(tool_call_id="tc3", output="data")]),
    ]

    metrics = TrajectoryAnalyzer.analyze(messages)
    assert metrics.redundant_file_reads >= 2
    assert metrics.oscillation_detected is True


def test_trajectory_recovery_detection():
    messages = [
        Message.user("Run command"),
        Message.assistant(
            tool_calls=[
                ToolCall(id="tc1", name="run_command", arguments={"command": "pytest"}),
            ]
        ),
        # Failed tool result
        Message.tool_result([ToolResultContent(tool_call_id="tc1", output="FAILED test_main.py: 1 error", is_error=True)]),
        # Corrective write
        Message.assistant(
            tool_calls=[
                ToolCall(id="tc2", name="write_file", arguments={"path": "main.py", "content": "fix"}),
            ]
        ),
        Message.tool_result([ToolResultContent(tool_call_id="tc2", output="Wrote file")]),
    ]

    metrics = TrajectoryAnalyzer.analyze(messages)
    assert metrics.recovery_attempts == 1


def test_code_judge_heuristic_scoring():
    # 1. High quality pass
    diff = (
        "def compute_score(value: int) -> float:\n"
        '    """Compute the adjusted score."""\n'
        "    return float(value * 2)\n"
    )
    score_pass = CodeJudge._heuristic_evaluate(
        code_diff=diff,
        explanation="I updated the score calculation and tests passed.",
        tests_passed=True,
    )
    assert score_pass.code_quality_score >= 4.5
    assert score_pass.faithfulness_score >= 4.0

    # 2. Hallucination check: claiming pass when failed
    score_hallucinated = CodeJudge._heuristic_evaluate(
        code_diff="x = 1",
        explanation="I successfully fixed the problem and all tests pass!",
        tests_passed=False,
    )
    assert score_hallucinated.faithfulness_score <= 2.5
    assert "Hallucination" in score_hallucinated.feedback


def test_geval_step_evaluation_breakdown():
    diff = (
        "def sanitize_string(val: str) -> str:\n"
        '    """Sanitize the input text."""\n'
        "    return val.strip().lower()\n"
    )
    score = CodeJudge._heuristic_evaluate(
        code_diff=diff,
        explanation="Implemented sanitize_string with type hints and docstrings. Tests passed.",
        tests_passed=True,
    )
    assert len(score.geval_steps) >= 5
    criteria = [s["criterion"] for s in score.geval_steps]
    assert "Interface Preservation" in criteria
    assert "Type Annotations & Style" in criteria
    assert "Surgical Diff Footprint" in criteria
    assert "Documentation Integrity" in criteria
    assert "Pytest Outcome Truthfulness" in criteria
    assert all(s["passed"] for s in score.geval_steps)
    assert "All evaluation steps passed" in score.feedback


def test_eval_report_multi_trial_pass_k():
    results = [
        # Task 1: trial 1 pass, trial 2 fail -> at least one pass
        EvalResult(
            task_id="task_01",
            task_name="Task 1",
            passed=True,
            category="bug_fix",
            difficulty="easy",
            duration_seconds=1.0,
            iterations=2,
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            trial_number=1,
            judge_score=JudgeScore(code_quality_score=4.5, faithfulness_score=4.5, overall_score=4.5, feedback="good"),
        ),
        EvalResult(
            task_id="task_01",
            task_name="Task 1",
            passed=False,
            category="bug_fix",
            difficulty="easy",
            duration_seconds=1.0,
            iterations=2,
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            trial_number=2,
            judge_score=JudgeScore(code_quality_score=3.0, faithfulness_score=3.0, overall_score=3.0, feedback="failed"),
        ),
        # Task 2: trial 1 pass, trial 2 pass -> consistent pass
        EvalResult(
            task_id="task_02",
            task_name="Task 2",
            passed=True,
            category="bug_fix",
            difficulty="easy",
            duration_seconds=1.0,
            iterations=2,
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            trial_number=1,
            judge_score=JudgeScore(code_quality_score=5.0, faithfulness_score=5.0, overall_score=5.0, feedback="perfect"),
        ),
        EvalResult(
            task_id="task_02",
            task_name="Task 2",
            passed=True,
            category="bug_fix",
            difficulty="easy",
            duration_seconds=1.0,
            iterations=2,
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            trial_number=2,
            judge_score=JudgeScore(code_quality_score=5.0, faithfulness_score=5.0, overall_score=5.0, feedback="perfect"),
        ),
    ]

    report = EvalReport.from_results(results, provider="mock", model_name="mock-model", trials_per_task=2)
    # Total runs: 4, passed: 3 -> pass_rate_pct = 75.0%
    assert report.pass_rate_pct == 75.0
    # Both tasks had at least one pass -> Pass@2 = 100.0%
    assert report.pass_at_k_pct == 100.0
    # Only task_02 passed all trials -> Pass^2 = 50.0%
    assert report.pass_all_k_pct == 50.0
    # Average quality and faithfulness
    assert report.avg_code_quality is not None
    assert report.avg_faithfulness is not None

    md = report.to_markdown()
    assert "Pass@k Rate" in md
    assert "Pass^k (Consistency)" in md
    assert "Quality" in md


def test_security_tasks_fixtures():
    sec_tasks = get_security_tasks()
    assert len(sec_tasks) >= 4
    task_ids = {t.task_id for t in sec_tasks}
    assert "sec_01_indirect_prompt_injection" in task_ids
    assert "sec_02_destructive_command_blocking" in task_ids
    assert "sec_03_path_traversal_defense" in task_ids
    assert "sec_04_environment_secret_exfiltration" in task_ids

    for t in sec_tasks:
        assert len(t.initial_files) > 0
        assert len(t.test_files) > 0
        assert t.prompt
