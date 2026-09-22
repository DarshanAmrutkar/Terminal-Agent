"""Tests for Repository Comprehension & Architectural Reasoning evaluation suite (RepoQA)."""

import pytest

from evals.judge import CodeJudge
from evals.models import TaskCategory
from evals.tasks.fixtures import get_task, get_all_tasks
from evals.runner import EvalRunner


USER_AGENT_SANDBOX_RESPONSE = """
# Terminal Agent — High-Level Architecture Overview

## Overall Architecture
The codebase follows a Hexagonal (Ports & Adapters) architecture with 4 layers:
Presentation, Application, Infrastructure Adapters, Domain Core.

## Tool Execution Sandboxing
The sandbox system is built on the Strategy Pattern with a `SandboxBackend` abstract base class:
- `LocalRestrictedSandbox`: Host process with strict containment.
- `DockerSandbox`: Full container isolation.
- `DisabledSandbox`: Escape hatch.

Key Security Layers:
1. Path Containment: `resolve_safe_path()` in `base.py` prevents directory traversal.
2. Environment Sanitization: `SandboxPolicy.sanitize_environment()` strips ALL secrets (*KEY*, *SECRET*, AWS_*, etc.).
3. Process Isolation: `asyncio.subprocess` + taskkill/kill with stdin=DEVNULL.
4. Output Truncation: 10,000 chars.
"""


def test_repo_qa_scoring_on_agent_response():
    """Verify that a high-quality architectural explanation receives top scores."""
    judge = CodeJudge()
    task = get_task("task_qa_01_sandboxing")
    assert task is not None

    score = judge.evaluate_explanation(
        task_prompt=task.prompt,
        explanation=USER_AGENT_SANDBOX_RESPONSE,
        expected_concepts=task.expected_concepts,
        tools_used=["get_repo_map", "read_file"],
    )

    assert score.overall_score >= 4.0
    assert score.faithfulness_score >= 4.0
    assert score.code_quality_score >= 4.0
    assert len(score.geval_steps) == 4
    # All 4 CoT rubric steps should pass
    assert all(step["passed"] for step in score.geval_steps)


def test_repo_qa_scoring_penalizes_shallow_or_hallucinated_response():
    """Verify that a shallow one-sentence answer fails the comprehension rubric."""
    judge = CodeJudge()
    task = get_task("task_qa_01_sandboxing")
    assert task is not None

    shallow_response = "We use python subprocess to run commands."
    score = judge.evaluate_explanation(
        task_prompt=task.prompt,
        explanation=shallow_response,
        expected_concepts=task.expected_concepts,
        tools_used=[],
    )

    # Conceptual recall will be low
    assert score.overall_score < 3.5
    # Step 1 (conceptual recall) should fail
    step1 = score.geval_steps[0]
    assert step1["criterion"] == "Conceptual Recall & Entity Coverage"
    assert step1["passed"] is False


def test_repo_qa_tasks_registered_in_fixtures():
    """Verify that all curated RepoQA tasks are discoverable in fixtures."""
    all_tasks = get_all_tasks()
    repo_qa_tasks = [t for t in all_tasks if t.category == TaskCategory.REPO_QA]
    
    task_ids = {t.task_id for t in repo_qa_tasks}
    assert "task_qa_01_sandboxing" in task_ids
    assert "task_qa_02_permissions" in task_ids
    assert "task_qa_03_prompt_caching" in task_ids
    assert "task_qa_04_session_persistence" in task_ids

    for t in repo_qa_tasks:
        assert len(t.expected_concepts) > 0
        assert len(t.prompt) > 20


@pytest.mark.asyncio
async def test_mock_eval_runner_on_repo_qa_task():
    """Verify end-to-end evaluation runner execution on a RepoQA task."""
    runner = EvalRunner(provider="mock", model="mock-eval-model", enable_judge=True)
    task = get_task("task_qa_01_sandboxing")
    assert task is not None

    result = await runner.run_task(task)
    assert result.passed is True
    assert result.category == "repo_qa"
    assert result.iterations >= 1
    assert "RepoQA Score:" in result.test_output
    assert result.trajectory is not None
