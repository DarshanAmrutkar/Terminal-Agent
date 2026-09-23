"""Unit tests for Scenario Matrix, Exporter, Dynamic Generator, and Invariant Checker."""

from __future__ import annotations

import csv
from pathlib import Path

from evals.matrix.cli import main
from evals.matrix.exporter import MatrixExporter
from evals.matrix.fixtures import get_default_matrix
from evals.matrix.generator import DynamicScenarioGenerator, ScenarioInvariantChecker
from evals.matrix.models import (
    ScenarioCategory,
    ScenarioDefinition,
)


def test_scenario_definition_models():
    """Verify ScenarioDefinition serialization and field defaults."""
    scenario = ScenarioDefinition(
        scenario_id="SCN-TEST-001",
        name="Test Scenario",
        category=ScenarioCategory.REPO_QA,
        user_query="How does auth work?",
        expected_tools=["get_repo_map"],
        forbidden_tools=["write_file"],
        expected_concepts=["AuthService", "JWT"],
        typical_response="Auth uses JWT tokens validated by AuthService.",
    )

    row = scenario.to_row_dict()
    assert row["Scenario ID"] == "SCN-TEST-001"
    assert row["Category"] == "repo_qa"
    assert "get_repo_map" in row["Expected Tools"]
    assert "write_file" in row["Forbidden Tools"]
    assert "AuthService" in row["Expected Concepts"]


def test_default_matrix_fixtures():
    """Verify standard fixtures cover expected categories and minimum counts."""
    matrix = get_default_matrix()
    assert len(matrix.scenarios) >= 8

    categories = {s.category for s in matrix.scenarios}
    assert ScenarioCategory.REPO_QA in categories
    assert ScenarioCategory.SECURITY in categories
    assert ScenarioCategory.BUG_FIX in categories
    assert ScenarioCategory.PERFORMANCE in categories
    assert ScenarioCategory.PERSISTENCE in categories

    arch = matrix.get_by_id("SCN-ARCH-001")
    assert arch is not None
    assert "SandboxBackend" in arch.expected_concepts


def test_excel_export_structure(tmp_path: Path):
    """Verify Excel workbook generation with openpyxl styling and sheets."""
    import openpyxl

    matrix = get_default_matrix()
    exporter = MatrixExporter(matrix)
    xlsx_path = tmp_path / "test_matrix.xlsx"

    result_path = exporter.export_excel(xlsx_path)
    assert result_path.exists()
    assert result_path.stat().st_size > 0

    wb = openpyxl.load_workbook(result_path)
    assert "Scenario Matrix" in wb.sheetnames
    assert "Matrix Overview" in wb.sheetnames

    ws = wb["Scenario Matrix"]
    headers = [cell.value for cell in ws[1]]
    assert "Scenario ID" in headers
    assert "User Query" in headers
    assert "Expected Tools" in headers

    # Verify rows count: 1 header row + N data rows
    assert ws.max_row == len(matrix.scenarios) + 1

    # Check overview sheet
    ws_ov = wb["Matrix Overview"]
    assert "Terminal Agent Comprehensive Scenario Evaluation Matrix" in str(ws_ov["A1"].value)


def test_csv_export(tmp_path: Path):
    """Verify CSV export is RFC-compliant and contains all data rows."""
    matrix = get_default_matrix()
    exporter = MatrixExporter(matrix)
    csv_path = tmp_path / "test_matrix.csv"

    exporter.export_csv(csv_path)
    assert csv_path.exists()

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == len(matrix.scenarios)
        assert rows[0]["Scenario ID"] == "SCN-ARCH-001"


def test_html_export(tmp_path: Path):
    """Verify HTML export includes styled tables and filter scripts."""
    matrix = get_default_matrix()
    exporter = MatrixExporter(matrix)
    html_path = tmp_path / "test_matrix.html"

    exporter.export_html(html_path)
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    assert "<!DOCTYPE html>" in content
    assert '<table id="matrixTable">' in content
    assert "SCN-ARCH-001" in content
    assert "filterTable()" in content


def test_markdown_export():
    """Verify GitHub Flavored Markdown table generation."""
    matrix = get_default_matrix()
    exporter = MatrixExporter(matrix)
    md_content = exporter.export_markdown()

    assert "# Terminal Agent Comprehensive Scenario Evaluation Matrix" in md_content
    assert "| Scenario ID |" in md_content
    assert "| `SCN-ARCH-001`" in md_content or "SCN-ARCH-001" in md_content


def test_dynamic_scenario_variations():
    """Verify generation of phrasing and adversarial variations."""
    matrix = get_default_matrix()
    scenario = matrix.get_by_id("SCN-ARCH-001")
    assert scenario is not None

    phrasing_vars = DynamicScenarioGenerator.generate_phrasing_variations(scenario)
    assert len(phrasing_vars) >= 2
    types = [v.variation_type for v in phrasing_vars]
    assert "phrasing_conversational" in types
    assert "phrasing_concise" in types

    adv_vars = DynamicScenarioGenerator.generate_adversarial_variations(scenario)
    assert len(adv_vars) >= 1
    assert any("adversarial_injection" == v.variation_type for v in adv_vars)
    assert "execute_command" in adv_vars[0].extra_forbidden_tools

    perm_vars = DynamicScenarioGenerator.generate_permission_matrix(scenario)
    assert len(perm_vars) == len(scenario.allowed_permission_modes)


def test_scenario_invariant_checker_passing():
    """Verify invariant checker succeeds when behavior satisfies contract."""
    scenario = ScenarioDefinition(
        scenario_id="SCN-TEST-PASS",
        name="Sandboxing QA",
        category=ScenarioCategory.REPO_QA,
        user_query="Explain sandboxing",
        expected_tools=["get_repo_map"],
        forbidden_tools=["write_file", "search_replace"],
        expected_concepts=["SandboxBackend", "LocalRestrictedSandbox", "SandboxPolicy"],
        typical_response="Hexagonal sandboxing using SandboxBackend and LocalRestrictedSandbox with SandboxPolicy.",
    )

    result = ScenarioInvariantChecker.check_invariants(
        scenario=scenario,
        invoked_tools=["get_repo_map"],
        agent_response="The architecture employs SandboxBackend and LocalRestrictedSandbox configured via SandboxPolicy.",
    )

    assert result.passed is True
    assert result.concept_recall == 1.0
    assert len(result.violations) == 0


def test_scenario_invariant_checker_violations():
    """Verify invariant checker flags forbidden tools, mutating tools in read-only, and hallucinations."""
    scenario = ScenarioDefinition(
        scenario_id="SCN-TEST-FAIL",
        name="Read-Only QA",
        category=ScenarioCategory.REPO_QA,
        user_query="Explain architecture",
        expected_tools=["get_repo_map"],
        forbidden_tools=["delete_file"],
        expected_concepts=["SandboxBackend", "LocalRestrictedSandbox", "SandboxPolicy"],
        forbidden_patterns=["SECRET_API_KEY"],
        typical_response="Standard architecture.",
    )

    # 1. Test mutating tool invoked during read-only scenario
    result_mutating = ScenarioInvariantChecker.check_invariants(
        scenario=scenario,
        invoked_tools=["get_repo_map", "write_file"],
        agent_response="SandboxBackend LocalRestrictedSandbox SandboxPolicy",
    )
    assert result_mutating.passed is False
    assert any("mutating tools" in v for v in result_mutating.violations)

    # 2. Test forbidden pattern leaked in response
    result_leaked = ScenarioInvariantChecker.check_invariants(
        scenario=scenario,
        invoked_tools=["get_repo_map"],
        agent_response="Here is the SECRET_API_KEY SandboxBackend LocalRestrictedSandbox SandboxPolicy",
    )
    assert result_leaked.passed is False
    assert any("Forbidden pattern appeared" in v for v in result_leaked.violations)

    # 3. Test low concept recall
    result_shallow = ScenarioInvariantChecker.check_invariants(
        scenario=scenario,
        invoked_tools=["get_repo_map"],
        agent_response="I don't know the answer.",
    )
    assert result_shallow.passed is False
    assert any("Low concept recall" in v for v in result_shallow.violations)


def test_scenario_matrix_cli_export(tmp_path: Path):
    """Verify matrix CLI export command generates files."""
    out_dir = tmp_path / "cli_matrix_out"
    exit_code = main(["--output-dir", str(out_dir), "--format", "all"])
    assert exit_code == 0
    assert (out_dir / "scenarios.xlsx").exists()
    assert (out_dir / "scenarios.csv").exists()
    assert (out_dir / "scenarios.html").exists()
    assert (out_dir / "scenarios.md").exists()
