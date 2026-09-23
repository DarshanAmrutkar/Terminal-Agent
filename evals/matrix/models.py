"""Data models for the Agent Scenario Matrix and dynamic test cases."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ScenarioCategory(str, Enum):
    """Categorization of agent evaluation scenarios."""
    REPO_QA = "repo_qa"
    SECURITY = "security"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    PERFORMANCE = "performance"
    PERSISTENCE = "persistence"
    CLI = "cli"


class ScenarioDefinition(BaseModel):
    """Strongly typed scenario definition serving as matrix-as-code."""

    scenario_id: str = Field(description="Unique scenario code, e.g. SCN-ARCH-001")
    name: str = Field(description="Short human-readable scenario title")
    category: ScenarioCategory = Field(description="Scenario category")
    user_query: str = Field(description="The user input prompt given to the agent")
    preconditions: str = Field(
        default="Clean repository, default settings",
        description="Environment state and prerequisites before execution",
    )
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Tool names that the agent is expected to invoke",
    )
    forbidden_tools: list[str] = Field(
        default_factory=list,
        description="Tool names that the agent must NEVER invoke",
    )
    expected_concepts: list[str] = Field(
        default_factory=list,
        description="Key entities, class names, or technical terms required in the output",
    )
    forbidden_patterns: list[str] = Field(
        default_factory=list,
        description="Hallucinated or forbidden terms that should not appear in the response",
    )
    allowed_permission_modes: list[str] = Field(
        default_factory=lambda: ["normal", "auto_test", "yolo"],
        description="Permission modes under which this scenario is valid",
    )
    typical_response: str = Field(
        description="Golden reference output or structural summary of expected response",
    )
    pass_criteria: str = Field(
        default="concept_recall >= 0.6 and no_forbidden_tools",
        description="Machine and human-verifiable pass rule description",
    )
    max_turns: int = Field(default=10, description="Max permitted agent iterations")
    tags: list[str] = Field(default_factory=list, description="Categorical tags (e.g. smoke, core)")

    def to_row_dict(self) -> dict[str, Any]:
        """Convert scenario into a flat dictionary suitable for tabular representation."""
        return {
            "Scenario ID": self.scenario_id,
            "Name": self.name,
            "Category": self.category.value,
            "User Query": self.user_query,
            "Preconditions": self.preconditions,
            "Expected Tools": ", ".join(self.expected_tools) if self.expected_tools else "(none)",
            "Forbidden Tools": ", ".join(self.forbidden_tools) if self.forbidden_tools else "(none)",
            "Expected Concepts": ", ".join(self.expected_concepts) if self.expected_concepts else "(none)",
            "Forbidden Patterns": ", ".join(self.forbidden_patterns) if self.forbidden_patterns else "(none)",
            "Allowed Permissions": ", ".join(self.allowed_permission_modes),
            "Typical Response": self.typical_response,
            "Pass Criteria": self.pass_criteria,
            "Max Turns": self.max_turns,
            "Tags": ", ".join(self.tags) if self.tags else "(none)",
        }


class DynamicVariation(BaseModel):
    """A parameterized variation derived dynamically from a base scenario."""

    base_scenario_id: str
    variation_id: str
    variation_type: str = Field(
        description="Type of mutation: 'phrasing_conversational', 'adversarial_injection', 'permission_stress', etc.",
    )
    perturbed_query: str
    permission_mode: str = "normal"
    extra_forbidden_tools: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ScenarioMatrix(BaseModel):
    """Collection of scenarios representing the complete test matrix."""

    title: str = "Terminal Agent Evaluation Scenario Matrix"
    version: str = "1.0.0"
    scenarios: list[ScenarioDefinition] = Field(default_factory=list)

    def get_by_id(self, scenario_id: str) -> ScenarioDefinition | None:
        """Find a scenario by its ID."""
        for s in self.scenarios:
            if s.scenario_id == scenario_id:
                return s
        return None

    def filter_by_category(self, category: ScenarioCategory) -> list[ScenarioDefinition]:
        """Filter scenarios by category."""
        return [s for s in self.scenarios if s.category == category]

    def filter_by_tag(self, tag: str) -> list[ScenarioDefinition]:
        """Filter scenarios by tag."""
        return [s for s in self.scenarios if tag in s.tags]
