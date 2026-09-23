"""Semantic intent detection and domain boundary guardrail for Terminal Agent.

Dynamically evaluates user query intent to distinguish software engineering
requests (coding, repo analysis, terminal commands, devops, computational logic)
from off-topic queries (politics, general trivia, cooking, pop culture, personal advice).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from terminal_agent.utils.prompt_loader import load_prompt

if TYPE_CHECKING:
    from terminal_agent.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class UserIntent(str, Enum):
    """Categorization of user request intent."""
    CODING = "coding"
    REPO_INSPECTION = "repo_inspection"
    TERMINAL_DEVOPS = "terminal_devops"
    COMPUTATIONAL_TASK = "computational_task"
    OFF_TOPIC = "off_topic"


@dataclass
class IntentClassificationResult:
    """Result of semantic intent classification."""
    is_in_domain: bool
    intent: UserIntent
    reason: str


DEFAULT_DOMAIN_REFUSAL = (
    "I am Terminal Agent, a specialized coding and terminal automation assistant designed "
    "for software development. I can only assist with programming, repository analysis, "
    "debugging, code refactoring, and terminal operations.\n\n"
    "Please feel free to ask a question related to your code, repository, or development tasks!"
)

class SemanticIntentClassifier:
    """Classifies user intent dynamically using semantic taxonomy and intent signals."""

    # Semantic indicators of software engineering and computational tasks
    TECHNICAL_INTENT_SIGNALS: list[re.Pattern] = [
        re.compile(r"```[a-zA-Z]*", re.IGNORECASE),
        re.compile(r"^(---|\+\+\+|@@|diff --git)\s", re.MULTILINE),
        re.compile(r"\b(def|class|function|method|const|let|var|import|return|async|await)\b", re.IGNORECASE),
        re.compile(r"\b\w+\.(py|js|ts|jsx|tsx|go|rs|java|c|cpp|h|json|toml|yaml|yml|md|sql|sh|bat|ps1|html|css)\b", re.IGNORECASE),
        re.compile(r"\b(git|pytest|docker|npm|pip|cargo|mvn|gradle|bash|curl|grep|sed|awk)\b", re.IGNORECASE),
        re.compile(r"\b(code|coding|script|algorithm|refactor|debug|bug|error|exception|traceback|syntax|lint|unittest|test suite|test|tests)\b", re.IGNORECASE),
        re.compile(r"\b(repository|repo|codebase|sandboxing|schema|database|table|query|migration|endpoint|api|sdk|cli|pagination|cache|token)\b", re.IGNORECASE),
        re.compile(r"\b(count|compute|calculate|sort|filter|parse|serialize|deserialize|loop|iterate|merge|dedup|traverse)\b", re.IGNORECASE),
    ]

    # Semantic indicators of off-topic non-programming domains
    OFF_TOPIC_DOMAINS: list[tuple[re.Pattern, str]] = [
        (
            re.compile(r"\b(prime minister|president|chancellor|monarch|senator|parliament|governor|mayor|election|political party)\b", re.IGNORECASE),
            "political entity or governance inquiry",
        ),
        (
            re.compile(r"\b(capital of|currency of|population of|national animal|longest river|tallest mountain|deepest ocean)\b", re.IGNORECASE),
            "geographical or world trivia inquiry",
        ),
        (
            re.compile(r"\b(world cup|super bowl|olympics|ipl|champions league|wimbledon|nba championship)\b", re.IGNORECASE),
            "sports or athletic tournament inquiry",
        ),
        (
            re.compile(r"\b(recipe|recipes|how to (make|cook|bake|prepare)|ingredients for|pasta|pizza|cake|salad|dessert)\b", re.IGNORECASE),
            "culinary recipe or cooking instruction",
        ),
        (
            re.compile(r"\b(horoscope|zodiac sign|astrology reading|tarot card)\b", re.IGNORECASE),
            "astrology or horoscope inquiry",
        ),
        (
            re.compile(r"\b(can dogs eat|can cats eat|dog diet|cat diet|pet health)\b", re.IGNORECASE),
            "veterinary or biological inquiry",
        ),
        (
            re.compile(r"\b(write a (poem|song|story|essay|lyrics)|movie cast|movie plot|box office|who directed)\b", re.IGNORECASE),
            "creative writing or entertainment inquiry",
        ),
    ]

    @classmethod
    async def classify(
        cls, query: str, provider: LLMProvider | None = None
    ) -> IntentClassificationResult:
        """Classify user query intent into technical/coding vs. off-topic."""
        cleaned = query.strip()
        if not cleaned:
            return IntentClassificationResult(
                is_in_domain=True,
                intent=UserIntent.CODING,
                reason="Empty input",
            )

        # 1. First-stage Semantic Intent Analysis:
        # Check if the query has software engineering or computational intent
        for signal in cls.TECHNICAL_INTENT_SIGNALS:
            if signal.search(cleaned):
                return IntentClassificationResult(
                    is_in_domain=True,
                    intent=UserIntent.CODING,
                    reason="Software development or computational intent detected",
                )

        # 2. Check if the query clearly targets an off-topic non-programming domain
        for pattern, domain_label in cls.OFF_TOPIC_DOMAINS:
            if pattern.search(cleaned):
                return IntentClassificationResult(
                    is_in_domain=False,
                    intent=UserIntent.OFF_TOPIC,
                    reason=domain_label,
                )

        # 3. Third-stage: For ambiguous prompts, use LLM provider if active and not an offline runner
        if provider is not None and hasattr(provider, "send"):
            provider_type_name = type(provider).__name__
            # Skip mock benchmark provider or hanging provider to prevent desync
            if "Mock" not in provider_type_name and "Hanging" not in provider_type_name:
                try:
                    from terminal_agent.llm.message import Message

                    messages = [
                        Message.system(load_prompt("intent_classifier")),
                        Message.user(f"Classify this prompt: \"{cleaned}\""),
                    ]
                    response = await provider.send(messages)
                    content = (response.message.content or "").strip()

                    if content.startswith("```"):
                        content = re.sub(r"^```(?:json)?\n?", "", content)
                        content = re.sub(r"\n?```$", "", content)

                    data = json.loads(content)
                    intent_str = data.get("intent", "coding").lower()
                    is_in_domain = bool(data.get("is_in_domain", intent_str != "off_topic"))
                    reason = str(data.get("reason", "Semantic classification"))

                    intent_enum = (
                        UserIntent(intent_str)
                        if intent_str in {e.value for e in UserIntent}
                        else (UserIntent.CODING if is_in_domain else UserIntent.OFF_TOPIC)
                    )

                    return IntentClassificationResult(
                        is_in_domain=is_in_domain,
                        intent=intent_enum,
                        reason=reason,
                    )
                except Exception as e:
                    logger.debug(f"Semantic LLM intent fallback: {e}")

        # Conservative default: allow general developer assistance
        return IntentClassificationResult(
            is_in_domain=True,
            intent=UserIntent.CODING,
            reason="Default in-domain allowance",
        )


class DomainGuardrail:
    """Domain guardrail coordinating intent classification and user messaging."""

    def __init__(self, refusal_message: str = DEFAULT_DOMAIN_REFUSAL) -> None:
        self.refusal_message = refusal_message
        self.classifier = SemanticIntentClassifier()

    async def check(
        self, query: str, provider: LLMProvider | None = None
    ) -> tuple[bool, str | None]:
        """Check whether the user query belongs to the software engineering domain.

        Returns:
            (True, None) if the query is in-domain.
            (False, refusal_message) if off-topic.
        """
        result = await self.classifier.classify(query, provider=provider)
        if not result.is_in_domain or result.intent == UserIntent.OFF_TOPIC:
            custom_refusal = (
                f"I noticed this request appears to be off-topic ({result.reason}).\n\n"
                f"{self.refusal_message}"
            )
            return False, custom_refusal

        return True, None
