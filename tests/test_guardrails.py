"""Unit tests for Dynamic Semantic Intent Guardrails."""

import json
from unittest.mock import AsyncMock

import pytest

from terminal_agent.core.agent import Agent
from terminal_agent.core.config import AgentConfig
from terminal_agent.guardrails.domain import (
    DomainGuardrail,
    SemanticIntentClassifier,
    UserIntent,
)
from terminal_agent.llm.message import LLMResponse, Message


class MockClassificationProvider:
    """Simulates LLM provider response for semantic intent classification."""

    def __init__(self, intent: str, is_in_domain: bool, reason: str):
        self.intent = intent
        self.is_in_domain = is_in_domain
        self.reason = reason

    async def send(self, messages, tools=None):
        payload = {
            "intent": self.intent,
            "is_in_domain": self.is_in_domain,
            "reason": self.reason,
        }
        return LLMResponse(
            message=Message.assistant(content=json.dumps(payload)),
            input_tokens=15,
            output_tokens=20,
        )


class TestSemanticIntentClassifier:
    """Test suite for the dynamic SemanticIntentClassifier."""

    @pytest.mark.asyncio
    async def test_semantic_classification_off_topic(self):
        """Verify dynamic semantic detection of diverse off-topic queries."""
        off_topic_samples = [
            "who is the prime minister of India?",
            "can dogs eat grapes?",
            "write a poem about winter leaves",
            "how do I make homemade pasta?",
            "what is the capital of Australia?",
        ]

        for query in off_topic_samples:
            result = await SemanticIntentClassifier.classify(query, provider=None)
            assert result.is_in_domain is False
            assert result.intent == UserIntent.OFF_TOPIC
            assert len(result.reason) > 0

    @pytest.mark.asyncio
    async def test_semantic_classification_coding_and_devops(self):
        """Verify dynamic semantic classification of in-domain software engineering requests."""
        in_domain_samples = [
            "implement quicksort in python",
            "why is pytest failing on line 42?",
            "read Dockerfile and explain container entrypoint",
            "refactor compute_total function across modules",
        ]

        for query in in_domain_samples:
            result = await SemanticIntentClassifier.classify(query, provider=None)
            assert result.is_in_domain is True
            assert result.intent in (UserIntent.CODING, UserIntent.REPO_INSPECTION, UserIntent.TERMINAL_DEVOPS, UserIntent.COMPUTATIONAL_TASK)

    @pytest.mark.asyncio
    async def test_fast_code_block_short_circuit(self):
        """Verify that direct code blocks or git diffs immediately bypass without LLM calls."""
        query_with_code = "```python\ndef solve():\n    return 42\n```"
        # Pass None provider — should still pass due to fast code fence detection
        result = await SemanticIntentClassifier.classify(query_with_code, provider=None)
        assert result.is_in_domain is True
        assert result.intent == UserIntent.CODING


class TestDomainGuardrail:
    """Test suite for the DomainGuardrail coordinator."""

    @pytest.mark.asyncio
    async def test_off_topic_refusal_formatting(self):
        """Verify that off-topic queries receive structured, polite declines with reasons."""
        guardrail = DomainGuardrail()

        is_in_domain, refusal = await guardrail.check(
            "who is the prime minister of India?", provider=None
        )
        assert is_in_domain is False
        assert refusal is not None
        assert "off-topic" in refusal
        assert "Terminal Agent" in refusal


class TestAgentGuardrailIntegration:
    """Test suite verifying agent loop interception of off-topic requests."""

    @pytest.mark.asyncio
    async def test_agent_intercepts_off_topic_query(self, tmp_path):
        """Verify agent immediately responds with polite refusal without running tools."""
        config = AgentConfig(enable_domain_guardrail=True)
        agent = Agent(config=config, working_directory=str(tmp_path), listeners=[])

        # Mock classification response
        provider = MockClassificationProvider(
            intent="off_topic",
            is_in_domain=False,
            reason="Inquiry about political figures",
        )
        agent.provider.send = provider.send
        agent.provider.stream = AsyncMock()

        off_topic_query = "who is the prime minister of India?"
        await agent.process_message(off_topic_query)

        # Agent stream (the ReAct loop) must NEVER have been called
        assert agent.provider.stream.call_count == 0

        # Session history should record the conversation cleanly
        messages = agent.session.get_messages()
        assert len(messages) >= 3
        assert messages[-2].content == off_topic_query
        assert "Terminal Agent" in messages[-1].content
        assert "off-topic" in messages[-1].content

    @pytest.mark.asyncio
    async def test_agent_guardrail_can_be_disabled(self, tmp_path):
        """Verify that setting enable_domain_guardrail=False allows query to reach LLM ReAct loop."""
        config = AgentConfig(enable_domain_guardrail=False)
        agent = Agent(config=config, working_directory=str(tmp_path), listeners=[])

        async def mock_stream(*args, **kwargs):
            from terminal_agent.llm.message import StreamEvent
            yield StreamEvent(type="text_delta", text="Simulated LLM response")
            yield StreamEvent(type="message_end")

        agent.provider.stream = mock_stream

        await agent.process_message("who is the prime minister of India?")

        messages = agent.session.get_messages()
        assert messages[-1].content == "Simulated LLM response"
