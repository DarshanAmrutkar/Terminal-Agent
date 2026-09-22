import pytest
from terminal_agent.llm.message import Message, Role, ToolCall, LLMResponse, StreamEvent
from terminal_agent.llm.anthropic import AnthropicProvider
from terminal_agent.utils.cost import CostTracker


def test_anthropic_format_tools_cache_control():
    provider = AnthropicProvider(model="claude-3-5-sonnet-20241022", api_key="fake-key")
    tools = [
        {"name": "read_file", "description": "Read file", "parameters": {"type": "object"}},
        {"name": "write_file", "description": "Write file", "parameters": {"type": "object"}},
        {"name": "run_command", "description": "Run command", "parameters": {"type": "object"}},
    ]

    formatted = provider.format_tools(tools, enable_cache=True)
    assert len(formatted) == 3
    # First two tools do not have cache_control
    assert "cache_control" not in formatted[0]
    assert "cache_control" not in formatted[1]
    # The final tool carries the ephemeral cache breakpoint
    assert formatted[2]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_prepare_messages_system_prompt_cache_control():
    provider = AnthropicProvider(model="claude-3-5-sonnet-20241022", api_key="fake-key")
    messages = [
        Message.system("You are a helpful coding assistant."),
        Message.user("Hello agent"),
    ]

    system_param, anthropic_messages = provider._prepare_messages(messages)
    assert isinstance(system_param, list)
    assert len(system_param) == 1
    assert system_param[0]["type"] == "text"
    assert "You are a helpful coding assistant." in system_param[0]["text"]
    assert system_param[0]["cache_control"] == {"type": "ephemeral"}

    assert len(anthropic_messages) == 1
    assert anthropic_messages[0]["role"] == "user"
    assert anthropic_messages[0]["content"] == "Hello agent"


def test_llm_response_cache_fields():
    msg = Message.assistant(content="Done")
    resp = LLMResponse(
        message=msg,
        input_tokens=1000,
        output_tokens=200,
        cache_creation_input_tokens=500,
        cache_read_input_tokens=300,
    )
    assert resp.input_tokens == 1000
    assert resp.output_tokens == 200
    assert resp.cache_creation_input_tokens == 500
    assert resp.cache_read_input_tokens == 300


def test_cost_tracker_prompt_caching_discount():
    # claude-3-5-sonnet pricing: input = $0.003 / 1k, output = $0.015 / 1k
    # Standard 1,000 input tokens = $0.003
    # With prompt cache read (90% discount): 1,000 cached tokens = $0.0003
    tracker = CostTracker(model_name="claude-3-5-sonnet-20241022")

    # Turn 1: Cache creation (1,000 tokens created, 1.25x rate = $0.00375)
    tracker.add_usage(
        input_tokens=1000,
        output_tokens=100,
        cache_creation_tokens=1000,
        cache_read_tokens=0
    )
    # output cost: 100 * 0.015 / 1000 = 0.0015
    # total: 0.00375 + 0.0015 = 0.00525
    assert pytest.approx(tracker.get_session_cost(), rel=1e-3) == 0.00525
    assert tracker.total_saved_usd == 0.0

    # Turn 2: Cache read hit (1,000 cached input tokens read at 0.10x rate = $0.00030)
    tracker.add_usage(
        input_tokens=1000,
        output_tokens=100,
        cache_creation_tokens=0,
        cache_read_tokens=1000
    )
    # savings for 1,000 tokens: standard $0.003 - discounted $0.0003 = $0.0027 saved!
    assert pytest.approx(tracker.total_saved_usd, rel=1e-3) == 0.0027
    assert tracker.total_cache_read_tokens == 1000

    summary = tracker.format_cost_summary()
    assert "cached" in summary
    assert "Saved $0.0027 via prompt caching" in summary


def test_cost_tracker_mixed_usage():
    tracker = CostTracker(model_name="claude-3-5-sonnet-20241022")
    # 2000 total input: 500 uncached, 500 created, 1000 read
    tracker.add_usage(
        input_tokens=2000,
        output_tokens=0,
        cache_creation_tokens=500,
        cache_read_tokens=1000
    )
    # uncached 500: 500 * 0.003 / 1000 = 0.0015
    # created 500: 500 * 0.003 * 1.25 / 1000 = 0.001875
    # read 1000: 1000 * 0.003 * 0.10 / 1000 = 0.000300
    # total cost = 0.0015 + 0.001875 + 0.0003 = 0.003675
    assert pytest.approx(tracker.get_session_cost(), rel=1e-3) == 0.003675
    # saved on read: 1000 * 0.003 * 0.90 / 1000 = 0.002700
    assert pytest.approx(tracker.total_saved_usd, rel=1e-3) == 0.002700
