"""Cost tracking for LLM token usage.

Pricing is approximate and sourced from public Anthropic/OpenAI pricing pages.
Update PRICING when model prices change. Costs display as informational only
\u2014 do not use this for billing.
"""

from __future__ import annotations


class CostTracker:
    """Tracks cumulative token usage and estimates API cost for a session.

    Design note: Why store pricing here?
    We could fetch pricing from an external source (API, config file), but
    that adds a network dependency and configuration complexity for a feature
    that's purely informational. Hardcoded pricing with a clear update path
    is simpler and more reliable. The cost display is approximate by nature
    \u2014 exact billing comes from the API provider's dashboard.
    """

    # Pricing per 1,000 tokens in USD.
    # Sources:
    #   Anthropic: https://www.anthropic.com/pricing
    #   OpenAI:    https://openai.com/pricing
    PRICING: dict[str, dict[str, float]] = {
        # Claude 4 family
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-opus-4-20250514":   {"input": 0.015, "output": 0.075},
        # Claude 3.7 family
        "claude-3-7-sonnet-20250219": {"input": 0.003, "output": 0.015},
        # Claude 3.5 family
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku-20241022":  {"input": 0.0008, "output": 0.004},
        # Claude 3 family
        "claude-3-opus-20240229":     {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229":   {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307":    {"input": 0.00025, "output": 0.00125},
        # OpenAI
        "gpt-4o":                     {"input": 0.005, "output": 0.015},
        "gpt-4o-mini":                {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo":                {"input": 0.010, "output": 0.030},
        "o1":                         {"input": 0.015, "output": 0.060},
        "o1-mini":                    {"input": 0.003, "output": 0.012},
        # OpenRouter popular models
        "anthropic/claude-3.5-sonnet":          {"input": 0.003, "output": 0.015},
        "anthropic/claude-3.7-sonnet":          {"input": 0.003, "output": 0.015},
        "openai/gpt-4o":                        {"input": 0.005, "output": 0.015},
        "openai/gpt-4o-mini":                   {"input": 0.00015, "output": 0.0006},
        "deepseek/deepseek-chat":               {"input": 0.00014, "output": 0.00028},
        "deepseek/deepseek-r1":                 {"input": 0.00055, "output": 0.00219},
        "meta-llama/llama-3.3-70b-instruct":    {"input": 0.00012, "output": 0.0003},
    }

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Add token counts from one LLM response to the running total."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def get_session_cost(self) -> float:
        """Estimate total session cost in USD.

        Returns 0.0 if the model is not in the pricing table rather than
        raising an error \u2014 cost tracking is informational, not critical path.
        """
        pricing = self.PRICING.get(self.model_name)
        if not pricing:
            return 0.0

        input_cost = (self.total_input_tokens / 1000.0) * pricing["input"]
        output_cost = (self.total_output_tokens / 1000.0) * pricing["output"]
        return input_cost + output_cost

    def is_model_known(self) -> bool:
        """Return True if this model has pricing data."""
        return self.model_name in self.PRICING

    def format_cost_summary(self) -> str:
        """Return a human-readable cost summary for display."""
        cost = self.get_session_cost()
        pricing_note = (
            "" if self.is_model_known()
            else f" [dim](pricing unavailable for {self.model_name})[/dim]"
        )
        return (
            f"Tokens Used: {self.total_input_tokens:,} in / "
            f"{self.total_output_tokens:,} out\n"
            f"Estimated Session Cost: ${cost:.4f}{pricing_note}"
        )
