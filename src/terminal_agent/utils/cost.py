"""Cost tracking for LLM token usage.

Pricing is approximate and sourced from public Anthropic/OpenAI pricing pages.
Update PRICING when model prices change. Costs display as informational only
— do not use this for billing.
"""

from __future__ import annotations

from typing import Any


class CostTracker:
    """Tracks cumulative token usage and estimates API cost for a session."""

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
        self.total_cache_creation_tokens: int = 0
        self.total_cache_read_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.total_saved_usd: float = 0.0
        self.usage_history: list[dict[str, Any]] = []

    def add_usage(
        self, 
        input_tokens: int, 
        output_tokens: int, 
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        model_name: str | None = None
    ) -> None:
        """Add token counts from one LLM response and calculate cost with active model pricing,
        taking prompt caching discounts (90% read discount, 1.25x write multiplier) into account."""
        model = model_name or self.model_name
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_creation_tokens += cache_creation_tokens
        self.total_cache_read_tokens += cache_read_tokens

        turn_cost = 0.0
        turn_saved = 0.0
        pricing = self.PRICING.get(model)
        if pricing:
            base_input = pricing["input"]
            base_output = pricing["output"]
            
            # Non-cached regular input
            uncached_input = max(0, input_tokens - cache_creation_tokens - cache_read_tokens)
            uncached_cost = (uncached_input / 1000.0) * base_input
            
            # Cache creation cost: 1.25x base input rate (Anthropic pricing spec)
            cache_creation_cost = (cache_creation_tokens / 1000.0) * (base_input * 1.25)
            
            # Cache read cost: 0.10x base input rate (90% discount!)
            cache_read_cost = (cache_read_tokens / 1000.0) * (base_input * 0.10)
            
            output_cost = (output_tokens / 1000.0) * base_output
            turn_cost = uncached_cost + cache_creation_cost + cache_read_cost + output_cost

            # Calculate savings from cache read tokens vs standard billing
            if cache_read_tokens > 0:
                standard_cost = (cache_read_tokens / 1000.0) * base_input
                turn_saved = max(0.0, standard_cost - cache_read_cost)

        self.total_cost_usd += turn_cost
        self.total_saved_usd += turn_saved
        self.usage_history.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cost": turn_cost,
            "saved": turn_saved,
        })

    def get_session_cost(self) -> float:
        """Estimate total session cost in USD."""
        return self.total_cost_usd

    def is_model_known(self, model_name: str | None = None) -> bool:
        """Return True if this model has pricing data."""
        return (model_name or self.model_name) in self.PRICING

    def format_cost_summary(self) -> str:
        """Return a human-readable cost summary for display."""
        cost = self.get_session_cost()
        pricing_note = (
            "" if self.is_model_known()
            else f" [dim](pricing unavailable for {self.model_name})[/dim]"
        )
        cache_info = ""
        if self.total_cache_read_tokens > 0:
            cache_info = f" ({self.total_cache_read_tokens:,} cached)"
        savings_info = ""
        if self.total_saved_usd > 0.0:
            savings_info = f" [green](Saved ${self.total_saved_usd:.4f} via prompt caching)[/green]"

        return (
            f"Tokens Used: {self.total_input_tokens:,} in{cache_info} / "
            f"{self.total_output_tokens:,} out\n"
            f"Estimated Session Cost: ${cost:.4f}{pricing_note}{savings_info}"
        )
