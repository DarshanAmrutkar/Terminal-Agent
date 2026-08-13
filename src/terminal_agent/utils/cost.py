class CostTracker:
    # Approximate pricing per 1K tokens in USD
    PRICING = {
        # Anthropic
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        # OpenAI
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "o3-mini": {"input": 0.0011, "output": 0.0044},
    }

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.requests = []

    def get_model_pricing(self, model: str) -> dict | None:
        if model in self.PRICING:
            return self.PRICING[model]
        for known_model, pricing in self.PRICING.items():
            if model.startswith(known_model):
                return pricing
        return None

    def add_usage(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        pricing = self.get_model_pricing(self.model_name)
        cost = 0.0
        if pricing:
            cost = (input_tokens / 1000.0) * pricing["input"] + (output_tokens / 1000.0) * pricing["output"]
        self.requests.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        })

    def get_session_cost(self) -> float:
        pricing = self.get_model_pricing(self.model_name)
        if not pricing:
            return 0.0
            
        input_cost = (self.total_input_tokens / 1000.0) * pricing["input"]
        output_cost = (self.total_output_tokens / 1000.0) * pricing["output"]
        return input_cost + output_cost

    def format_cost_summary(self) -> str:
        cost = self.get_session_cost()
        return (
            f"Tokens Used: {self.total_input_tokens} In / {self.total_output_tokens} Out\n"
            f"Estimated Session Cost: ${cost:.4f}"
        )

    def format_detailed_summary(self) -> str:
        lines = [self.format_cost_summary(), "Requests:"]
        for i, req in enumerate(self.requests, 1):
            lines.append(f"  {i}. In: {req['input_tokens']}, Out: {req['output_tokens']}, Cost: ${req['cost']:.4f}")
        return "\n".join(lines)
