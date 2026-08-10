class CostTracker:
    # Approximate pricing per 1K tokens in USD
    PRICING = {
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "gpt-4o": {"input": 0.005, "output": 0.015},
    }

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def add_usage(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def get_session_cost(self) -> float:
        pricing = self.PRICING.get(self.model_name)
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
