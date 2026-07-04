from abc import ABC, abstractmethod
from typing import AsyncGenerator

from mrds.domain.models import ModelResponse, PromptConfig


class BaseLLMRunner(ABC):
    """Abstract Base Class for all LLM Provider implementations."""

    @abstractmethod
    async def generate(self, prompt_config: PromptConfig, user_prompt: str) -> ModelResponse:
        """Generates a complete response synchronously (awaiting the full response)."""
        pass

    @abstractmethod
    async def stream(
        self, prompt_config: PromptConfig, user_prompt: str
    ) -> AsyncGenerator[str, None]:
        """Streams the response chunks back to the caller."""
        pass


def calculate_cost(
    provider: str, model_name: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """
    Calculates estimated cost based on token usage.
    Rates are per 1,000 tokens (mocked for simplicity, ideally loaded from config or DB).
    """
    costs = {
        "openai": {
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        },
        "anthropic": {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        },
        "gemini": {
            "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},
        },
    }

    provider_rates = costs.get(provider.lower(), {})
    model_rates = provider_rates.get(model_name.lower(), {"input": 0.0, "output": 0.0})

    input_cost = (prompt_tokens / 1000.0) * model_rates["input"]
    output_cost = (completion_tokens / 1000.0) * model_rates["output"]

    return input_cost + output_cost
