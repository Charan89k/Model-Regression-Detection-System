import time
from typing import AsyncGenerator

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from mrds.adapters.llm.base import BaseLLMRunner
from mrds.core.config import get_settings
from mrds.domain.models import LatencyMetrics, ModelResponse, PromptConfig, TokenUsage


class AnthropicRunner(BaseLLMRunner):
    """Anthropic API implementation using the official async SDK."""

    def __init__(self):
        settings = get_settings()
        api_key = settings.ANTHROPIC_KEY.get_secret_value() if settings.ANTHROPIC_KEY else ""
        self.client = AsyncAnthropic(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate(self, prompt_config: PromptConfig, user_prompt: str) -> ModelResponse:
        start_time = time.perf_counter()
        
        kwargs = {
            "model": prompt_config.model_name,
            "max_tokens": prompt_config.max_tokens or 4096,  # Anthropic requires max_tokens
            "temperature": prompt_config.temperature,
            "messages": [{"role": "user", "content": user_prompt}],
            "timeout": 30.0,
        }
        
        if prompt_config.system_prompt:
            kwargs["system"] = prompt_config.system_prompt
            
        response = await self.client.messages.create(**kwargs)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Extract text blocks
        text_content = "".join([block.text for block in response.content if hasattr(block, 'text')])

        return ModelResponse(
            raw_text=text_content,
            token_usage=TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            latency=LatencyMetrics(
                time_to_first_token_ms=None,
                total_latency_ms=latency_ms,
            ),
            finish_reason=response.stop_reason,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def stream(self, prompt_config: PromptConfig, user_prompt: str) -> AsyncGenerator[str, None]:
        kwargs = {
            "model": prompt_config.model_name,
            "max_tokens": prompt_config.max_tokens or 4096,
            "temperature": prompt_config.temperature,
            "messages": [{"role": "user", "content": user_prompt}],
            "timeout": 30.0,
        }
        
        if prompt_config.system_prompt:
            kwargs["system"] = prompt_config.system_prompt

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
