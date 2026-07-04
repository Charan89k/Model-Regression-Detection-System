import time
from typing import AsyncGenerator

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from mrds.adapters.llm.base import BaseLLMRunner
from mrds.core.config import get_settings
from mrds.domain.models import LatencyMetrics, ModelResponse, PromptConfig, TokenUsage


class OpenAIRunner(BaseLLMRunner):
    """OpenAI API implementation using the official async SDK."""

    def __init__(self):
        settings = get_settings()
        api_key = settings.OPENAI_KEY.get_secret_value() if settings.OPENAI_KEY else ""
        self.client = AsyncOpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate(self, prompt_config: PromptConfig, user_prompt: str) -> ModelResponse:
        messages = []
        if prompt_config.system_prompt:
            messages.append({"role": "system", "content": prompt_config.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        start_time = time.perf_counter()
        
        response = await self.client.chat.completions.create(
            model=prompt_config.model_name,
            messages=messages,
            temperature=prompt_config.temperature,
            max_tokens=prompt_config.max_tokens,
            timeout=30.0,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        choice = response.choices[0]
        usage = response.usage

        return ModelResponse(
            raw_text=choice.message.content or "",
            token_usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            latency=LatencyMetrics(
                time_to_first_token_ms=None,  # Not tracked on non-streaming
                total_latency_ms=latency_ms,
            ),
            finish_reason=choice.finish_reason,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def stream(self, prompt_config: PromptConfig, user_prompt: str) -> AsyncGenerator[str, None]:
        messages = []
        if prompt_config.system_prompt:
            messages.append({"role": "system", "content": prompt_config.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        stream_resp = await self.client.chat.completions.create(
            model=prompt_config.model_name,
            messages=messages,
            temperature=prompt_config.temperature,
            max_tokens=prompt_config.max_tokens,
            stream=True,
            timeout=30.0,
        )

        async for chunk in stream_resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
