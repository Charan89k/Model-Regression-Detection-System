import time
from typing import AsyncGenerator

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from mrds.core.resilience import circuit_breaker

from mrds.adapters.llm.base import BaseLLMRunner
from mrds.domain.models import LatencyMetrics, ModelResponse, PromptConfig, TokenUsage


class GeminiRunner(BaseLLMRunner):
    """Google Gemini API implementation using the google-generativeai SDK."""

    def __init__(self, api_key: str) -> None:
        # The library uses global configuration, which isn't ideal for DI, 
        # but required by the current version of google-generativeai.
        genai.configure(api_key=api_key)

    @circuit_breaker(threshold=3, cooldown=60.0)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate(self, prompt_config: PromptConfig, user_prompt: str) -> ModelResponse:
        model = genai.GenerativeModel(
            model_name=prompt_config.model_name,
            system_instruction=prompt_config.system_prompt
        )
        
        generation_config = genai.types.GenerationConfig(
            temperature=prompt_config.temperature,
            max_output_tokens=prompt_config.max_tokens,
        )

        start_time = time.perf_counter()
        
        response = await model.generate_content_async(
            contents=user_prompt,
            generation_config=generation_config
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Token usage extraction in Gemini requires accessing specific attributes
        prompt_tokens = 0
        completion_tokens = 0
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count

        finish_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason.name

        return ModelResponse(
            raw_text=response.text,
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency=LatencyMetrics(
                time_to_first_token_ms=None,
                total_latency_ms=latency_ms,
            ),
            finish_reason=finish_reason,
        )

    @circuit_breaker(threshold=3, cooldown=60.0)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def stream(self, prompt_config: PromptConfig, user_prompt: str) -> AsyncGenerator[str, None]:
        model = genai.GenerativeModel(
            model_name=prompt_config.model_name,
            system_instruction=prompt_config.system_prompt
        )
        
        generation_config = genai.types.GenerationConfig(
            temperature=prompt_config.temperature,
            max_output_tokens=prompt_config.max_tokens,
        )

        response = await model.generate_content_async(
            contents=user_prompt,
            generation_config=generation_config,
            stream=True
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text
