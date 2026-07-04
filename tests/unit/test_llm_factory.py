from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mrds.adapters.llm.anthropic_runner import AnthropicRunner
from mrds.adapters.llm.base import calculate_cost
from mrds.adapters.llm.factory import LLMFactory, LLMProviderNotSupportedError
from mrds.adapters.llm.gemini_runner import GeminiRunner
from mrds.adapters.llm.openai_runner import OpenAIRunner
from mrds.domain.models import PromptConfig


def test_factory_returns_correct_runner():
    factory = LLMFactory(openai_key="test", anthropic_key="test", gemini_key="test")

    openai = factory.get_runner("openai")
    assert isinstance(openai, OpenAIRunner)
    assert openai.client.api_key == "test"

    anthropic = factory.get_runner("anthropic")
    assert isinstance(anthropic, AnthropicRunner)
    assert anthropic.client.api_key == "test"

    gemini = factory.get_runner("gemini")
    assert isinstance(gemini, GeminiRunner)


def test_factory_raises_error_for_unknown_provider():
    factory = LLMFactory()
    with pytest.raises(LLMProviderNotSupportedError):
        factory.get_runner("unknown_provider")


def test_calculate_cost():
    # 1000 input tokens, 2000 output tokens for gpt-4-turbo
    cost = calculate_cost("openai", "gpt-4-turbo", 1000, 2000)
    assert cost == 0.01 + (0.03 * 2)  # 0.07

    # 500 input tokens, 500 output tokens for claude-3-opus-20240229
    cost = calculate_cost("anthropic", "claude-3-opus-20240229", 500, 500)
    assert cost == (0.015 * 0.5) + (0.075 * 0.5)  # 0.045

    # Unknown model should return 0
    cost = calculate_cost("openai", "unknown-model", 1000, 1000)
    assert cost == 0.0


@pytest.mark.asyncio
async def test_openai_runner_mocked():
    runner = OpenAIRunner(api_key="test_key")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked Response"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    with patch.object(
        runner.client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response

        prompt_config = PromptConfig(
            provider="openai",
            model_name="gpt-4-turbo",
            system_prompt="System prompt",
            user_template="User prompt",
            temperature=0.0,
        )

        result = await runner.generate(prompt_config, "Hello")

        assert result.raw_text == "Mocked Response"
        assert result.token_usage.prompt_tokens == 10
        assert result.token_usage.total_tokens == 15
        assert result.finish_reason == "stop"

        mock_create.assert_called_once()
