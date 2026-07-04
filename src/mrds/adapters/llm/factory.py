from mrds.adapters.llm.anthropic_runner import AnthropicRunner
from mrds.adapters.llm.base import BaseLLMRunner
from mrds.adapters.llm.gemini_runner import GeminiRunner
from mrds.adapters.llm.openai_runner import OpenAIRunner
from mrds.core.exceptions.base import MRDSError


class LLMProviderNotSupportedError(MRDSError):
    """Raised when an unknown LLM provider is requested."""
    pass


from typing import Dict

class LLMFactory:
    """Dependency injection factory for LLM runners, maintaining a cache of active runners."""

    def __init__(self, openai_key: str = "", anthropic_key: str = "", gemini_key: str = "") -> None:
        self.openai_key = openai_key
        self.anthropic_key = anthropic_key
        self.gemini_key = gemini_key
        self._cache: Dict[str, BaseLLMRunner] = {}

    def get_runner(self, provider_name: str) -> BaseLLMRunner:
        provider_name = provider_name.lower().strip()
        
        if provider_name in self._cache:
            return self._cache[provider_name]
            
        if provider_name == "openai":
            runner = OpenAIRunner(api_key=self.openai_key)
        elif provider_name == "anthropic":
            runner = AnthropicRunner(api_key=self.anthropic_key)
        elif provider_name == "gemini":
            runner = GeminiRunner(api_key=self.gemini_key)
        else:
            raise LLMProviderNotSupportedError(f"Unsupported LLM provider: {provider_name}")
            
        self._cache[provider_name] = runner
        return runner
