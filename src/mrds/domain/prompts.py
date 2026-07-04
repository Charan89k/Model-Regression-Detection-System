from typing import List, Optional

from pydantic import Field

from mrds.domain.models import MRDSDomainModel


class FewShotExample(MRDSDomainModel):
    """A single few-shot example."""
    input: str = Field(..., description="The input text for the few-shot example.")
    output: str = Field(..., description="The expected output for the few-shot example.")


class ModelConfigSchema(MRDSDomainModel):
    """Configuration for the LLM when using this prompt."""
    provider: str = Field(..., description="LLM provider name (e.g., 'openai').")
    model_name: str = Field(..., description="Model identifier (e.g., 'gpt-4-turbo').")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Max tokens.")


class PromptTemplateSchema(MRDSDomainModel):
    """The complete schema representing a versioned prompt loaded from YAML."""
    name: str = Field(..., description="Name of the prompt.")
    version: str = Field(..., description="Version of the prompt (e.g., '1.0.0').")
    author: Optional[str] = Field(default=None, description="Author or team who created the prompt.")
    description: Optional[str] = Field(default=None, description="Description of the prompt's purpose.")
    llm_config: ModelConfigSchema = Field(..., alias="model_config", description="Default model configuration.")
    system_prompt: Optional[str] = Field(default=None, description="System prompt template (Jinja2).")
    user_template: str = Field(..., description="User prompt template (Jinja2).")
    few_shot_examples: List[FewShotExample] = Field(
        default_factory=list, 
        description="List of few-shot examples."
    )
