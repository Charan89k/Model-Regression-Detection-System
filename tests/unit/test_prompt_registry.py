import pytest
import yaml
from pathlib import Path

from mrds.domain.prompts import PromptTemplateSchema
from mrds.use_cases.prompt_registry import (
    PromptNotFoundError,
    PromptRegistry,
    PromptRenderingError,
)


@pytest.fixture
def prompt_dir(tmp_path: Path) -> Path:
    # Create a temporary directory structure for prompts
    summarize_dir = tmp_path / "summarize"
    summarize_dir.mkdir()
    
    yaml_content = {
        "name": "summarize",
        "version": "1.0",
        "author": "test-team",
        "description": "A test summarization prompt",
        "model_config": {
            "provider": "openai",
            "model_name": "gpt-4-turbo",
            "temperature": 0.5,
            "max_tokens": 100
        },
        "system_prompt": "You are a helpful assistant.",
        "user_template": "Summarize this: {{ text }}",
        "few_shot_examples": [
            {
                "input": "Long text about dogs.",
                "output": "Dogs are great."
            }
        ]
    }
    
    file_path = summarize_dir / "v1.0.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f)
        
    return tmp_path


def test_prompt_registry_loads_and_validates(prompt_dir: Path):
    registry = PromptRegistry(base_dir=prompt_dir)
    
    prompt = registry.get_prompt(name="summarize", version="1.0")
    
    assert isinstance(prompt, PromptTemplateSchema)
    assert prompt.name == "summarize"
    assert prompt.version == "1.0"
    assert prompt.model_config.provider == "openai"
    assert prompt.model_config.temperature == 0.5
    assert len(prompt.few_shot_examples) == 1
    assert prompt.few_shot_examples[0].input == "Long text about dogs."


def test_prompt_registry_caching(prompt_dir: Path, mocker):
    registry = PromptRegistry(base_dir=prompt_dir)
    
    # First load
    prompt1 = registry.get_prompt(name="summarize", version="1.0")
    
    # Delete file to prove it loads from cache
    (prompt_dir / "summarize" / "v1.0.yaml").unlink()
    
    # Second load should succeed from cache
    prompt2 = registry.get_prompt(name="summarize", version="1.0")
    
    assert prompt1 is prompt2  # Same instance


def test_prompt_registry_not_found(prompt_dir: Path):
    registry = PromptRegistry(base_dir=prompt_dir)
    
    with pytest.raises(PromptNotFoundError):
        registry.get_prompt(name="nonexistent", version="1.0")


def test_render_template_success(prompt_dir: Path):
    registry = PromptRegistry(base_dir=prompt_dir)
    
    template = "Hello {{ name }}, you have {{ count }} new messages."
    variables = {"name": "Alice", "count": 5}
    
    result = registry.render_template(template, variables)
    assert result == "Hello Alice, you have 5 new messages."


def test_render_template_missing_variables(prompt_dir: Path):
    registry = PromptRegistry(base_dir=prompt_dir)
    
    template = "Hello {{ name }}, you have {{ count }} new messages."
    variables = {"name": "Alice"}  # Missing 'count'
    
    with pytest.raises(PromptRenderingError, match="count"):
        registry.render_template(template, variables)
