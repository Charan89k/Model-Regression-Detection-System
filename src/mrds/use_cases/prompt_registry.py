from pathlib import Path
from typing import Any, Dict

import yaml
from jinja2 import StrictUndefined, Template
from jinja2.exceptions import UndefinedError

from mrds.core.exceptions.base import MRDSError
from mrds.domain.prompts import PromptTemplateSchema


class PromptNotFoundError(MRDSError):
    """Raised when a prompt YAML file cannot be found."""

    pass


class PromptRenderingError(MRDSError):
    """Raised when Jinja2 fails to render a template due to missing variables."""

    pass


class PromptRegistry:
    """
    Registry for loading, validating, caching, and rendering YAML-based prompt templates.
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self._cache: Dict[str, PromptTemplateSchema] = {}

    def get_prompt(self, name: str, version: str) -> PromptTemplateSchema:
        """
        Loads a prompt template from disk or cache.
        Expected path format: {base_dir}/{name}/v{version}.yaml
        """
        cache_key = f"{name}@{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        file_path = (self.base_dir / name / f"v{version}.yaml").resolve()
        try:
            if not file_path.is_relative_to(self.base_dir.resolve()):
                raise PromptNotFoundError("Access denied: path traversal attempt detected.")
        except ValueError as e:
            raise PromptNotFoundError("Access denied: path traversal attempt detected.") from e

        if not file_path.exists():
            raise PromptNotFoundError(f"Prompt template not found at {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise MRDSError(f"Failed to parse YAML at {file_path}: {e}") from e

        # Validate against the Pydantic schema
        schema = PromptTemplateSchema.model_validate(raw_data)

        self._cache[cache_key] = schema
        return schema

    def render_template(self, template_str: str, variables: Dict[str, Any]) -> str:
        """
        Renders a Jinja2 template string with the provided variables.
        Uses StrictUndefined to fail fast if variables are missing.
        """
        try:
            template = Template(template_str, undefined=StrictUndefined)
            return template.render(**variables)
        except UndefinedError as e:
            raise PromptRenderingError(f"Failed to render template: {e}") from e
