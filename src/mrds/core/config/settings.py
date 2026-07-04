import os
from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class BaseAppConfig(BaseSettings):
    """Base application configuration."""

    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT)
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: Literal["text", "json"] = Field(default="text")

    # LLM Providers (Secrets)
    OPENAI_KEY: SecretStr | None = Field(default=None)
    ANTHROPIC_KEY: SecretStr | None = Field(default=None)
    GEMINI_KEY: SecretStr | None = Field(default=None)

    # Database
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///mrds.db")

    # Notifications
    SLACK_WEBHOOK_URL: SecretStr | None = Field(default=None)

    # API
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


class DevelopmentConfig(BaseAppConfig):
    """Configuration for development environment."""
    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT)
    LOG_LEVEL: str = Field(default="DEBUG")
    LOG_FORMAT: Literal["text", "json"] = Field(default="text")


class TestingConfig(BaseAppConfig):
    """Configuration for testing environment."""
    ENVIRONMENT: Environment = Field(default=Environment.TESTING)
    LOG_LEVEL: str = Field(default="DEBUG")
    LOG_FORMAT: Literal["text", "json"] = Field(default="text")
    
    # In tests, we enforce these defaults if not overridden by env vars
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///:memory:"
    )
    OPENAI_KEY: SecretStr = Field(default=SecretStr("test_openai_key"))
    ANTHROPIC_KEY: SecretStr = Field(default=SecretStr("test_anthropic_key"))
    GEMINI_KEY: SecretStr = Field(default=SecretStr("test_gemini_key"))


class ProductionConfig(BaseAppConfig):
    """Configuration for production environment."""
    ENVIRONMENT: Environment = Field(default=Environment.PRODUCTION)
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: Literal["text", "json"] = Field(default="json")


@lru_cache
def get_settings() -> BaseAppConfig:
    """
    Configuration factory that returns the appropriate settings based on the environment.
    Uses lru_cache to ensure settings are only parsed once.
    """
    env_state = os.getenv("ENVIRONMENT", "development").lower()
    
    if env_state == Environment.TESTING.value:
        return TestingConfig()
    elif env_state == Environment.PRODUCTION.value:
        return ProductionConfig()
    
    return DevelopmentConfig()
