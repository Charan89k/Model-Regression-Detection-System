class MRDSError(Exception):
    """Base exception for all Model Regression Detection System errors."""

    pass


class EntityNotFoundError(MRDSError):
    """Raised when a requested entity is not found in the repository."""

    pass


class LLMProviderError(MRDSError):
    """Raised when an external LLM provider encounters an error."""

    pass


class ConfigurationError(MRDSError):
    """Raised when there is a misconfiguration in the environment or settings."""

    pass
