from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Type

from mrds.core.exceptions.base import MRDSError
from mrds.domain.models import EvalCase, ModelResponse, Score


class ScorerNotFoundError(MRDSError):
    """Raised when a requested scorer is not found in the registry."""
    pass


class BaseScorer(ABC):
    """Abstract base class for all scorers."""

    def __init__(self, **kwargs: Any):
        self.config = kwargs

    @abstractmethod
    async def score(self, case: EvalCase, response: ModelResponse) -> Score:
        """Evaluates the model response against the case and returns a Score."""
        pass


# ---------------------------------------------------------------------------
# Plugin Registry
# ---------------------------------------------------------------------------

_SCORER_REGISTRY: Dict[str, Type[BaseScorer]] = {}


def register_scorer(name: str) -> Callable[[Type[BaseScorer]], Type[BaseScorer]]:
    """Decorator to register a custom scorer."""
    def decorator(cls: Type[BaseScorer]) -> Type[BaseScorer]:
        _SCORER_REGISTRY[name] = cls
        return cls
    return decorator


def get_scorer(name: str, **kwargs: Any) -> BaseScorer:
    """Instantiates a registered scorer by name."""
    if name not in _SCORER_REGISTRY:
        raise ScorerNotFoundError(f"Scorer '{name}' is not registered.")
    return _SCORER_REGISTRY[name](**kwargs)
