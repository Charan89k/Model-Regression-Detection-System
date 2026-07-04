import time
from functools import wraps
from typing import Any, Callable, Coroutine, TypeVar

from mrds.core.logging.setup import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and blocks a call."""

    pass


class CircuitBreaker:
    """
    A simple thread-safe/async-safe stateful circuit breaker.
    If 'failure_threshold' sequential failures occur, the breaker opens.
    It remains open for 'cooldown_seconds'.
    """

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self.failures = 0
        self.last_failure_time = 0.0
        self.is_open = False

    def record_success(self) -> None:
        if self.failures > 0 or self.is_open:
            logger.info("Circuit breaker closed (recovered)")
        self.failures = 0
        self.is_open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            if not self.is_open:
                logger.error("Circuit breaker tripped open!", failures=self.failures)
            self.is_open = True

    def can_execute(self) -> bool:
        if not self.is_open:
            return True

        # Check if cooldown has elapsed
        if time.time() - self.last_failure_time > self.cooldown_seconds:
            # Half-open state - allow a test request
            return True

        return False


def circuit_breaker(threshold: int = 5, cooldown: float = 60.0) -> Callable[..., Any]:
    """Decorator to apply a circuit breaker to an async function."""
    breaker = CircuitBreaker(failure_threshold=threshold, cooldown_seconds=cooldown)

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            if not breaker.can_execute():
                logger.warning("Call rejected due to open circuit breaker", function=func.__name__)
                raise CircuitOpenError("Circuit is OPEN")

            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception:
                # We don't want to trip on simple validation errors, but we trip on all
                # for simplicity here, as LLM calls failing implies provider issues.
                breaker.record_failure()
                raise

        return wrapper

    return decorator
