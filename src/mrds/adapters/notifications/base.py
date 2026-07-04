from abc import ABC, abstractmethod
from uuid import UUID

from mrds.domain.models import RunComparison


class BaseNotificationAdapter(ABC):
    """
    Abstract interface for sending alerts and notifications.
    Decouples core business logic from specific providers (e.g., Slack, Email).
    """

    @abstractmethod
    async def send_evaluation_summary(
        self, run_id: UUID, total_cases: int, success_cases: int, accuracy: float
    ) -> None:
        """Sends a summary of a completed evaluation run."""
        pass

    @abstractmethod
    async def send_regression_alert(self, comparison: RunComparison) -> None:
        """Sends an alert detailing regression failures or drift."""
        pass
