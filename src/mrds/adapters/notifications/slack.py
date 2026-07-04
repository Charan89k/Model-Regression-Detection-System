import logging
from typing import Any
from uuid import UUID

import httpx

from mrds.adapters.notifications.base import BaseNotificationAdapter
from mrds.core.config import get_settings
from mrds.domain.models import RunComparison

logger = logging.getLogger(__name__)


class SlackNotificationAdapter(BaseNotificationAdapter):
    """
    Sends notifications to Slack using Incoming Webhooks and Block Kit formatting.
    Fails silently if SLACK_WEBHOOK_URL is not configured.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.webhook_url = (
            self.settings.SLACK_WEBHOOK_URL.get_secret_value()
            if self.settings.SLACK_WEBHOOK_URL
            else None
        )

    async def _post_payload(self, payload: dict[str, Any]) -> None:
        """Internal helper to execute the HTTP POST to Slack."""
        if not self.webhook_url:
            logger.debug("SLACK_WEBHOOK_URL not configured. Skipping notification.")
            return

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.webhook_url, json=payload, timeout=5.0)
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    async def send_evaluation_summary(
        self, run_id: UUID, total_cases: int, success_cases: int, accuracy: float
    ) -> None:
        """Formats and sends a run summary using Slack Block Kit."""
        color = "#36a64f" if accuracy >= 0.9 else "#ffcc00"
        if accuracy < 0.7:
            color = "#ff0000"

        payload = {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": "Evaluation Run Completed :test_tube:",
                                "emoji": True,
                            },
                        },
                        {
                            "type": "section",
                            "fields": [
                                {"type": "mrkdwn", "text": f"*Run ID:*\n`{str(run_id)}`"},
                                {"type": "mrkdwn", "text": f"*Accuracy:*\n{accuracy * 100:.1f}%"},
                                {"type": "mrkdwn", "text": f"*Total Cases:*\n{total_cases}"},
                                {"type": "mrkdwn", "text": f"*Success:*\n{success_cases}"},
                            ],
                        },
                    ],
                }
            ]
        }
        await self._post_payload(payload)

    async def send_regression_alert(self, comparison: RunComparison) -> None:
        """Formats and sends regression/drift alerts."""
        has_critical = any(a.severity == "critical" for a in comparison.alerts)
        has_warning = any(a.severity == "warning" for a in comparison.alerts)

        color = "#ff0000" if has_critical else ("#ffcc00" if has_warning else "#36a64f")
        header_text = (
            "🚨 Critical Regression Detected"
            if has_critical
            else ("⚠️ Regression Warning" if has_warning else "✅ No Regressions Detected")
        )

        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": header_text, "emoji": True}},
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Baseline Accuracy:*\n{comparison.baseline_accuracy * 100:.1f}%",
                    },
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"*Candidate Accuracy:*\n{comparison.candidate_accuracy * 100:.1f}%"
                        ),
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Accuracy Delta:*\n{comparison.accuracy_delta * 100:+.1f}%",
                    },
                    {"type": "mrkdwn", "text": f"*New Failures:*\n{len(comparison.new_failures)}"},
                ],
            },
        ]

        if comparison.alerts:
            alert_text = "\n".join(
                [f"• [{a.severity.upper()}] {a.message}" for a in comparison.alerts]
            )
            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*Alerts:*\n{alert_text}"}}
            )

        payload = {"attachments": [{"color": color, "blocks": blocks}]}
        await self._post_payload(payload)
