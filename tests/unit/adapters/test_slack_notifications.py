import uuid
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from mrds.adapters.notifications.slack import SlackNotificationAdapter
from mrds.core.config import get_settings
from mrds.domain.models import RegressionAlert, RunComparison


@pytest.fixture
def mock_settings():
    settings = get_settings()
    settings.SLACK_WEBHOOK_URL = SecretStr("MOCK_SLACK_WEBHOOK_URL")
    return settings


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_send_evaluation_summary_success(mock_post, mock_settings):
    adapter = SlackNotificationAdapter()
    adapter.webhook_url = mock_settings.SLACK_WEBHOOK_URL.get_secret_value()
    
    run_id = uuid.uuid4()
    
    # 95% accuracy (Success)
    await adapter.send_evaluation_summary(run_id, 100, 95, 0.95)
    
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == adapter.webhook_url
    
    payload = kwargs["json"]
    # Check green color
    assert payload["attachments"][0]["color"] == "#36a64f"
    assert "Evaluation Run Completed" in payload["attachments"][0]["blocks"][0]["text"]["text"]


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_send_regression_alert_critical(mock_post, mock_settings):
    adapter = SlackNotificationAdapter()
    adapter.webhook_url = mock_settings.SLACK_WEBHOOK_URL.get_secret_value()
    
    comparison = RunComparison(
        baseline_run_id="b", candidate_run_id="c",
        baseline_accuracy=0.90, candidate_accuracy=0.85,
        accuracy_delta=-0.05, new_failures=[uuid.uuid4()], recovered_failures=[],
        category_deltas={},
        alerts=[
            RegressionAlert(alert_type="drop", message="Huge drop", severity="critical")
        ]
    )
    
    await adapter.send_regression_alert(comparison)
    
    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    
    # Check red color for critical
    assert payload["attachments"][0]["color"] == "#ff0000"
    assert "Critical Regression Detected" in payload["attachments"][0]["blocks"][0]["text"]["text"]


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_skip_if_no_webhook(mock_post):
    settings = get_settings()
    settings.SLACK_WEBHOOK_URL = None
    
    adapter = SlackNotificationAdapter()
    await adapter.send_evaluation_summary(uuid.uuid4(), 100, 100, 1.0)
    
    # Should not trigger network call
    mock_post.assert_not_called()
