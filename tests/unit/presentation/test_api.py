from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from mrds.presentation.api.dependencies import get_evaluation_orchestrator
from mrds.presentation.api.main import app


@pytest.fixture
def test_client():
    return TestClient(app)


def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "mrds"}


def test_trigger_run_missing_fields(test_client):
    # Missing required dataset_version and prompt fields
    response = test_client.post("/runs", json={"dataset_name": "support"})
    assert response.status_code == 422  # FastAPI validation error


def test_trigger_run_success(test_client):
    # Mock the orchestrator to avoid DB/File I/O
    mock_orchestrator = AsyncMock()

    # Since we return a fake empty list, the router logic raises:
    # "No evaluation results generated."
    # Let's test that logic first.

    mock_orchestrator.run_evaluation.return_value = []

    app.dependency_overrides[get_evaluation_orchestrator] = lambda: mock_orchestrator

    response = test_client.post(
        "/runs",
        json={
            "dataset_name": "support_routing",
            "dataset_version": "1.0",
            "prompt_name": "router",
            "prompt_version": "1.0",
        },
    )

    assert response.status_code == 400
    assert "No evaluation results" in response.json()["detail"]

    # Cleanup overrides
    app.dependency_overrides.clear()
