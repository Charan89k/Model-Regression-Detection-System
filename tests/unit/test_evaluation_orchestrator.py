import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mrds.domain.models import LatencyMetrics, ModelResponse, TokenUsage
from mrds.use_cases.dataset_loader import DatasetLoader
from mrds.use_cases.evaluation_orchestrator import EvaluationOrchestrator
from mrds.use_cases.prompt_registry import PromptRegistry


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    d_dir = tmp_path / "datasets" / "test_data"
    d_dir.mkdir(parents=True)
    with open(d_dir / "v1.0.json", "w") as f:
        json.dump(
            {
                "metadata": {"name": "test_data", "version": "1.0"},
                "cases": [{"id": "c1", "difficulty": "easy", "variables": {"text": "hello"}}],
            },
            f,
        )
    return tmp_path / "datasets"


@pytest.fixture
def mock_prompt(tmp_path: Path) -> Path:
    p_dir = tmp_path / "prompts" / "test_prompt"
    p_dir.mkdir(parents=True)
    with open(p_dir / "v1.0.yaml", "w") as f:
        f.write(
            """
name: test_prompt
version: 1.0
model_config:
  provider: openai
  model_name: gpt-4
  temperature: 0.0
user_template: "User says: {{ text }}"
"""
        )
    return tmp_path / "prompts"


@pytest.mark.asyncio
async def test_run_evaluation(mock_dataset: Path, mock_prompt: Path, tmp_path: Path):
    loader = DatasetLoader(mock_dataset)
    registry = PromptRegistry(mock_prompt)
    reports_dir = tmp_path / "reports"

    orchestrator = EvaluationOrchestrator(
        dataset_loader=loader, prompt_registry=registry, reports_dir=reports_dir
    )

    mock_runner = AsyncMock()
    mock_runner.generate.return_value = ModelResponse(
        raw_text="Mock Output",
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        latency=LatencyMetrics(total_latency_ms=10.0),
    )

    with patch("mrds.use_cases.evaluation_orchestrator.LLMFactory.get_runner", return_value=mock_runner):
        results = await orchestrator.run_evaluation(
            dataset_name="test_data",
            dataset_version="1.0",
            prompt_name="test_prompt",
            prompt_version="1.0",
            triggered_by="pytest",
        )

    # Verify return objects
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].response.raw_text == "Mock Output"
    assert results[0].run_metadata.triggered_by == "pytest"
    assert results[0].prompt_config.provider == "openai"

    # Verify JSONL persistence on disk
    report_files = list(reports_dir.glob("*.jsonl"))
    assert len(report_files) == 1

    with open(report_files[0], "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        persisted_data = json.loads(lines[0])
        assert persisted_data["success"] is True
        assert persisted_data["response"]["raw_text"] == "Mock Output"


@pytest.mark.asyncio
async def test_run_evaluation_concurrent_writes(mock_dataset: Path, mock_prompt: Path, tmp_path: Path):
    # Add a lot of cases to the dataset
    d_dir = mock_dataset / "test_data"
    cases = [{"id": f"c{i}", "difficulty": "easy", "variables": {"text": f"hello {i}"}} for i in range(100)]
    with open(d_dir / "v1.0.json", "w") as f:
        json.dump(
            {
                "metadata": {"name": "test_data", "version": "1.0"},
                "cases": cases,
            },
            f,
        )

    loader = DatasetLoader(mock_dataset)
    registry = PromptRegistry(mock_prompt)
    reports_dir = tmp_path / "reports_concurrent"

    orchestrator = EvaluationOrchestrator(
        dataset_loader=loader, prompt_registry=registry, reports_dir=reports_dir
    )

    mock_runner = AsyncMock()
    mock_runner.generate.return_value = ModelResponse(
        raw_text="Mock Output",
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        latency=LatencyMetrics(total_latency_ms=10.0),
    )

    with patch("mrds.use_cases.evaluation_orchestrator.LLMFactory.get_runner", return_value=mock_runner):
        results = await orchestrator.run_evaluation(
            dataset_name="test_data",
            dataset_version="1.0",
            prompt_name="test_prompt",
            prompt_version="1.0",
            triggered_by="pytest",
            concurrency_limit=50,
        )

    assert len(results) == 100
    report_files = list(reports_dir.glob("*.jsonl"))
    assert len(report_files) == 1
    
    with open(report_files[0], "r") as f:
        lines = f.readlines()
        assert len(lines) == 100
