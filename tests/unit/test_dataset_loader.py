import json
from pathlib import Path

import pytest

from mrds.domain.datasets import DatasetStatistics
from mrds.domain.models import EvalSuite
from mrds.use_cases.dataset_loader import (
    DatasetLoader,
    DatasetNotFoundError,
    DatasetValidationError,
)


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    routing_dir = tmp_path / "support_routing"
    routing_dir.mkdir()
    
    dataset_content = {
        "metadata": {
            "name": "support_routing",
            "version": "1.0",
            "description": "Test dataset"
        },
        "cases": [
            {
                "id": "c1",
                "difficulty": "easy",
                "variables": {"text": "hello"},
                "expected_output": "greeting"
            },
            {
                "id": "c2",
                "difficulty": "hard",
                "variables": {"text": "complex issue"},
                "expected_output": "support"
            }
        ]
    }
    
    with open(routing_dir / "v1.0.json", "w") as f:
        json.dump(dataset_content, f)
        
    # Write an invalid schema file
    with open(routing_dir / "v2.0.json", "w") as f:
        json.dump({"invalid": "schema"}, f)
        
    return tmp_path


def test_load_dataset_success(dataset_dir: Path):
    loader = DatasetLoader(base_dir=dataset_dir)
    suite = loader.load_dataset("support_routing", "1.0")
    
    assert isinstance(suite, EvalSuite)
    assert suite.name == "support_routing_v1.0"
    assert len(suite.cases) == 2
    assert suite.cases[0].variables["text"] == "hello"


def test_load_dataset_not_found(dataset_dir: Path):
    loader = DatasetLoader(base_dir=dataset_dir)
    with pytest.raises(DatasetNotFoundError):
        loader.load_dataset("nonexistent", "1.0")


def test_load_dataset_invalid_schema(dataset_dir: Path):
    loader = DatasetLoader(base_dir=dataset_dir)
    with pytest.raises(DatasetValidationError):
        loader.load_dataset("support_routing", "2.0")


def test_get_statistics(dataset_dir: Path):
    loader = DatasetLoader(base_dir=dataset_dir)
    stats = loader.get_statistics("support_routing", "1.0")
    
    assert isinstance(stats, DatasetStatistics)
    assert stats.total_cases == 2
    assert stats.difficulty_distribution["easy"] == 1
    assert stats.difficulty_distribution["hard"] == 1
    assert stats.difficulty_distribution["expert"] == 0
