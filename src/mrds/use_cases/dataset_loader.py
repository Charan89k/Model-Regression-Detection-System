import json
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from mrds.core.exceptions.base import MRDSError
from mrds.domain.datasets import DatasetSchema, DatasetStatistics, DifficultyLevel
from mrds.domain.models import EvalCase, EvalSuite


class DatasetNotFoundError(MRDSError):
    """Raised when a dataset JSON file cannot be found."""
    pass


class DatasetValidationError(MRDSError):
    """Raised when a dataset JSON file fails schema validation."""
    pass


class DatasetLoader:
    """
    Loads, validates, and converts JSON datasets into EvalSuite domain models.
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def load_dataset(self, name: str, version: str) -> EvalSuite:
        """
        Loads a dataset from disk, validates it, and converts it to an EvalSuite.
        Expected path format: {base_dir}/{name}/v{version}.json
        """
        schema = self._load_and_validate(name, version)
        
        # Convert schema to domain model EvalSuite
        cases = []
        for case_schema in schema.cases:
            cases.append(
                EvalCase(
                    id=uuid4(),
                    variables=case_schema.variables,
                    expected_output=case_schema.expected_output,
                    evaluation_criteria=case_schema.evaluation_criteria,
                )
            )

        return EvalSuite(
            id=uuid4(),
            name=f"{schema.metadata.name}_v{schema.metadata.version}",
            description=schema.metadata.description,
            cases=cases,
        )

    def get_statistics(self, name: str, version: str) -> DatasetStatistics:
        """
        Calculates and returns statistics for a specific dataset version.
        """
        schema = self._load_and_validate(name, version)
        
        dist = {level: 0 for level in DifficultyLevel}
        for case in schema.cases:
            dist[case.difficulty] += 1
            
        return DatasetStatistics(
            total_cases=len(schema.cases),
            difficulty_distribution=dist
        )

    def _load_and_validate(self, name: str, version: str) -> DatasetSchema:
        file_path = self.base_dir / name / f"v{version}.json"
        
        if not file_path.exists():
            raise DatasetNotFoundError(f"Dataset not found at {file_path}")
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetValidationError(f"Failed to parse JSON at {file_path}: {e}")
            
        try:
            return DatasetSchema.model_validate(raw_data)
        except ValidationError as e:
            raise DatasetValidationError(f"Dataset schema validation failed: {e}")
