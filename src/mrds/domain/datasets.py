from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from mrds.domain.models import MRDSDomainModel


class DifficultyLevel(str, Enum):
    """Difficulty levels for evaluation cases."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class DatasetMetadata(MRDSDomainModel):
    """Metadata for a versioned dataset."""

    name: str = Field(..., min_length=1, description="Unique name of the dataset.")
    version: str = Field(..., min_length=1, description="Version of the dataset (e.g., '1.0.0').")
    author: Optional[str] = Field(
        default=None, description="Author or team who created the dataset."
    )
    description: Optional[str] = Field(
        default=None, description="Description of the dataset's purpose."
    )


class DatasetCaseSchema(MRDSDomainModel):
    """A single evaluation case within a dataset."""

    id: str = Field(..., min_length=1, description="Unique string identifier for the case.")
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM, description="Difficulty of the case."
    )
    variables: Dict[str, Any] = Field(
        ..., description="Variables to inject into the prompt template."
    )
    expected_output: Optional[str] = Field(
        default=None, description="The exact expected output, if applicable."
    )
    evaluation_criteria: List[str] = Field(
        default_factory=list, description="List of criteria for an LLM judge to evaluate this case."
    )


class DatasetSchema(MRDSDomainModel):
    """The complete schema representing a versioned dataset loaded from JSON."""

    metadata: DatasetMetadata = Field(..., description="Dataset metadata.")
    cases: List[DatasetCaseSchema] = Field(
        ..., min_length=1, description="List of test cases in this dataset."
    )


class DatasetStatistics(MRDSDomainModel):
    """Calculated statistics for a dataset."""

    total_cases: int = Field(..., description="Total number of cases.")
    difficulty_distribution: Dict[DifficultyLevel, int] = Field(
        ..., description="Count of cases per difficulty level."
    )
