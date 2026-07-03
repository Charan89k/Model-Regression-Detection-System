from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Payload to trigger a new evaluation run."""
    dataset_name: str = Field(..., description="Name of the dataset")
    dataset_version: str = Field(..., description="Version of the dataset")
    prompt_name: str = Field(..., description="Name of the prompt")
    prompt_version: str = Field(..., description="Version of the prompt")
    triggered_by: str = Field(default="api", description="Who triggered this run")
    concurrency_limit: int = Field(default=10, ge=1, le=50, description="Max concurrent LLM requests")


class RunSummaryResponse(BaseModel):
    """Summary of a completed evaluation run."""
    run_id: UUID
    total_cases: int
    success_cases: int
    accuracy: float
    message: str


class CompareRequest(BaseModel):
    """Payload to compare two historical runs."""
    baseline_run_id: UUID
    candidate_run_id: UUID


class RegressionAlertDTO(BaseModel):
    alert_type: str
    message: str
    severity: str


class CompareResponse(BaseModel):
    """Response containing the regression detection results."""
    baseline_run_id: str
    candidate_run_id: str
    baseline_accuracy: float
    candidate_accuracy: float
    accuracy_delta: float
    new_failures: List[UUID]
    recovered_failures: List[UUID]
    category_deltas: Dict[str, float]
    alerts: List[RegressionAlertDTO]
