from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from mrds.domain.models import RegressionThresholds
from mrds.infrastructure.db.repositories.run_repository import RunRepository
from mrds.presentation.api.dependencies import get_regression_detector, get_run_repository
from mrds.presentation.api.schemas import CompareResponse
from mrds.use_cases.regression_detector import RegressionDetector

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/compare", response_model=CompareResponse)
async def compare_runs(
    baseline_id: UUID,
    candidate_id: UUID,
    detector: RegressionDetector = Depends(get_regression_detector),
    repository: RunRepository = Depends(get_run_repository),
) -> dict[str, Any]:
    """
    Compares a candidate run against a baseline run to detect regressions.
    Generates exact deltas, new failures, and statistical warnings.
    """
    baseline_results = await repository.get_run_results(baseline_id)
    candidate_results = await repository.get_run_results(candidate_id)

    if not baseline_results:
        raise HTTPException(status_code=404, detail=f"Baseline run {baseline_id} not found.")
    if not candidate_results:
        raise HTTPException(status_code=404, detail=f"Candidate run {candidate_id} not found.")

    thresholds = RegressionThresholds(max_accuracy_drop=0.02, max_new_failures=5)

    # In a full implementation, we'd fetch categories from the dataset or pass them here
    try:
        comparison = detector.compare_runs(
            baseline_results=baseline_results,
            candidate_results=candidate_results,
            thresholds=thresholds,
            case_categories={},  # Missing case categories in this iteration
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return comparison.model_dump()
