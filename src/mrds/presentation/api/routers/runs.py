from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from mrds.domain.models import EvaluationResult
from mrds.infrastructure.db.repositories.run_repository import RunRepository
from mrds.presentation.api.dependencies import get_evaluation_orchestrator, get_run_repository
from mrds.presentation.api.schemas import RunRequest, RunSummaryResponse
from mrds.use_cases.evaluation_orchestrator import EvaluationOrchestrator

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunSummaryResponse, status_code=status.HTTP_201_CREATED)
async def trigger_run(
    request: RunRequest,
    orchestrator: EvaluationOrchestrator = Depends(get_evaluation_orchestrator),
    repository: RunRepository = Depends(get_run_repository),
) -> RunSummaryResponse:
    """
    Triggers a new evaluation run.
    1. Executes the evaluation cases concurrently.
    2. Persists the final results to the database transactionally.
    """
    try:
        results = await orchestrator.run_evaluation(
            dataset_name=request.dataset_name,
            dataset_version=request.dataset_version,
            prompt_name=request.prompt_name,
            prompt_version=request.prompt_version,
            triggered_by=request.triggered_by,
            concurrency_limit=request.concurrency_limit,
        )
    except Exception as e:
        # Catch broad exceptions from orchestrator (e.g. dataset not found)
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not results:
        raise HTTPException(status_code=400, detail="No evaluation results generated.")

    # Save to DB
    await repository.save_run(results)

    # Calculate summary
    total_cases = len(results)
    success_cases = sum(1 for r in results if r.success)
    run_id = results[0].run_metadata.run_id

    return RunSummaryResponse(
        run_id=run_id,
        total_cases=total_cases,
        success_cases=success_cases,
        accuracy=success_cases / total_cases if total_cases > 0 else 0.0,
        message="Evaluation run completed and saved successfully.",
    )


@router.get("/{run_id}")
async def get_run(
    run_id: UUID, repository: RunRepository = Depends(get_run_repository)
) -> List[EvaluationResult]:
    """Fetches full results for a specific run ID."""
    results = await repository.get_run_results(run_id)
    if not results:
        raise HTTPException(status_code=404, detail="Run not found.")
    return results
