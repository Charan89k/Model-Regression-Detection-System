from typing import List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from mrds.domain.models import (
    EvaluationResult,
    LatencyMetrics,
    ModelResponse,
    PromptConfig,
    RunMetadata,
    Score,
    TokenUsage,
)
from mrds.infrastructure.db.models.orm import EvaluationModel, RunModel, ScoreModel


class RunRepository:
    """
    Repository for persisting and querying Evaluation Runs.
    Translates between pure Pydantic Domain models and SQLAlchemy ORM models.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_run(self, results: List[EvaluationResult]) -> None:
        """Saves a complete evaluation run into the database."""
        if not results:
            return

        run_metadata = results[0].run_metadata
        run_id = str(run_metadata.run_id)

        # Create Run ORM object
        run_model = RunModel(
            id=run_id,
            timestamp=run_metadata.timestamp,
            triggered_by=run_metadata.triggered_by,
            environment=run_metadata.environment,
            git_commit_hash=run_metadata.git_commit_hash,
        )

        # Create Evaluation ORM objects
        for res in results:
            eval_model = EvaluationModel(
                id=str(res.id),
                run_id=run_id,
                case_id=str(res.case_id),
                success=res.success,
                # Store complex configurations as JSON to avoid over-normalization
                prompt_config=res.prompt_config.model_dump(),
                response_raw_text=res.response.raw_text,
                response_token_usage=res.response.token_usage.model_dump(),
                response_latency=res.response.latency.model_dump(),
                response_finish_reason=res.response.finish_reason,
            )

            # Map Scores
            for score in res.scores:
                score_model = ScoreModel(
                    metric_name=score.metric_name, value=score.value, max_value=score.max_value
                )
                eval_model.scores.append(score_model)

            run_model.evaluations.append(eval_model)

        self.session.add(run_model)
        await self.session.flush()

    async def get_run_results(self, run_id: UUID) -> List[EvaluationResult]:
        """Retrieves all evaluation results for a specific run ID."""
        stmt = (
            select(EvaluationModel)
            .where(EvaluationModel.run_id == str(run_id))
            .options(joinedload(EvaluationModel.run), joinedload(EvaluationModel.scores))
        )

        result = await self.session.execute(stmt)
        eval_models = result.unique().scalars().all()

        domain_results = []
        for em in eval_models:
            # Map ORM back to Pydantic Domain models
            run_meta = RunMetadata(
                run_id=UUID(em.run.id),
                timestamp=em.run.timestamp,
                triggered_by=em.run.triggered_by,
                environment=em.run.environment,
                git_commit_hash=em.run.git_commit_hash,
            )

            prompt_config = PromptConfig.model_validate(em.prompt_config)

            response = ModelResponse(
                raw_text=em.response_raw_text,
                token_usage=TokenUsage.model_validate(em.response_token_usage),
                latency=LatencyMetrics.model_validate(em.response_latency),
                finish_reason=em.response_finish_reason,
            )

            scores = [
                Score(metric_name=sm.metric_name, value=sm.value, max_value=sm.max_value)
                for sm in em.scores
            ]

            domain_results.append(
                EvaluationResult(
                    id=UUID(em.id),
                    case_id=UUID(em.case_id),
                    run_metadata=run_meta,
                    prompt_config=prompt_config,
                    response=response,
                    scores=scores,
                    success=em.success,
                )
            )

        return domain_results
