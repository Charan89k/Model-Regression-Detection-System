from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from mrds.infrastructure.db.repositories.run_repository import RunRepository
from mrds.infrastructure.db.session import async_session_factory
from mrds.use_cases.dataset_loader import DatasetLoader
from mrds.use_cases.evaluation_orchestrator import EvaluationOrchestrator
from mrds.use_cases.prompt_registry import PromptRegistry
from mrds.use_cases.regression_detector import RegressionDetector


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to provide a database session."""
    async with async_session_factory() as session:
        yield session


def get_run_repository(session: AsyncSession = Depends(get_db_session)) -> RunRepository:
    """Dependency to provide the RunRepository."""
    return RunRepository(session)


def get_dataset_loader() -> DatasetLoader:
    """Dependency to provide the DatasetLoader."""
    return DatasetLoader(base_dir="datasets")


def get_prompt_registry() -> PromptRegistry:
    """Dependency to provide the PromptRegistry."""
    return PromptRegistry(base_dir="prompts")


def get_evaluation_orchestrator(
    loader: DatasetLoader = Depends(get_dataset_loader),
    registry: PromptRegistry = Depends(get_prompt_registry),
) -> EvaluationOrchestrator:
    """Dependency to provide the EvaluationOrchestrator."""
    return EvaluationOrchestrator(
        dataset_loader=loader, prompt_registry=registry, reports_dir="reports"
    )


def get_regression_detector() -> RegressionDetector:
    """Dependency to provide the RegressionDetector."""
    return RegressionDetector()
