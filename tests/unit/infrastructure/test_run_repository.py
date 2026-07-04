from typing import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from mrds.domain.models import (
    EvaluationResult,
    LatencyMetrics,
    ModelResponse,
    PromptConfig,
    RunMetadata,
    Score,
    TokenUsage,
)
from mrds.infrastructure.db.models.base import Base
from mrds.infrastructure.db.repositories.run_repository import RunRepository


@pytest_asyncio.fixture
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", echo=False, connect_args={"check_same_thread": False}
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_save_and_get_run(async_session: AsyncSession):
    repo = RunRepository(async_session)

    run_meta = RunMetadata(triggered_by="pytest", environment="test")
    prompt = PromptConfig(provider="openai", model_name="gpt-4", user_template="test")
    resp = ModelResponse(
        raw_text="mock response",
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        latency=LatencyMetrics(total_latency_ms=120.5),
        finish_reason="stop",
    )
    score1 = Score(metric_name="exact_match", value=1.0)
    score2 = Score(metric_name="semantic_similarity", value=0.85)

    result = EvaluationResult(
        case_id=uuid4(),
        run_metadata=run_meta,
        prompt_config=prompt,
        response=resp,
        scores=[score1, score2],
        success=True,
    )

    # Act: Save Run inside a transaction
    async with async_session.begin():
        await repo.save_run([result])

    # Act: Retrieve Run
    saved_results = await repo.get_run_results(run_meta.run_id)

    # Assert
    assert len(saved_results) == 1
    saved_res = saved_results[0]

    assert saved_res.id == result.id
    assert saved_res.case_id == result.case_id
    assert saved_res.success is True

    # Verify nested JSON columns deserialized correctly
    assert saved_res.response.raw_text == "mock response"
    assert saved_res.response.token_usage.total_tokens == 15
    assert saved_res.response.latency.total_latency_ms == 120.5
    assert saved_res.response.finish_reason == "stop"

    # Verify relationships
    assert len(saved_res.scores) == 2
    metric_names = [s.metric_name for s in saved_res.scores]
    assert "exact_match" in metric_names
    assert "semantic_similarity" in metric_names

    assert saved_res.run_metadata.triggered_by == "pytest"
