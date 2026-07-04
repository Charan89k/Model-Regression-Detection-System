from uuid import uuid4

import pytest

from mrds.domain.models import EvalCase, LatencyMetrics, ModelResponse, TokenUsage
from mrds.use_cases.scoring.base import ScorerNotFoundError, get_scorer


@pytest.fixture
def dummy_case():
    return EvalCase(
        id=uuid4(),
        variables={},
        expected_output="Hello World",
        evaluation_criteria=["Must say hello"],
    )


@pytest.fixture
def dummy_response():
    return ModelResponse(
        raw_text="hello world ",
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        latency=LatencyMetrics(total_latency_ms=10.0),
    )


@pytest.mark.asyncio
async def test_exact_match_scorer(dummy_case, dummy_response):
    scorer = get_scorer("exact_match")
    score = await scorer.score(dummy_case, dummy_response)
    # Default is ignore_case=True and it strips whitespace
    assert score.value == 1.0

    scorer_strict = get_scorer("exact_match", ignore_case=False)
    score_strict = await scorer_strict.score(dummy_case, dummy_response)
    assert score_strict.value == 0.0


@pytest.mark.asyncio
async def test_regex_scorer(dummy_case):
    scorer = get_scorer("regex", pattern=r"\d{3}-\d{4}")

    resp_match = ModelResponse(
        raw_text="My number is 555-1234.",
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        latency=LatencyMetrics(total_latency_ms=10.0),
    )
    score_match = await scorer.score(dummy_case, resp_match)
    assert score_match.value == 1.0

    resp_no_match = ModelResponse(
        raw_text="My number is 5551234.",
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        latency=LatencyMetrics(total_latency_ms=10.0),
    )
    score_no_match = await scorer.score(dummy_case, resp_no_match)
    assert score_no_match.value == 0.0


@pytest.mark.asyncio
async def test_weighted_scorer(dummy_case, dummy_response):
    config = {
        "weights": {"exact_match": 0.8, "regex": 0.2},
        "scorer_configs": {"regex": {"pattern": r"hello"}},
    }
    scorer = get_scorer("weighted", **config)
    score = await scorer.score(dummy_case, dummy_response)

    # exact_match gives 1.0, regex gives 1.0 -> total 1.0
    assert score.value == 1.0


def test_registry_not_found():
    with pytest.raises(ScorerNotFoundError):
        get_scorer("invalid_scorer")
