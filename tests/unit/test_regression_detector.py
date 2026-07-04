from uuid import uuid4

import pytest

from mrds.domain.models import (
    EvaluationResult,
    LatencyMetrics,
    ModelResponse,
    PromptConfig,
    RegressionThresholds,
    RunMetadata,
    TokenUsage,
)
from mrds.use_cases.regression_detector import RegressionDetector


@pytest.fixture
def mock_eval_results():
    case1_id = uuid4()
    case2_id = uuid4()
    case3_id = uuid4()

    baseline_run = RunMetadata(triggered_by="test", environment="local")
    candidate_run = RunMetadata(triggered_by="test2", environment="local")

    prompt = PromptConfig(provider="openai", model_name="gpt-4", user_template="test")
    resp = ModelResponse(
        raw_text="mock",
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        latency=LatencyMetrics(total_latency_ms=1.0),
    )

    # Baseline: c1 passes, c2 passes, c3 fails
    # Candidate: c1 passes, c2 fails (new failure), c3 passes (recovered)

    b1 = EvaluationResult(
        case_id=case1_id,
        run_metadata=baseline_run,
        prompt_config=prompt,
        response=resp,
        success=True,
    )
    b2 = EvaluationResult(
        case_id=case2_id,
        run_metadata=baseline_run,
        prompt_config=prompt,
        response=resp,
        success=True,
    )
    b3 = EvaluationResult(
        case_id=case3_id,
        run_metadata=baseline_run,
        prompt_config=prompt,
        response=resp,
        success=False,
    )

    c1 = EvaluationResult(
        case_id=case1_id,
        run_metadata=candidate_run,
        prompt_config=prompt,
        response=resp,
        success=True,
    )
    c2 = EvaluationResult(
        case_id=case2_id,
        run_metadata=candidate_run,
        prompt_config=prompt,
        response=resp,
        success=False,
    )
    c3 = EvaluationResult(
        case_id=case3_id,
        run_metadata=candidate_run,
        prompt_config=prompt,
        response=resp,
        success=True,
    )

    categories = {case1_id: "easy", case2_id: "hard", case3_id: "hard"}

    return [b1, b2, b3], [c1, c2, c3], categories, case2_id, case3_id


def test_compare_runs(mock_eval_results):
    baseline, candidate, categories, case2_id, case3_id = mock_eval_results
    detector = RegressionDetector()
    thresholds = RegressionThresholds(max_accuracy_drop=0.01, max_new_failures=0)

    comparison = detector.compare_runs(
        baseline_results=baseline,
        candidate_results=candidate,
        thresholds=thresholds,
        case_categories=categories,
    )

    # Baseline accuracy: 2/3 = 66.6%
    # Candidate accuracy: 2/3 = 66.6%
    # Delta: 0.0
    assert abs(comparison.accuracy_delta) < 0.0001

    # New failure: case2
    assert comparison.new_failures == [case2_id]

    # Recovered: case3
    assert comparison.recovered_failures == [case3_id]

    # Category deltas
    # easy: baseline 1/1, candidate 1/1 -> delta 0
    # hard: baseline 1/2 (c2 pass, c3 fail), candidate 1/2 (c2 fail, c3 pass) -> delta 0
    assert comparison.category_deltas["easy"] == 0.0
    assert comparison.category_deltas["hard"] == 0.0

    # Alerts: we set max_new_failures=0, and we have 1.
    assert len(comparison.alerts) == 1
    assert comparison.alerts[0].alert_type == "high_new_failures"


def test_detect_trends():
    detector = RegressionDetector()

    # 3 consecutive drops
    alerts = detector.detect_trends([0.9, 0.85, 0.80, 0.75])
    assert len(alerts) == 1
    assert alerts[0].alert_type == "negative_trend"

    # No consecutive drops
    alerts2 = detector.detect_trends([0.9, 0.85, 0.9, 0.85])
    assert len(alerts2) == 0


def test_statistical_warning(mock_eval_results):
    baseline, candidate, categories, case2_id, case3_id = mock_eval_results
    # Make candidate have 0 success to force accuracy drop
    for c in candidate:
        c.success = False

    detector = RegressionDetector()
    thresholds = RegressionThresholds(max_accuracy_drop=0.05)

    comparison = detector.compare_runs(baseline, candidate, thresholds)

    # Should have accuracy_drop critical alert, and statistical_warning due to small sample (n=3)
    alert_types = [a.alert_type for a in comparison.alerts]
    assert "accuracy_drop" in alert_types
    assert "statistical_warning" in alert_types
