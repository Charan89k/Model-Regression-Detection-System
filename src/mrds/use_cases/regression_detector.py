from typing import Dict, List, Optional
from uuid import UUID

from mrds.domain.models import (
    EvaluationResult,
    RegressionAlert,
    RegressionThresholds,
    RunComparison,
)


class RegressionDetector:
    """
    Engine to compare evaluation runs, generate deltas, and flag statistical regressions.
    Decoupled from DB logic so it can operate on pure domain lists.
    """

    def compare_runs(
        self,
        baseline_results: List[EvaluationResult],
        candidate_results: List[EvaluationResult],
        thresholds: RegressionThresholds,
        case_categories: Optional[Dict[UUID, str]] = None,
    ) -> RunComparison:
        """
        Computes exact deltas between a baseline and a candidate run.
        Assumes both lists contain results for the same set of EvalCases.
        """
        baseline_map = {res.case_id: res for res in baseline_results}
        candidate_map = {res.case_id: res for res in candidate_results}

        # Intersect cases
        common_cases = set(baseline_map.keys()).intersection(set(candidate_map.keys()))
        if not common_cases:
            raise ValueError("No common cases between baseline and candidate runs.")

        baseline_success_count = sum(1 for cid in common_cases if baseline_map[cid].success)
        candidate_success_count = sum(1 for cid in common_cases if candidate_map[cid].success)

        total_common = len(common_cases)
        baseline_accuracy = baseline_success_count / total_common
        candidate_accuracy = candidate_success_count / total_common
        accuracy_delta = candidate_accuracy - baseline_accuracy

        # Identify flips
        new_failures = []
        recovered_failures = []

        # Category tracking
        # We calculate the delta per category.
        # {category: {"baseline_success": 0, "candidate_success": 0, "total": 0}}
        cat_stats: Dict[str, Dict[str, int]] = {}

        for cid in common_cases:
            b_success = baseline_map[cid].success
            c_success = candidate_map[cid].success

            if b_success and not c_success:
                new_failures.append(cid)
            elif not b_success and c_success:
                recovered_failures.append(cid)

            if case_categories and cid in case_categories:
                cat = case_categories[cid]
                if cat not in cat_stats:
                    cat_stats[cat] = {"b_success": 0, "c_success": 0, "total": 0}
                cat_stats[cat]["total"] += 1
                cat_stats[cat]["b_success"] += int(b_success)
                cat_stats[cat]["c_success"] += int(c_success)

        # Calculate category deltas
        category_deltas = self._calculate_category_deltas(cat_stats)

        # Generate Alerts
        alerts = self._generate_alerts(accuracy_delta, total_common, len(new_failures), thresholds)

        baseline_run_id = (
            str(baseline_results[0].run_metadata.run_id) if baseline_results else "unknown"
        )
        candidate_run_id = (
            str(candidate_results[0].run_metadata.run_id) if candidate_results else "unknown"
        )

        return RunComparison(
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            baseline_accuracy=baseline_accuracy,
            candidate_accuracy=candidate_accuracy,
            accuracy_delta=accuracy_delta,
            new_failures=new_failures,
            recovered_failures=recovered_failures,
            category_deltas=category_deltas,
            alerts=alerts,
        )

    def _calculate_category_deltas(self, cat_stats: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        category_deltas = {}
        for cat, stats in cat_stats.items():
            b_acc = stats["b_success"] / stats["total"]
            c_acc = stats["c_success"] / stats["total"]
            category_deltas[cat] = c_acc - b_acc
        return category_deltas

    def _generate_alerts(
        self,
        accuracy_delta: float,
        total_common: int,
        new_failures_count: int,
        thresholds: RegressionThresholds,
    ) -> List[RegressionAlert]:
        alerts = []
        if accuracy_delta < -thresholds.max_accuracy_drop:
            alerts.append(
                RegressionAlert(
                    alert_type="accuracy_drop",
                    message=(
                        f"Accuracy dropped by {-accuracy_delta * 100:.1f}%, "
                        f"exceeding threshold of {thresholds.max_accuracy_drop * 100:.1f}%"
                    ),
                    severity="critical",
                )
            )

        if new_failures_count > thresholds.max_new_failures:
            alerts.append(
                RegressionAlert(
                    alert_type="high_new_failures",
                    message=(
                        f"Found {new_failures_count} new failures, "
                        f"exceeding threshold of {thresholds.max_new_failures}"
                    ),
                    severity="critical",
                )
            )

        # Statistical Warning (Heuristic)
        # If accuracy dropped but sample size is small (< 30), it might just be noise.
        if accuracy_delta < -0.01 and total_common < 30:
            alerts.append(
                RegressionAlert(
                    alert_type="statistical_warning",
                    message=(
                        f"Accuracy drop detected on a very small sample size (n={total_common}). "
                        "Result may not be statistically significant."
                    ),
                    severity="warning",
                )
            )

        return alerts

    def detect_trends(self, historical_accuracies: List[float]) -> List[RegressionAlert]:
        """
        Analyzes a sequence of historical accuracies (ordered oldest to newest).
        Looks for slow degradations (e.g., dropping over 3+ consecutive runs).
        """
        if len(historical_accuracies) < 3:
            return []

        alerts = []

        # Check for consecutive drops
        drops = 0
        for i in range(1, len(historical_accuracies)):
            if historical_accuracies[i] < historical_accuracies[i - 1]:
                drops += 1
            else:
                drops = 0

        if drops >= 3:
            alerts.append(
                RegressionAlert(
                    alert_type="negative_trend",
                    message=f"Accuracy has dropped for {drops} consecutive runs.",
                    severity="warning",
                )
            )

        return alerts
