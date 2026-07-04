import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from mrds.domain.models import RunComparison


class HTMLReportGenerator:
    """
    Generates a standalone HTML report from evaluation results and regression analyses.
    Uses Jinja2 and TailwindCSS.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        if not templates_dir:
            # Default to the directory of this file / templates
            templates_dir = str(Path(__file__).parent / "templates")

        self.env = Environment(loader=FileSystemLoader(templates_dir))

    def generate_report(
        self,
        comparison: RunComparison,
        historical_runs: List[
            Dict[str, float]
        ],  # List of dicts like {"timestamp": str, "accuracy": float}
        prompt_version: str,
        output_path: Path,
    ) -> None:
        """
        Generates the HTML file.
        """
        template = self.env.get_template("report_template.html")

        # Prepare Chart.js data
        labels = [run["timestamp"] for run in historical_runs]
        # Chart.js expects percentages if we formatted Y axis that way
        data_points = [run["accuracy"] * 100 for run in historical_runs]

        chart_data = {
            "labels": labels,
            "datasets": [
                {
                    "label": "Accuracy",
                    "data": data_points,
                    "borderColor": "#4f46e5",  # indigo-600
                    "backgroundColor": "rgba(79, 70, 229, 0.1)",
                    "borderWidth": 2,
                    "fill": True,
                    "tension": 0.3,
                }
            ],
        }

        html_content = template.render(
            candidate_run_id=comparison.candidate_run_id,
            timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            prompt_version=prompt_version,
            baseline_accuracy=comparison.baseline_accuracy,
            candidate_accuracy=comparison.candidate_accuracy,
            accuracy_delta=comparison.accuracy_delta,
            new_failures_count=len(comparison.new_failures),
            alerts=comparison.alerts,
            category_deltas=comparison.category_deltas,
            chart_data=json.dumps(chart_data),
        )

        # Write to disk
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
