from pathlib import Path

from mrds.domain.models import RegressionAlert, RunComparison
from mrds.presentation.reporting.generator import HTMLReportGenerator


def test_html_report_generation(tmp_path):
    # Setup Generator
    # We pass the real templates directory path
    templates_dir = (
        Path(__file__).parent.parent.parent.parent
        / "src"
        / "mrds"
        / "presentation"
        / "reporting"
        / "templates"
    )
    generator = HTMLReportGenerator(templates_dir=str(templates_dir))

    # Mock Data
    alerts = [
        RegressionAlert(
            alert_type="accuracy_drop",
            message="Accuracy dropped significantly.",
            severity="critical",
        )
    ]

    comparison = RunComparison(
        baseline_run_id="test-baseline-id",
        candidate_run_id="test-candidate-id",
        baseline_accuracy=0.90,
        candidate_accuracy=0.85,
        accuracy_delta=-0.05,
        new_failures=[],
        recovered_failures=[],
        category_deltas={"easy": 0.0, "hard": -0.10},
        alerts=alerts,
    )

    historical = [
        {"timestamp": "2026-07-01", "accuracy": 0.88},
        {"timestamp": "2026-07-02", "accuracy": 0.90},
        {"timestamp": "2026-07-03", "accuracy": 0.85},
    ]

    output_file = tmp_path / "report.html"

    # Act
    generator.generate_report(
        comparison=comparison,
        historical_runs=historical,
        prompt_version="router v1.0",
        output_path=output_file,
    )

    # Assert
    assert output_file.exists()

    content = output_file.read_text(encoding="utf-8")

    # Verify Jinja rendering
    assert "test-candidate-id" in content
    assert "router v1.0" in content
    assert "85.0%" in content  # candidate accuracy
    assert "-5.0%" in content  # delta
    assert "Accuracy dropped significantly." in content
    assert "hard" in content
    assert "-10.0%" in content  # hard category delta

    # Verify Chart.js JSON embedded properly
    assert "2026-07-01" in content
    assert "88.0" in content
