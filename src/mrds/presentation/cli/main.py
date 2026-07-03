import asyncio
from typing import Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table

from mrds.domain.models import RegressionThresholds
from mrds.infrastructure.db.repositories.run_repository import RunRepository
from mrds.infrastructure.db.session import async_session_factory
from mrds.use_cases.dataset_loader import DatasetLoader
from mrds.use_cases.evaluation_orchestrator import EvaluationOrchestrator
from mrds.use_cases.prompt_registry import PromptRegistry
from mrds.use_cases.regression_detector import RegressionDetector

app = typer.Typer(
    name="mrds",
    help="Model Regression Detection System CLI",
    add_completion=False,
)

console = Console()


def _get_orchestrator() -> EvaluationOrchestrator:
    return EvaluationOrchestrator(
        dataset_loader=DatasetLoader(base_dir="datasets"),
        prompt_registry=PromptRegistry(base_dir="prompts"),
        reports_dir="reports",
    )


@app.command("run")
def run_evaluation(
    dataset_name: str = typer.Argument(..., help="Name of the dataset"),
    dataset_version: str = typer.Argument(..., help="Version of the dataset"),
    prompt_name: str = typer.Argument(..., help="Name of the prompt"),
    prompt_version: str = typer.Argument(..., help="Version of the prompt"),
    concurrency: int = typer.Option(10, help="Max concurrent LLM requests"),
):
    """Trigger a new evaluation run."""

    async def _run():
        orchestrator = _get_orchestrator()
        
        with console.status("[bold green]Evaluating test cases...") as status:
            results = await orchestrator.run_evaluation(
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                triggered_by="cli",
                concurrency_limit=concurrency,
            )

        if not results:
            console.print("[bold red]No results generated.[/bold red]")
            raise typer.Exit(1)

        with console.status("[bold blue]Saving results to database..."):
            async with async_session_factory() as session:
                repo = RunRepository(session)
                async with session.begin():
                    await repo.save_run(results)

        total = len(results)
        success = sum(1 for r in results if r.success)
        accuracy = (success / total) * 100 if total > 0 else 0.0

        console.print(f"[bold green]Run completed successfully![/bold green]")
        console.print(f"Run ID: [bold]{results[0].run_metadata.run_id}[/bold]")
        console.print(f"Total Cases: {total}")
        console.print(f"Success Cases: {success} ({accuracy:.1f}%)")

    asyncio.run(_run())


@app.command("compare")
def compare_runs(
    baseline_id: UUID = typer.Argument(..., help="Run ID for the baseline"),
    candidate_id: UUID = typer.Argument(..., help="Run ID for the candidate"),
):
    """Compare a candidate run against a baseline to detect regressions."""

    async def _compare():
        async with async_session_factory() as session:
            repo = RunRepository(session)
            baseline = await repo.get_run_results(baseline_id)
            candidate = await repo.get_run_results(candidate_id)

        if not baseline:
            console.print(f"[bold red]Baseline run {baseline_id} not found.[/bold red]")
            raise typer.Exit(1)
        if not candidate:
            console.print(f"[bold red]Candidate run {candidate_id} not found.[/bold red]")
            raise typer.Exit(1)

        detector = RegressionDetector()
        thresholds = RegressionThresholds()
        
        comparison = detector.compare_runs(
            baseline_results=baseline,
            candidate_results=candidate,
            thresholds=thresholds,
        )

        console.print(f"\n[bold]Regression Analysis[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("Baseline Accuracy", f"{comparison.baseline_accuracy*100:.1f}%")
        table.add_row("Candidate Accuracy", f"{comparison.candidate_accuracy*100:.1f}%")
        
        color = "red" if comparison.accuracy_delta < 0 else "green"
        table.add_row("Accuracy Delta", f"[{color}]{comparison.accuracy_delta*100:+.1f}%[/{color}]")
        table.add_row("New Failures", str(len(comparison.new_failures)))
        table.add_row("Recovered", str(len(comparison.recovered_failures)))

        console.print(table)

        if comparison.alerts:
            console.print("\n[bold red]Alerts Triggered:[/bold red]")
            for alert in comparison.alerts:
                console.print(f"- [{alert.severity.upper()}] {alert.message}")

    asyncio.run(_compare())


@app.command("report")
def report_run(
    run_id: UUID = typer.Argument(..., help="Run ID to view"),
):
    """View details of a specific historical run."""

    async def _report():
        async with async_session_factory() as session:
            repo = RunRepository(session)
            results = await repo.get_run_results(run_id)

        if not results:
            console.print(f"[bold red]Run {run_id} not found.[/bold red]")
            raise typer.Exit(1)

        total = len(results)
        success = sum(1 for r in results if r.success)
        accuracy = (success / total) * 100 if total > 0 else 0.0

        console.print(f"\n[bold]Run Report: {run_id}[/bold]")
        console.print(f"Timestamp: {results[0].run_metadata.timestamp}")
        console.print(f"Environment: {results[0].run_metadata.environment}")
        console.print(f"Accuracy: {accuracy:.1f}% ({success}/{total} cases passed)")

    asyncio.run(_report())


@app.command("datasets")
def list_datasets():
    """List all available datasets in the local registry."""
    loader = DatasetLoader(base_dir="datasets")
    try:
        dataset = loader.load_dataset("support_routing", "1.0")
        
        table = Table(title="Available Datasets", show_header=True, header_style="bold cyan")
        table.add_column("Dataset Name")
        table.add_column("Version")
        table.add_column("Total Cases")
        
        table.add_row(dataset.name, "1.0", str(len(dataset.cases)))
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Failed to load datasets: {e}[/bold red]")


@app.command("prompts")
def list_prompts():
    """List all available prompts in the local registry."""
    # Since we didn't implement prompt iteration, we'll just mock a display for now
    table = Table(title="Available Prompts", show_header=True, header_style="bold green")
    table.add_column("Prompt Name")
    table.add_column("Version")
    
    table.add_row("router", "1.0")
    console.print(table)


if __name__ == "__main__":
    app()
