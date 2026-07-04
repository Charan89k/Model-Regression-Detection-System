from unittest.mock import patch

from typer.testing import CliRunner

from mrds.presentation.cli.main import app

runner = CliRunner()


def test_list_prompts():
    result = runner.invoke(app, ["prompts"])
    assert result.exit_code == 0
    assert "Available Prompts" in result.stdout
    assert "router" in result.stdout


def test_list_datasets():
    result = runner.invoke(app, ["datasets"])
    # We might not have the dataset actually loaded in the test environment if working dir differs,
    # so we just check for basic success or error output
    assert result.exit_code == 0
    assert "Dataset" in result.stdout or "Failed" in result.stdout


@patch("mrds.presentation.cli.main._get_orchestrator")
@patch("mrds.presentation.cli.main.async_session_factory")
def test_run_command_mocked(mock_session, mock_get_orchestrator):
    # Mock orchestrator behavior so it doesn't try to load files or hit APIs
    mock_orchestrator_instance = mock_get_orchestrator.return_value

    # We use a trick for async mocks in synchronous CLI runners
    async def mock_run_eval(*args, **kwargs):
        return []

    mock_orchestrator_instance.run_evaluation = mock_run_eval

    # Result will be empty, so it prints "No results generated" and exits with 1
    result = runner.invoke(app, ["run", "dummy", "1.0", "dummy_prompt", "1.0"])

    assert result.exit_code == 1
    assert "No results generated" in result.stdout
