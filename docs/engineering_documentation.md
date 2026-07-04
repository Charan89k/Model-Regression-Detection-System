# MRDS Internal Engineering Documentation

This document serves as the internal technical reference for the Model Regression Detection System (MRDS). It covers architectural patterns, design decisions, operational guidelines, and developer workflows.

---

## Architecture

MRDS follows **Domain-Driven Design (DDD)** and **Clean Architecture** principles. The codebase is structured into distinct layers to decouple business logic from external frameworks and infrastructure:

1.  **`domain/`**: Contains core enterprise logic, Pydantic data models (`EvalSuite`, `EvalCase`), schemas, and domain-specific exceptions. Has no external dependencies other than Pydantic.
2.  **`use_cases/`**: Contains application-specific business rules. Includes the `EvaluationOrchestrator`, `RegressionDetector`, `PromptRegistry`, `DatasetLoader`, and the scoring engine (`BaseScorer`).
3.  **`adapters/`**: Interfaces with the outside world. This includes LLM client wrappers (`OpenAI`, `Anthropic`, `Gemini`) and notification systems (`Slack`).
4.  **`presentation/`**: Contains the entry points for the application. Currently includes the FastAPI application (`api/`), Typer CLI (`cli/`), and HTML reporting templates (`reporting/`).
5.  **`core/`**: Shared utilities, base exceptions, and centralized configuration (`settings.py`).

## Decisions & Tradeoffs

> [!NOTE]
> **Decision:** File-based Configuration for Datasets and Prompts
> We use raw JSON for datasets (`datasets/`) and YAML for prompts (`prompts/`).
> *   **Tradeoff:** This ensures datasets and prompts are first-class citizens in source control (Git), allowing PR reviews for prompt changes and easy rollbacks. However, it sacrifices the queryability and scalability of storing these in a relational database.

> [!NOTE]
> **Decision:** Jinja2 for Prompt Templating
> We use Jinja2 with `StrictUndefined` for injecting variables into prompts.
> *   **Tradeoff:** Jinja2 provides powerful control structures (loops, conditionals) making complex prompts easy to build. The `StrictUndefined` rule ensures fail-fast behavior if a dataset case is missing a required variable, preventing silent bad evaluations, but requires strict alignment between dataset schemas and prompt templates.

> [!NOTE]
> **Decision:** Asynchronous Execution
> The core evaluation loop utilizes `asyncio` for concurrent LLM requests.
> *   **Tradeoff:** Dramatically speeds up evaluation runs over large datasets. However, it adds complexity to debugging and requires async-compatible adapters and database drivers (`asyncpg`, `aiosqlite`).

---

## Developer Workflows

### How to Add Datasets
Datasets are stored in the `datasets/` directory. To add a new dataset or version:
1.  Create a directory for your dataset if it doesn't exist: `datasets/{dataset_name}/`
2.  Create a JSON file named by version: `v{version}.json` (e.g., `datasets/support_routing/v1.0.json`).
3.  Ensure the JSON strictly conforms to the `DatasetSchema`. It must contain `metadata` and a list of `cases`. Each case must define `variables` that map exactly to the Jinja2 variables in your target prompt.

### How to Add Prompt Versions
Prompts are stored in the `prompts/` directory. To add a new prompt version:
1.  Create a directory for your prompt family: `prompts/{prompt_name}/`
2.  Create a YAML file named by version: `v{version}.yaml` (e.g., `prompts/support_routing/v1.1.yaml`).
3.  Ensure it conforms to the `PromptTemplateSchema` containing `metadata`, `system_message`, and `user_message`.
4.  Use Jinja2 syntax (e.g., `{{ variable_name }}`) for dynamic injection.

### How to Add Scorers
To implement a custom evaluation metric:
1.  Navigate to `src/mrds/use_cases/scoring/`.
2.  Create a new class that inherits from `BaseScorer`.
3.  Implement the asynchronous `score(self, expected: str, actual: str, criteria: dict) -> float` method.
4.  Import and add your new scorer to the `__all__` list in `src/mrds/use_cases/scoring/__init__.py`.
5.  Register it using the `@register_scorer("scorer_name")` decorator.

---

## Operational Runbook & Production Considerations

## Deployment
*   **Docker:** The application is fully containerized using a multi-stage Dockerfile. It runs as a non-root user (`mrds_user`) for enhanced security.
*   **Docker Compose:** A `docker-compose.yml` is provided for orchestration. It natively maps the `.env` file and handles volume mounting for the SQLite database.

## CI/CD Pipeline
> [!IMPORTANT]
> The GitHub Actions pipeline (`.github/workflows/ci.yml`) acts as a mandatory gatekeeper. Branch protection rules MUST be configured to require the `lint-and-typecheck`, `test`, and `docker-build` jobs to pass before any PR can be merged to `main`.

## Prompt Change Auditing
*   A specialized GitHub Action (`.github/workflows/prompt-changes.yml`) automatically monitors the `prompts/` directory.
*   If a developer alters a prompt, the Action injects a warning comment into the PR, reminding reviewers to execute a regression run (`mrds evaluate`) to validate the change against the golden datasets.

## Scheduled Regressions
*   The `regression.yml` GitHub Action is scheduled to run a full sweep every Sunday at midnight using the `mrds evaluate` CLI command. Failures will automatically trigger Slack alerts via the `SlackNotificationAdapter`.
