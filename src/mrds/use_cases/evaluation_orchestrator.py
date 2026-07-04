import asyncio
import aiofiles
from pathlib import Path
from typing import List

from mrds.adapters.llm.factory import LLMFactory
from mrds.core.logging.setup import get_logger
from mrds.domain.models import (
    EvalCase,
    EvaluationResult,
    LatencyMetrics,
    ModelResponse,
    PromptConfig,
    RunMetadata,
    TokenUsage,
)
from mrds.use_cases.dataset_loader import DatasetLoader
from mrds.use_cases.prompt_registry import PromptRegistry

logger = get_logger(__name__)


class EvaluationOrchestrator:
    """
    Central orchestrator that glues Datasets, Prompts, and LLMs together
    to execute massive evaluations concurrently.
    """

    def __init__(
        self, dataset_loader: DatasetLoader, prompt_registry: PromptRegistry, reports_dir: str | Path
    ):
        self.dataset_loader = dataset_loader
        self.prompt_registry = prompt_registry
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def run_evaluation(
        self,
        dataset_name: str,
        dataset_version: str,
        prompt_name: str,
        prompt_version: str,
        triggered_by: str,
        concurrency_limit: int = 10,
    ) -> List[EvaluationResult]:
        logger.info(
            "Starting evaluation",
            dataset=f"{dataset_name}@{dataset_version}",
            prompt=f"{prompt_name}@{prompt_version}",
            concurrency_limit=concurrency_limit,
        )

        dataset = self.dataset_loader.load_dataset(dataset_name, dataset_version)
        prompt_schema = self.prompt_registry.get_prompt(prompt_name, prompt_version)

        run_metadata = RunMetadata(triggered_by=triggered_by, environment="local")

        semaphore = asyncio.Semaphore(concurrency_limit)
        report_file = self.reports_dir / f"run_{run_metadata.run_id}.jsonl"

        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def writer_task() -> None:
            async with aiofiles.open(report_file, "a", encoding="utf-8") as f:
                while True:
                    item = await queue.get()
                    if item is None:
                        queue.task_done()
                        break
                    await f.write(item + "\n")
                    queue.task_done()
        
        writer = asyncio.create_task(writer_task())

        # Map YAML PromptSchema to Domain PromptConfig
        domain_prompt_config = PromptConfig(
            provider=prompt_schema.model_config.provider,
            model_name=prompt_schema.model_config.model_name,
            system_prompt=prompt_schema.system_prompt,
            user_template=prompt_schema.user_template,
            temperature=prompt_schema.model_config.temperature,
            max_tokens=prompt_schema.model_config.max_tokens,
        )

        async def process_case(case: EvalCase) -> EvaluationResult:
            async with semaphore:
                # Render the prompt using Jinja2
                user_prompt = self.prompt_registry.render_template(
                    prompt_schema.user_template, case.variables
                )

                # Get runner via Dependency Injection
                runner = LLMFactory.get_runner(prompt_schema.model_config.provider)

                try:
                    response = await runner.generate(domain_prompt_config, user_prompt)
                    success = True
                except Exception as e:
                    logger.exception("LLM generation failed for case", case_id=str(case.id), error=str(e))
                    # Provide a fallback response so the pipeline continues
                    response = ModelResponse(
                        raw_text=f"ERROR: {str(e)}",
                        token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                        latency=LatencyMetrics(total_latency_ms=0.0),
                    )
                    success = False

                result = EvaluationResult(
                    case_id=case.id,
                    run_metadata=run_metadata,
                    prompt_config=domain_prompt_config,
                    response=response,
                    success=success,
                )

                # Persist intermediate results incrementally (fail-safe)
                await queue.put(result.model_dump_json())

                return result

        # Fan out tasks concurrently
        tasks = [process_case(case) for case in dataset.cases]
        results = await asyncio.gather(*tasks)

        # Signal writer to shutdown and wait for it
        await queue.put(None)
        await writer

        logger.info("Evaluation complete", total_cases=len(results), report_file=str(report_file))
        return list(results)
