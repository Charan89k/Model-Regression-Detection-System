import re

from mrds.adapters.llm.factory import LLMFactory
from mrds.domain.models import EvalCase, ModelResponse, PromptConfig, Score
from mrds.use_cases.scoring.base import BaseScorer, get_scorer, register_scorer


@register_scorer("exact_match")
class ExactMatchScorer(BaseScorer):
    """Checks if the actual response exactly matches the expected output."""
    
    async def score(self, case: EvalCase, response: ModelResponse) -> Score:
        if not case.expected_output:
            return Score(metric_name="exact_match", value=0.0)
            
        ignore_case = self.config.get("ignore_case", True)
        actual = response.raw_text.strip()
        expected = case.expected_output.strip()
        
        if ignore_case:
            actual = actual.lower()
            expected = expected.lower()
            
        value = 1.0 if actual == expected else 0.0
        return Score(metric_name="exact_match", value=value)


@register_scorer("regex")
class RegexScorer(BaseScorer):
    """Evaluates if the response matches a provided regular expression."""
    
    async def score(self, case: EvalCase, response: ModelResponse) -> Score:
        pattern = self.config.get("pattern")
        if not pattern:
            raise ValueError("RegexScorer requires a 'pattern' in its config.")
            
        match = re.search(pattern, response.raw_text)
        value = 1.0 if match else 0.0
        return Score(metric_name="regex_match", value=value)


@register_scorer("semantic_similarity")
class SemanticSimilarityScorer(BaseScorer):
    """
    Mocks an embedding-based semantic similarity check. 
    In production, this calls an Embedding API and computes cosine similarity.
    """
    
    async def score(self, case: EvalCase, response: ModelResponse) -> Score:
        if not case.expected_output:
            return Score(metric_name="semantic_similarity", value=0.0)
            
        # Mock logic: word overlap overlap translated to pseudo-cosine similarity
        expected_words = set(case.expected_output.lower().split())
        actual_words = set(response.raw_text.lower().split())
        
        if not expected_words:
            return Score(metric_name="semantic_similarity", value=0.0)
            
        overlap = len(expected_words.intersection(actual_words)) / len(expected_words)
        sim_score = min(1.0, overlap + 0.2) if overlap > 0 else 0.0
        
        return Score(metric_name="semantic_similarity", value=sim_score)


@register_scorer("llm_judge")
class LLMJudgeScorer(BaseScorer):
    """Uses a powerful LLM to grade the response based on evaluation criteria."""
    
    async def score(self, case: EvalCase, response: ModelResponse) -> Score:
        provider = self.config.get("provider", "openai")
        model_name = self.config.get("model_name", "gpt-4-turbo")
        
        if not case.evaluation_criteria:
            return Score(metric_name="llm_judge", value=0.0)
            
        criteria = "\n".join([f"- {c}" for c in case.evaluation_criteria])
        system_prompt = (
            "You are an impartial judge. Evaluate the user's response based on the criteria. "
            "Reply with EXACTLY a single float between 0.0 and 1.0."
        )
        user_prompt = f"Criteria:\n{criteria}\n\nResponse to evaluate:\n{response.raw_text}"
        
        prompt_config = PromptConfig(
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_template=user_prompt,
            temperature=0.0,
            max_tokens=10
        )
        
        runner = LLMFactory.get_runner(provider)
        judge_response = await runner.generate(prompt_config, user_prompt)
        
        try:
            value = float(judge_response.raw_text.strip())
            value = max(0.0, min(1.0, value))
        except ValueError:
            value = 0.0  # Failed to parse judge output
            
        return Score(metric_name="llm_judge", value=value)


@register_scorer("weighted")
class WeightedScorer(BaseScorer):
    """
    Aggregates multiple child scorers based on a weight map.
    Example config: 
        {"weights": {"exact_match": 0.5, "llm_judge": 0.5}, "scorer_configs": {...}}
    """
    
    async def score(self, case: EvalCase, response: ModelResponse) -> Score:
        weights = self.config.get("weights", {})
        scorer_configs = self.config.get("scorer_configs", {})
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            return Score(metric_name="weighted_aggregate", value=0.0)
            
        final_value = 0.0
        for scorer_name, weight in weights.items():
            child_config = scorer_configs.get(scorer_name, {})
            scorer = get_scorer(scorer_name, **child_config)
            child_score = await scorer.score(case, response)
            final_value += child_score.value * (weight / total_weight)
            
        return Score(metric_name="weighted_aggregate", value=final_value)
