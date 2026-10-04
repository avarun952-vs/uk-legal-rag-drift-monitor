from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .config import settings
from .drift import kl_divergence, population_stability_index
from .generator import generate_answer
from .retrieval import Retriever


@dataclass
class ChatResult:
    answer: str
    passages: List[Dict]
    drift_metrics: Dict[str, float]


class LegalChatbotPipeline:
    def __init__(self) -> None:
        self.retriever = Retriever(settings.embedding_model, settings.vector_dir)
        self.retriever.load()

    def ask(self, question: str) -> ChatResult:
        passages = self.retriever.search(question, top_k=settings.top_k)
        scores = [p["score"] for p in passages]

        baseline_mean = self.retriever.baseline_stats.get("embedding_norm_mean", 1.0)
        baseline_std = self.retriever.baseline_stats.get("embedding_norm_std", 0.01)

        expected_distribution = [baseline_mean - baseline_std, baseline_mean, baseline_mean + baseline_std]
        actual_distribution = scores if scores else [0.0, 0.0, 0.0]

        drift_metrics = {
            "psi": population_stability_index(expected_distribution, actual_distribution, bins=5),
            "kl_divergence": kl_divergence(expected_distribution, actual_distribution, bins=5),
        }

        answer = generate_answer(
            provider=settings.llm_provider,
            question=question,
            passages=passages,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            openai_base_url=settings.openai_base_url,
            ollama_model=settings.ollama_model,
            ollama_base_url=settings.ollama_base_url,
        )
        return ChatResult(answer=answer, passages=passages, drift_metrics=drift_metrics)
