from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List

from .config import settings
from .drift import score_window_drift
from .generator import generate_answer
from .retrieval import Retriever


@dataclass
class ChatResult:
    answer: str
    passages: List[Dict]
    drift_metrics: Dict


class LegalChatbotPipeline:
    def __init__(self, window_size: int = 200) -> None:
        self.retriever = Retriever(settings.embedding_model, settings.vector_dir)
        self.retriever.load()
        # Rolling window of recent top-1 scores, compared against the baseline scores
        # recorded for the reference query set when the index was first built.
        self._recent_top1: deque[float] = deque(maxlen=window_size)

    def _retrieval_drift(self) -> Dict:
        baseline = self.retriever.baseline_stats.get("reference_top1_scores") or []
        if not baseline:
            return {"status": "no_baseline"}
        return score_window_drift(baseline, list(self._recent_top1), min_samples=settings.min_drift_window)

    def ask(self, question: str) -> ChatResult:
        passages = self.retriever.search(question, top_k=settings.top_k)
        top1 = max((p["score"] for p in passages), default=0.0)
        if passages:
            self._recent_top1.append(top1)

        drift_metrics = self._retrieval_drift()
        drift_metrics["top1_score"] = top1

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