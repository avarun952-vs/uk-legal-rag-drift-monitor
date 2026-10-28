from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List

from .config import settings
from .drift import score_window_drift
from .generator import generate_answer, low_confidence_answer
from .retrieval import Retriever


@dataclass
class ChatResult:
    answer: str
    passages: List[Dict]
    drift_metrics: Dict
    low_confidence: bool = False


class LegalChatbotPipeline:
    def __init__(
        self,
        window_size: int = 200,
        min_confidence: float | None = None,
        provider: str | None = None,
    ) -> None:
        self.retriever = Retriever(settings.embedding_model, settings.vector_dir)
        self.retriever.load()
        self.min_confidence = settings.min_confidence if min_confidence is None else min_confidence
        self.provider = settings.llm_provider if provider is None else provider
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
        drift_metrics["min_confidence"] = self.min_confidence

        # Confidence gate: if even the best passage is a weak match, do not ask the
        # language model to write an answer. Show the closest sourced extracts and say
        # clearly that the match is weak, so the user checks the sources themselves.
        if not passages or top1 < self.min_confidence:
            return ChatResult(
                answer=low_confidence_answer(question, passages, top1, self.min_confidence),
                passages=passages,
                drift_metrics=drift_metrics,
                low_confidence=True,
            )

        answer = generate_answer(
            provider=self.provider,
            question=question,
            passages=passages,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            openai_base_url=settings.openai_base_url,
            ollama_model=settings.ollama_model,
            ollama_base_url=settings.ollama_base_url,
        )
        return ChatResult(answer=answer, passages=passages, drift_metrics=drift_metrics)
