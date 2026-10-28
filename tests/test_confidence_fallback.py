from collections import deque
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import legal_chatbot.pipeline as pipeline_module  # noqa: E402
from legal_chatbot.pipeline import LegalChatbotPipeline  # noqa: E402


class StubRetriever:
    """Returns fixed passages so the gate can be tested without an index."""

    def __init__(self, score):
        self.score = score
        self.baseline_stats = {}

    def search(self, question, top_k=4):
        if self.score is None:
            return []
        return [{
            "chunk_id": "doc::1",
            "source": "sample_ico_guidance.txt",
            "title": "Sample Ico Guidance",
            "content": "Organisations should only collect the personal data they need.",
            "score": self.score,
        }]


def _pipeline(score, threshold=0.35):
    p = LegalChatbotPipeline.__new__(LegalChatbotPipeline)  # skip loading a real index
    p.retriever = StubRetriever(score)
    p.min_confidence = threshold
    p.provider = "openai"
    p._recent_top1 = deque(maxlen=200)
    return p


@pytest.fixture
def llm_calls(monkeypatch):
    calls = []

    def fake_generate_answer(**kwargs):
        calls.append(kwargs)
        return "LLM ANSWER"

    monkeypatch.setattr(pipeline_module, "generate_answer", fake_generate_answer)
    return calls


def test_weak_match_skips_llm_and_returns_sourced_extracts(llm_calls):
    result = _pipeline(score=0.12).ask("Who won the football?")
    assert result.low_confidence is True
    assert llm_calls == []
    assert "Low retrieval confidence" in result.answer
    assert "sample_ico_guidance.txt" in result.answer
    assert result.passages  # sources are still shown


def test_strong_match_calls_llm(llm_calls):
    result = _pipeline(score=0.62).ask("What is data minimisation?")
    assert result.low_confidence is False
    assert result.answer == "LLM ANSWER"
    assert len(llm_calls) == 1


def test_score_equal_to_threshold_is_answered(llm_calls):
    result = _pipeline(score=0.35, threshold=0.35).ask("What is data minimisation?")
    assert result.low_confidence is False
    assert len(llm_calls) == 1


def test_no_passages_is_low_confidence(llm_calls):
    result = _pipeline(score=None).ask("Anything")
    assert result.low_confidence is True
    assert llm_calls == []
    assert "No relevant passages" in result.answer


def test_drift_metrics_report_threshold(llm_calls):
    result = _pipeline(score=0.5, threshold=0.4).ask("What is data minimisation?")
    assert result.drift_metrics["min_confidence"] == 0.4
    assert result.drift_metrics["top1_score"] == 0.5
