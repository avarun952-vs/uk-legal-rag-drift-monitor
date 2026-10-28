from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_chatbot.ingest import Chunk  # noqa: E402
from legal_chatbot.retrieval import Retriever  # noqa: E402


def _force_tfidf(retriever: Retriever) -> None:
    """Use TF-IDF so the tests run offline and never download an embedding model."""

    def encoder():
        if retriever._encoder is None:
            retriever._encoder = TfidfVectorizer(stop_words="english")
            retriever._mode = "tfidf"
        return retriever._encoder

    retriever._get_encoder = encoder


CHUNKS = [
    Chunk("gdpr::1", "gdpr.txt", "Gdpr", "Personal data must be processed lawfully, fairly and transparently."),
    Chunk("gdpr::2", "gdpr.txt", "Gdpr", "Data minimisation means collecting only the personal data you need."),
    Chunk("fca::1", "fca.txt", "Fca", "The Consumer Duty requires firms to deliver good outcomes for retail customers."),
    Chunk("fca::2", "fca.txt", "Fca", "Firms must show their products offer fair value to retail customers."),
]


def test_faiss_and_numpy_rank_identically(tmp_path):
    pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((50, 32)).astype("float32")

    retriever = Retriever("unused", tmp_path)
    retriever._metadata = [{"chunk_id": str(i)} for i in range(50)]
    retriever._build_index(vectors)
    assert retriever.search_backend == "faiss"

    query = rng.standard_normal(32).astype("float32")
    faiss_scores, faiss_idx = retriever._rank(query, 5)

    retriever._index = None  # same vectors, NumPy path
    assert retriever.search_backend == "numpy"
    numpy_scores, numpy_idx = retriever._rank(query, 5)

    assert list(faiss_idx) == list(numpy_idx)
    np.testing.assert_allclose(faiss_scores, numpy_scores, atol=1e-5)


def test_scores_are_cosine_similarities(tmp_path):
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((20, 16)).astype("float32") * 10  # not unit length
    retriever = Retriever("unused", tmp_path)
    retriever._metadata = [{"chunk_id": str(i)} for i in range(20)]
    retriever._build_index(vectors)

    scores, idx = retriever._rank(vectors[3], 1)
    assert idx[0] == 3
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_top_k_larger_than_corpus_is_safe(tmp_path):
    retriever = Retriever("unused", tmp_path)
    _force_tfidf(retriever)
    retriever.build(CHUNKS)
    results = retriever.search("fair value for customers", top_k=50)
    assert len(results) == len(CHUNKS)


def test_build_save_load_and_search(tmp_path):
    builder = Retriever("unused", tmp_path)
    _force_tfidf(builder)
    builder.build(CHUNKS, reference_queries=["What is data minimisation?"])
    if builder.search_backend == "faiss":
        assert (tmp_path / "index.faiss").exists()

    loaded = Retriever("unused", tmp_path)
    loaded.load()
    assert loaded.search_backend == builder.search_backend
    top = loaded.search("What does the Consumer Duty require?", top_k=1)[0]
    assert top["chunk_id"] == "fca::1"
    assert len(loaded.baseline_stats["reference_top1_scores"]) == 1


def test_load_without_index_file_rebuilds_in_memory(tmp_path):
    builder = Retriever("unused", tmp_path)
    _force_tfidf(builder)
    builder.build(CHUNKS)
    (tmp_path / "index.faiss").unlink(missing_ok=True)  # simulate an index made before FAISS

    loaded = Retriever("unused", tmp_path)
    loaded.load()
    assert loaded.search("data minimisation", top_k=1)[0]["chunk_id"] == "gdpr::2"
