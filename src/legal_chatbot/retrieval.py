from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk
from .utils import ensure_dir, load_json, save_json

try:  # FAISS is the primary search backend; NumPy is kept as a fallback.
    import faiss  # type: ignore

    HAS_FAISS = True
except ImportError:  # pragma: no cover - depends on the environment
    faiss = None
    HAS_FAISS = False


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class Retriever:
    """Dense retrieval over a local corpus.

    Embeddings:
        sentence-transformers (preferred) or TF-IDF when the model cannot be
        downloaded, for example on a restricted machine.

    Search:
        FAISS inner-product index over L2-normalised vectors, which is the same as
        cosine similarity. If FAISS is not installed, the same ranking is done with
        NumPy so the project still runs.
    """

    def __init__(self, model_name: str, vector_dir: Path) -> None:
        self.model_name = model_name
        self.vector_dir = ensure_dir(vector_dir)
        self.matrix_path = self.vector_dir / "vectors.npy"
        self.meta_path = self.vector_dir / "metadata.pkl"
        self.baseline_path = self.vector_dir / "baseline_stats.json"
        self.vectorizer_path = self.vector_dir / "vectorizer.pkl"
        self.mode_path = self.vector_dir / "mode.json"
        self.baseline_vectors_path = self.vector_dir / "baseline_vectors.npy"
        self.index_path = self.vector_dir / "index.faiss"

        self._metadata: List[Dict[str, Any]] = []
        self._vectors: np.ndarray | None = None
        self._index = None
        self._encoder = None
        self._mode = "unknown"

    # ------------------------------------------------------------------ encoding
    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder

        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
            self._mode = "sentence_transformer"
            return self._encoder
        except Exception:
            self._encoder = TfidfVectorizer(stop_words="english")
            self._mode = "tfidf"
            return self._encoder

    def _encode_query(self, query: str) -> np.ndarray:
        encoder = self._encoder
        if encoder is None:
            if self._mode == "sentence_transformer":
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(self.model_name)
                self._encoder = encoder
            else:
                with self.vectorizer_path.open("rb") as f:
                    encoder = pickle.load(f)
                self._encoder = encoder

        if self._mode == "sentence_transformer":
            vector = encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)
        else:
            vector = encoder.transform([query]).toarray()
        return _l2_normalise(vector)

    # --------------------------------------------------------------------- index
    def _build_index(self, vectors: np.ndarray) -> None:
        """Store normalised vectors and, when available, a FAISS index over them."""
        self._vectors = _l2_normalise(vectors)
        if HAS_FAISS and self._vectors.size:
            index = faiss.IndexFlatIP(self._vectors.shape[1])
            index.add(self._vectors)
            self._index = index
        else:
            self._index = None

    def build(
        self,
        chunks: List[Chunk],
        reference_queries: List[str] | None = None,
        reset_baseline: bool = False,
    ) -> None:
        texts = [c.content for c in chunks]
        encoder = self._get_encoder()

        if self._mode == "sentence_transformer":
            vectors = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            vectors = encoder.fit_transform(texts).toarray()
            with self.vectorizer_path.open("wb") as f:
                pickle.dump(encoder, f)

        self._build_index(np.asarray(vectors, dtype="float32"))
        self._metadata = [asdict(c) for c in chunks]

        np.save(self.matrix_path, self._vectors)
        with self.meta_path.open("wb") as f:
            pickle.dump(self._metadata, f)
        save_json(self.mode_path, {"mode": self._mode})
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))

        # The baseline is a frozen snapshot. It is written on the first build (or when
        # explicitly reset) and left alone on later rebuilds, so drift is always
        # measured against the same reference rather than against the last build.
        if reset_baseline or not self.baseline_path.exists() or not self.baseline_vectors_path.exists():
            np.save(self.baseline_vectors_path, self._vectors)
            save_json(
                self.baseline_path,
                {
                    "document_count": len(chunks),
                    "retrieval_mode": self._mode,
                    "reference_top1_scores": self.top1_scores(reference_queries or []),
                },
            )

    def load(self) -> None:
        if not self.matrix_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Vector index not found. Run scripts/build_index.py first.")

        vectors = np.load(self.matrix_path)
        with self.meta_path.open("rb") as f:
            self._metadata = pickle.load(f)

        self._mode = (load_json(self.mode_path, default={}) or {}).get("mode", "tfidf")

        if HAS_FAISS and self.index_path.exists():
            self._vectors = _l2_normalise(vectors)
            self._index = faiss.read_index(str(self.index_path))
        else:
            # Older index folders have no index.faiss; build the index in memory.
            self._build_index(vectors)

        if self._mode == "tfidf":
            if not self.vectorizer_path.exists():
                raise FileNotFoundError("TF-IDF vectorizer file is missing.")
            with self.vectorizer_path.open("rb") as f:
                self._encoder = pickle.load(f)
        else:
            self._encoder = None

    # -------------------------------------------------------------------- search
    def _rank(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) of the top_k chunks for a normalised query vector."""
        top_k = max(1, min(top_k, len(self._metadata)))
        query_vector = _l2_normalise(query_vector.reshape(1, -1))
        if self._index is not None:
            scores, indices = self._index.search(query_vector, top_k)
            return scores[0], indices[0]
        scores = (self._vectors @ query_vector.T).reshape(-1)
        indices = np.argsort(scores)[::-1][:top_k]
        return scores[indices], indices

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self._vectors is None:
            self.load()

        scores, indices = self._rank(self._encode_query(query), top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores, indices):
            if idx < 0:  # FAISS pads with -1 when there are fewer vectors than top_k
                continue
            item = dict(self._metadata[int(idx)])
            item["score"] = float(score)
            results.append(item)
        return results

    def top1_scores(self, queries: List[str]) -> List[float]:
        """Top-1 cosine score for each query (used to build and check the baseline)."""
        return [self.search(q, top_k=1)[0]["score"] for q in queries]

    # ---------------------------------------------------------------- properties
    @property
    def baseline_stats(self) -> Dict[str, Any]:
        return load_json(self.baseline_path, default={}) or {}

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def search_backend(self) -> str:
        return "faiss" if self._index is not None else "numpy"

    @property
    def chunk_count(self) -> int:
        return len(self._metadata)

    @property
    def vectors(self) -> np.ndarray | None:
        return self._vectors

    @property
    def baseline_vectors(self) -> np.ndarray | None:
        if not self.baseline_vectors_path.exists():
            return None
        return np.load(self.baseline_vectors_path)
