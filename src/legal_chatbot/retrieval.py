from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk
from .utils import ensure_dir, load_json, save_json


class Retriever:
    """Dense retrieval with graceful fallback.

    Preferred mode:
        sentence-transformers embeddings + cosine similarity

    Fallback mode:
        TF-IDF vectorisation + cosine similarity

    This design keeps the project runnable even when a transformer model
    cannot be downloaded in restricted environments.
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

        self._metadata: List[Dict[str, Any]] = []
        self._vectors: np.ndarray | None = None
        self._encoder = None
        self._mode = "unknown"

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
            vectors = np.asarray(vectors, dtype="float32")
        else:
            vectors = encoder.fit_transform(texts).toarray().astype("float32")
            with self.vectorizer_path.open("wb") as f:
                pickle.dump(encoder, f)

        np.save(self.matrix_path, vectors)
        with self.meta_path.open("wb") as f:
            pickle.dump([asdict(c) for c in chunks], f)
        save_json(self.mode_path, {"mode": self._mode})

        self._vectors = vectors
        self._metadata = [asdict(c) for c in chunks]

        # The baseline is a frozen snapshot. It is written on the first build (or when
        # explicitly reset) and left alone on later rebuilds, so drift is always
        # measured against the same reference rather than against the last build.
        if reset_baseline or not self.baseline_path.exists() or not self.baseline_vectors_path.exists():
            np.save(self.baseline_vectors_path, vectors)
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

        self._vectors = np.load(self.matrix_path)
        with self.meta_path.open("rb") as f:
            self._metadata = pickle.load(f)

        self._mode = (load_json(self.mode_path, default={}) or {}).get("mode", "tfidf")

        if self._mode == "tfidf":
            if not self.vectorizer_path.exists():
                raise FileNotFoundError("TF-IDF vectorizer file is missing.")
            with self.vectorizer_path.open("rb") as f:
                self._encoder = pickle.load(f)
        else:
            self._encoder = None

    @property
    def baseline_stats(self) -> Dict[str, Any]:
        return load_json(self.baseline_path, default={}) or {}

    def _encode_query(self, query: str) -> np.ndarray:
        encoder = self._encoder
        if encoder is None:
            if self._mode == "sentence_transformer":
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(self.model_name)
            else:
                with self.vectorizer_path.open("rb") as f:
                    encoder = pickle.load(f)
        
        if hasattr(encoder, 'encode') and 'SentenceTransformer' in str(type(encoder)):
            vector = encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)
        else:
            # TF-IDF fallback
            vector = encoder.transform([query]).toarray().astype("float32")
            # l2 normalize
            denom = np.linalg.norm(vector, axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            vector = vector / denom
        
        return np.asarray(vector, dtype="float32")

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self._vectors is None:
            self.load()

        query_vector = self._encode_query(query)
        matrix = self._vectors
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        matrix_norm = matrix / row_norms

        scores = matrix_norm @ query_vector.T
        scores = scores.reshape(-1)

        top_k = max(1, min(top_k, len(scores)))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            item = dict(self._metadata[int(idx)])
            item["score"] = float(scores[int(idx)])
            results.append(item)
        return results

    def top1_scores(self, queries: List[str]) -> List[float]:
        """Top-1 cosine score for each query (used to build and check the baseline)."""
        return [self.search(q, top_k=1)[0]["score"] for q in queries]

    @property
    def mode(self) -> str:
        return self._mode

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
