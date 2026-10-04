"""Drift metrics for the legal RAG pipeline.

Two different things can drift, and they are measured separately:

* Corpus drift: document embeddings change after the knowledge base is updated
  (new guidance published, legislation amended). Measured by comparing the current
  index against a saved baseline snapshot.
* Retrieval drift: top-1 similarity scores for incoming queries move away from
  the scores a fixed reference query set achieved at baseline time.

PSI is only meaningful on a reasonable number of samples, so every function that
returns a status also reports how many samples it used and whether that is enough.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

EPS = 1e-6
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
MIN_WINDOW = 20
MIN_RELIABLE_SAMPLES = 30


def _safe_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("Expected at least one value for drift calculation.")
    return arr


def _proportions(expected: np.ndarray, actual: np.ndarray, bins: int):
    """Bin both samples on quantile edges of the *baseline* (expected) sample.

    Outer bins are open-ended, so values outside the baseline range are counted
    instead of silently dropped.
    """
    inner = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))

    def props(values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(inner, values, side="right")
        counts = np.bincount(idx, minlength=inner.size + 1).astype(float)
        return np.clip(counts / counts.sum(), EPS, None)

    return props(expected), props(actual)


def population_stability_index(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    e, a = _proportions(_safe_array(expected), _safe_array(actual), bins)
    return float(np.sum((a - e) * np.log(a / e)))


def kl_divergence(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    """KL(actual || expected) over binned probability mass (always >= 0)."""
    e, a = _proportions(_safe_array(expected), _safe_array(actual), bins)
    return float(max(0.0, np.sum(a * np.log(a / e))))


def drift_status(psi: float) -> str:
    """Conventional PSI bands: <0.10 stable, 0.10-0.25 moderate, >0.25 significant."""
    if psi < PSI_MODERATE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def _unit_rows(matrix) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] == 0:
        raise ValueError("Expected a non-empty 2-D array of vectors.")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _centroid(matrix) -> np.ndarray:
    c = _unit_rows(matrix).mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n else c


def centroid_shift(baseline_vectors, new_vectors) -> float:
    """Cosine distance between the baseline and current corpus centroids."""
    base, new = _unit_rows(baseline_vectors), _unit_rows(new_vectors)
    if base.shape[1] != new.shape[1]:
        raise ValueError("Embedding dimensions differ between baseline and current index.")
    return float(max(0.0, 1.0 - float(_centroid(base) @ _centroid(new))))


def corpus_drift(baseline_vectors, new_vectors, bins: int = 5) -> Dict[str, object]:
    """Compare the current corpus embeddings with the baseline snapshot.

    Each chunk is reduced to its cosine similarity to the baseline centroid; PSI
    and KL are computed between the baseline and current distributions of that
    value. The centroid shift is reported alongside as an interpretable number.
    """
    base, new = _unit_rows(baseline_vectors), _unit_rows(new_vectors)
    if base.shape[1] != new.shape[1]:
        raise ValueError("Embedding dimensions differ between baseline and current index.")
    centre = _centroid(base)
    base_feat, new_feat = base @ centre, new @ centre
    psi = population_stability_index(base_feat, new_feat, bins=bins)
    return {
        "psi": psi,
        "kl_divergence": kl_divergence(base_feat, new_feat, bins=bins),
        "centroid_shift": centroid_shift(base, new),
        "status": drift_status(psi),
        "n_baseline": int(base.shape[0]),
        "n_new": int(new.shape[0]),
        "low_sample": min(base.shape[0], new.shape[0]) < MIN_RELIABLE_SAMPLES,
    }


def score_window_drift(
    baseline_scores: Sequence[float],
    recent_scores: Sequence[float],
    min_samples: int = MIN_WINDOW,
    bins: int = 5,
) -> Dict[str, object]:
    """PSI/KL of recent top-1 retrieval scores against the baseline scores.

    Returns status "insufficient_data" until `min_samples` scores are available,
    because PSI on a handful of queries is noise, not a drift signal.
    """
    n = len(recent_scores)
    if n < min_samples:
        return {"status": "insufficient_data", "n": n, "needed": min_samples}
    psi = population_stability_index(baseline_scores, recent_scores, bins=bins)
    return {
        "status": drift_status(psi),
        "psi": psi,
        "kl_divergence": kl_divergence(baseline_scores, recent_scores, bins=bins),
        "n": n,
        "low_sample": n < MIN_RELIABLE_SAMPLES or len(baseline_scores) < MIN_RELIABLE_SAMPLES,
    }