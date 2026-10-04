from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_chatbot.drift import (  # noqa: E402
    centroid_shift,
    corpus_drift,
    drift_status,
    kl_divergence,
    population_stability_index,
    score_window_drift,
)


def _cluster(direction, n, rng, noise=0.4):
    vectors = direction + noise * rng.standard_normal((n, direction.size))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _directions(dim=64):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(dim)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(dim)
    b -= (b @ a) * a  # make b orthogonal to a
    b /= np.linalg.norm(b)
    return a, b


def test_drift_metrics_zero_when_same():
    values = [0.1, 0.2, 0.3, 0.4]
    assert population_stability_index(values, values) == 0.0
    assert kl_divergence(values, values) == 0.0


def test_drift_metrics_positive_when_shifted():
    baseline = [0.10, 0.11, 0.12, 0.13]
    shifted = [0.70, 0.75, 0.80, 0.82]
    assert population_stability_index(baseline, shifted) > 0
    assert kl_divergence(baseline, shifted) > 0


def test_psi_small_for_same_distribution():
    rng = np.random.default_rng(1)
    baseline = rng.normal(0.55, 0.05, 500)
    fresh = rng.normal(0.55, 0.05, 500)
    assert population_stability_index(baseline, fresh, bins=5) < 0.10


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(2)
    baseline = rng.normal(0.55, 0.05, 500)
    shifted = rng.normal(0.35, 0.05, 500)
    assert population_stability_index(baseline, shifted, bins=5) > 0.25


def test_values_outside_baseline_range_are_counted():
    baseline = np.linspace(0.4, 0.6, 50)
    far_outside = np.full(50, 5.0)
    assert population_stability_index(baseline, far_outside, bins=5) > 0.25


def test_kl_is_never_negative():
    rng = np.random.default_rng(3)
    for _ in range(20):
        a, b = rng.random(40), rng.random(40)
        assert kl_divergence(a, b, bins=5) >= 0.0


def test_drift_status_bands():
    assert drift_status(0.05) == "stable"
    assert drift_status(0.15) == "moderate"
    assert drift_status(0.40) == "significant"


def test_corpus_drift_identical_corpus_is_exactly_zero():
    a, _ = _directions()
    vectors = _cluster(a, 200, np.random.default_rng(4))
    result = corpus_drift(vectors, vectors.copy())
    assert result["psi"] == 0.0
    assert result["status"] == "stable"
    assert result["centroid_shift"] < 1e-9


def test_corpus_drift_resampled_same_topic_stays_stable():
    a, _ = _directions()
    base = _cluster(a, 300, np.random.default_rng(5))
    fresh = _cluster(a, 300, np.random.default_rng(6))
    result = corpus_drift(base, fresh)
    assert result["status"] == "stable"
    assert result["centroid_shift"] < 0.05


def test_corpus_drift_detects_topic_change():
    a, b = _directions()
    base = _cluster(a, 300, np.random.default_rng(7))
    changed = _cluster(b, 300, np.random.default_rng(8))
    result = corpus_drift(base, changed)
    assert result["status"] == "significant"
    assert result["centroid_shift"] > 0.3


def test_corpus_drift_partial_update_is_detected_but_smaller_than_full_change():
    a, b = _directions()
    rng = np.random.default_rng(9)
    base = _cluster(a, 300, rng)
    partial = np.vstack([_cluster(a, 200, rng), _cluster(b, 100, rng)])
    full = _cluster(b, 300, rng)
    assert corpus_drift(base, partial)["psi"] < corpus_drift(base, full)["psi"]
    assert corpus_drift(base, partial)["psi"] > 0.10


def test_corpus_drift_flags_small_samples():
    a, _ = _directions()
    small = _cluster(a, 10, np.random.default_rng(10))
    assert corpus_drift(small, small.copy())["low_sample"] is True


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        centroid_shift(np.ones((5, 8)), np.ones((5, 9)))
    with pytest.raises(ValueError):
        corpus_drift(np.ones((5, 8)), np.ones((5, 9)))


def test_window_drift_refuses_to_report_on_too_few_queries():
    result = score_window_drift([0.5] * 30, [0.1, 0.2, 0.3], min_samples=20)
    assert result["status"] == "insufficient_data"
    assert result["n"] == 3 and result["needed"] == 20
    assert "psi" not in result


def test_window_drift_stable_and_significant():
    rng = np.random.default_rng(11)
    baseline = rng.normal(0.55, 0.05, 200)
    same = rng.normal(0.55, 0.05, 100)
    worse = rng.normal(0.30, 0.05, 100)
    assert score_window_drift(baseline, same)["status"] == "stable"
    assert score_window_drift(baseline, worse)["status"] == "significant"