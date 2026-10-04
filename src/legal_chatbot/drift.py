from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _safe_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("Expected at least one value for drift calculation.")
    return arr


def population_stability_index(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    expected_arr = _safe_array(expected)
    actual_arr = _safe_array(actual)

    min_v = min(expected_arr.min(), actual_arr.min())
    max_v = max(expected_arr.max(), actual_arr.max())

    if math.isclose(min_v, max_v):
        return 0.0

    breakpoints = np.linspace(min_v, max_v, bins + 1)
    expected_hist, _ = np.histogram(expected_arr, bins=breakpoints)
    actual_hist, _ = np.histogram(actual_arr, bins=breakpoints)

    expected_pct = np.where(expected_hist == 0, 1e-6, expected_hist / expected_hist.sum())
    actual_pct = np.where(actual_hist == 0, 1e-6, actual_hist / actual_hist.sum())

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def kl_divergence(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    expected_arr = _safe_array(expected)
    actual_arr = _safe_array(actual)

    min_v = min(expected_arr.min(), actual_arr.min())
    max_v = max(expected_arr.max(), actual_arr.max())

    if math.isclose(min_v, max_v):
        return 0.0

    breakpoints = np.linspace(min_v, max_v, bins + 1)
    expected_hist, _ = np.histogram(expected_arr, bins=breakpoints, density=True)
    actual_hist, _ = np.histogram(actual_arr, bins=breakpoints, density=True)

    expected_hist = np.where(expected_hist == 0, 1e-6, expected_hist)
    actual_hist = np.where(actual_hist == 0, 1e-6, actual_hist)

    return float(np.sum(actual_hist * np.log(actual_hist / expected_hist)))
