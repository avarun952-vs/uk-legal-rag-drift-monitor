from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_chatbot.drift import kl_divergence, population_stability_index  # noqa: E402


def test_drift_metrics_zero_when_same():
    values = [0.1, 0.2, 0.3, 0.4]
    assert population_stability_index(values, values) == 0.0
    assert kl_divergence(values, values) == 0.0


def test_drift_metrics_positive_when_shifted():
    baseline = [0.10, 0.11, 0.12, 0.13]
    shifted = [0.70, 0.75, 0.80, 0.82]
    assert population_stability_index(baseline, shifted) > 0
    assert kl_divergence(baseline, shifted) > 0
