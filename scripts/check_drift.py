"""Compare the current index with the saved baseline and report drift.

Usage:  python scripts/check_drift.py
Exit code: 0 = stable/moderate, 1 = significant drift, 2 = no baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_chatbot.config import settings  # noqa: E402
from legal_chatbot.drift import corpus_drift, score_window_drift  # noqa: E402
from legal_chatbot.ingest import load_reference_queries  # noqa: E402
from legal_chatbot.retrieval import Retriever  # noqa: E402


def main() -> int:
    retriever = Retriever(settings.embedding_model, settings.vector_dir)
    retriever.load()
    stats = retriever.baseline_stats
    baseline_vectors = retriever.baseline_vectors
    if not stats or baseline_vectors is None:
        print("No baseline found. Run: python scripts/build_index.py --reset-baseline")
        return 2

    significant = False
    print(f"Index mode: {retriever.mode} (baseline: {stats.get('retrieval_mode')})")
    print(f"Chunks: baseline={stats.get('document_count')}, current={retriever.chunk_count}\n")

    print("1. Corpus drift (embeddings vs baseline snapshot)")
    if retriever.mode == "sentence_transformer" and stats.get("retrieval_mode") == "sentence_transformer":
        res = corpus_drift(baseline_vectors, retriever.vectors)
        significant |= res["status"] == "significant"
        print(f"   status={res['status']}  PSI={res['psi']:.3f}  KL={res['kl_divergence']:.3f}  "
              f"centroid_shift={res['centroid_shift']:.3f}")
        if res["low_sample"]:
            print(f"   note: only {res['n_baseline']} baseline / {res['n_new']} current chunks; "
                  "treat as indicative, not conclusive.")
    else:
        print("   skipped: needs sentence-transformer embeddings in both baseline and current index "
              "(TF-IDF vector dimensions change with the vocabulary).")

    print("\n2. Retrieval drift (reference queries, top-1 score)")
    base_scores = stats.get("reference_top1_scores") or []
    queries = load_reference_queries(settings.reference_queries)
    if not base_scores or not queries:
        print("   skipped: no reference queries recorded in the baseline.")
    else:
        current = retriever.top1_scores(queries)
        res = score_window_drift(base_scores, current, min_samples=1)
        significant |= res["status"] == "significant"
        print(f"   status={res['status']}  PSI={res['psi']:.3f}  KL={res['kl_divergence']:.3f}  "
              f"mean top-1: baseline={sum(base_scores)/len(base_scores):.3f} "
              f"current={sum(current)/len(current):.3f}")
        if res["low_sample"]:
            print(f"   note: {len(current)} reference queries is a small sample; treat as indicative.")

    print("\nOverall:", "SIGNIFICANT drift - review the corpus change." if significant else "no significant drift.")
    return 1 if significant else 0


if __name__ == "__main__":
    raise SystemExit(main())