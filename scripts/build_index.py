from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_chatbot.config import settings  # noqa: E402
from legal_chatbot.ingest import create_chunks, load_reference_queries  # noqa: E402
from legal_chatbot.retrieval import Retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the vector index.")
    parser.add_argument(
        "--reset-baseline",
        action="store_true",
        help="Overwrite the drift baseline with this build. Without it, the baseline from the "
        "first build is kept so later rebuilds can be compared against it.",
    )
    args = parser.parse_args()

    chunks = create_chunks(
        source_dir=settings.source_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    retriever = Retriever(settings.embedding_model, settings.vector_dir)
    retriever.build(
        chunks,
        reference_queries=load_reference_queries(settings.reference_queries),
        reset_baseline=args.reset_baseline,
    )
    print(f"Built vector index with {len(chunks)} chunks at {settings.vector_dir}")
    print("Baseline reset." if args.reset_baseline else "Baseline kept (created on first build).")


if __name__ == "__main__":
    main()