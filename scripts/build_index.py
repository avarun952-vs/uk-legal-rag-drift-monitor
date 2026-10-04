from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_chatbot.config import settings  # noqa: E402
from legal_chatbot.ingest import create_chunks  # noqa: E402
from legal_chatbot.retrieval import Retriever  # noqa: E402


def main() -> None:
    chunks = create_chunks(
        source_dir=settings.source_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    retriever = Retriever(settings.embedding_model, settings.vector_dir)
    retriever.build(chunks)
    print(f"Built vector index with {len(chunks)} chunks at {settings.vector_dir}")


if __name__ == "__main__":
    main()
