from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Chunk:
    chunk_id: str
    source: str
    title: str
    content: str


def load_text_documents(source_dir: Path) -> List[tuple[str, str]]:
    documents: List[tuple[str, str]] = []
    for path in sorted(source_dir.glob("*.txt")):
        documents.append((path.name, path.read_text(encoding="utf-8")))
    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in {source_dir.resolve()}. "
            "Add source documents before building the index."
        )
    return documents


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        yield text[start:end].strip()
        if end == length:
            break
        start = max(0, end - chunk_overlap)


def create_chunks(source_dir: Path, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for filename, raw_text in load_text_documents(source_dir):
        title = filename.replace("_", " ").replace(".txt", "").title()
        for idx, piece in enumerate(split_text(raw_text, chunk_size, chunk_overlap), start=1):
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{filename}::chunk_{idx}",
                    source=filename,
                    title=title,
                    content=piece,
                )
            )
    return chunks
