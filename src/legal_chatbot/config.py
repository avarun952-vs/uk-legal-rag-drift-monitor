from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "extractive").strip().lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1").strip()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    source_dir: Path = Path(os.getenv("SOURCE_DIR", "data/source_documents"))
    vector_dir: Path = Path(os.getenv("VECTOR_DIR", "data/vector_store"))
    reference_queries: Path = Path(os.getenv("REFERENCE_QUERIES", "data/reference_queries.txt"))
    min_drift_window: int = int(os.getenv("MIN_DRIFT_WINDOW", "20"))
    # Below this top-1 cosine score the pipeline does not ask the LLM for an answer and
    # returns the closest sourced extracts with a warning instead. 0.35 was chosen from
    # the reference queries (in-domain top-1 scores sit roughly between 0.36 and 0.72
    # with sentence-transformer embeddings). Re-check it if the corpus or model changes.
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.35"))


settings = Settings()
