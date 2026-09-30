"""Cấu hình tập trung, đọc từ biến môi trường / file .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

_EMBED_PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
}


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    ollama_host: str = "http://127.0.0.1:11434"
    allow_remote_ollama: bool = False

    llm_model: str = "qwen3:8b"
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192
    llm_think: bool = False

    embed_model: str = "bge-m3"
    embed_query_prefix: str = ""
    embed_doc_prefix: str = ""

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    candidate_k: int = 30
    min_score: float = 0.30
    vector_weight: float = 0.80
    lexical_weight: float = 0.20
    max_chunks_per_source: int = 3

    parser: str = "pypdf"
    rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidate_factor: int = 3

    # Multimodal ingestion: ảnh rời được chuyển thành văn bản mô tả/OCR bằng vision model.
    vision_model: str = ""
    vision_enabled: bool = False

    raw_dir: Path = Path("data/raw")
    db_dir: Path = Path("data/index")
    judge_model: str = ""

    @property
    def judge(self) -> str:
        return self.judge_model or self.llm_model

    def __post_init__(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("RAG_CHUNK_SIZE phải >= 100")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP phải nằm trong [0, chunk_size)")
        if self.top_k < 1 or self.candidate_k < self.top_k:
            raise ValueError("RAG_CANDIDATE_K phải >= RAG_TOP_K >= 1")
        if self.parser not in {"pypdf", "docling"}:
            raise ValueError("PARSER phải là 'pypdf' hoặc 'docling'")
        if self.vector_weight < 0 or self.lexical_weight < 0:
            raise ValueError("VECTOR_WEIGHT và LEXICAL_WEIGHT phải >= 0")
        if self.vector_weight + self.lexical_weight <= 0:
            raise ValueError("Tổng VECTOR_WEIGHT + LEXICAL_WEIGHT phải > 0")
        if self.max_chunks_per_source < 1:
            raise ValueError("RAG_MAX_CHUNKS_PER_SOURCE phải >= 1")


def load_settings() -> Settings:
    embed_model = os.getenv("EMBED_MODEL", "bge-m3")
    default_q, default_d = _EMBED_PREFIXES.get(embed_model.split(":")[0], ("", ""))
    return Settings(
        ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        allow_remote_ollama=_bool("ALLOW_REMOTE_OLLAMA", False),
        llm_model=os.getenv("LLM_MODEL", "qwen3:8b"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        llm_num_ctx=int(os.getenv("LLM_NUM_CTX", "8192")),
        llm_think=_bool("LLM_THINK", False),
        embed_model=embed_model,
        embed_query_prefix=os.getenv("EMBED_QUERY_PREFIX") or default_q,
        embed_doc_prefix=os.getenv("EMBED_DOC_PREFIX") or default_d,
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        candidate_k=int(os.getenv("RAG_CANDIDATE_K", "30")),
        min_score=float(os.getenv("RAG_MIN_SCORE", "0.30")),
        vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.80")),
        lexical_weight=float(os.getenv("LEXICAL_WEIGHT", "0.20")),
        max_chunks_per_source=int(os.getenv("RAG_MAX_CHUNKS_PER_SOURCE", "3")),
        parser=os.getenv("PARSER", "pypdf").strip().lower(),
        rerank=_bool("RERANK", False),
        rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_candidate_factor=int(os.getenv("RERANK_CANDIDATE_FACTOR", "3")),
        vision_model=os.getenv("VISION_MODEL", "").strip(),
        vision_enabled=_bool("VISION_ENABLED", False),
        raw_dir=Path(os.getenv("RAG_RAW_DIR", "data/raw")),
        db_dir=Path(os.getenv("RAG_DB_DIR", "data/index")),
        judge_model=os.getenv("JUDGE_MODEL", "").strip(),
    )
