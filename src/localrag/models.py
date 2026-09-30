"""Các kiểu dữ liệu dùng chung giữa các module."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    """Một đoạn liền mạch của tài liệu."""

    text: str
    heading: str = ""
    page: int | None = None
    kind: str = "text"


@dataclass
class Document:
    source: str
    file_hash: str
    sections: list[Section] = field(default_factory=list)


@dataclass
class Chunk:
    source: str
    index: int
    text: str
    heading: str = ""
    page: int | None = None
    kind: str = "text"

    @property
    def id(self) -> str:
        """ID ổn định: cùng nguồn + vị trí + nội dung -> cùng ID."""
        raw = f"{self.source}\x00{self.index}\x00{self.text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    @property
    def embed_text(self) -> str:
        """Văn bản đem đi embed: gắn heading để giữ ngữ cảnh mục."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


@dataclass
class SearchResult:
    text: str
    source: str
    score: float
    heading: str = ""
    page: int | None = None
    rerank_score: float | None = None
    vector_score: float | None = None
    lexical_score: float | None = None
    chunk_id: str = ""
    kind: str = "text"

    def __post_init__(self) -> None:
        if self.vector_score is None:
            self.vector_score = self.score

    @property
    def label(self) -> str:
        page = f" p.{self.page}" if self.page else ""
        return f"{self.source}{page}"


@dataclass
class RAGAnswer:
    answer: str
    sources: list[SearchResult]
    grounded: bool
    latency_ms: float | None = None
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    citations: list[int] = field(default_factory=list)
    citation_valid: bool = True
