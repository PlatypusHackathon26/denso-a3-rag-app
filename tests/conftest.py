"""Fake embedder/LLM để test toàn bộ pipeline mà không cần Ollama hay mạng."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest

from localrag.config import Settings

DIM = 256


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _vec(text: str) -> list[float]:
    v = [0.0] * DIM
    for tok in _tokens(text):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeEmbedder:
    """Bag-of-words băm: đủ để câu hỏi trùng từ khoá xếp hạng đúng đoạn."""

    model = "fake-embed"

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [_vec(t) for t in texts]

    def embed_query(self, text):
        return _vec(text)


class FakeLLM:
    def __init__(self, reply="OK [1]"):
        self.reply = reply
        self.prompts: list[tuple[str, str]] = []

    def generate(self, system, user):
        self.prompts.append((system, user))
        return self.reply

    def stream(self, system, user):
        self.prompts.append((system, user))
        yield from self.reply.split(" ")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    raw = tmp_path / "raw"
    raw.mkdir()
    return Settings(
        embed_model="fake-embed",
        raw_dir=raw,
        db_dir=tmp_path / "index",
        chunk_size=300,
        chunk_overlap=40,
        top_k=3,
        min_score=0.35,
    )


@pytest.fixture
def make_pipeline(settings):
    from localrag.pipeline import RAGPipeline

    def _make(llm=None, **kw):
        return RAGPipeline(
            settings, embedder=kw.pop("embedder", FakeEmbedder()), llm=llm or FakeLLM(), **kw
        )

    return _make
