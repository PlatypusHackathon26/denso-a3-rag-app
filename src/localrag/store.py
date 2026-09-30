"""STORE + RETRIEVE: LanceDB cục bộ (cosine) và siêu dữ liệu chỉ mục."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lancedb
import pyarrow as pa

from .models import Chunk, SearchResult

TABLE_NAME = "chunks"
META_FILE = "index_meta.json"


@dataclass
class IndexMeta:
    signature: dict = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, db_dir: Path) -> "IndexMeta | None":
        path = db_dir / META_FILE
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(signature=data.get("signature", {}), files=data.get("files", {}))

    def save(self, db_dir: Path) -> None:
        db_dir.mkdir(parents=True, exist_ok=True)
        tmp = db_dir / (META_FILE + ".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(db_dir / META_FILE)


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class VectorStore:
    def __init__(self, db_dir: Path):
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(db_dir))
        self._table = self._open()

    def _open(self):
        try:
            return self._db.open_table(TABLE_NAME)
        except ValueError:
            return None

    def count(self) -> int:
        return self._table.count_rows() if self._table is not None else 0

    def reset(self) -> None:
        if self._table is not None:
            self._db.drop_table(TABLE_NAME)
            self._table = None

    def _create(self, dim: int):
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("text", pa.string()),
                pa.field("source", pa.string()),
                pa.field("heading", pa.string()),
                pa.field("page", pa.int32()),
                pa.field("chunk_index", pa.int32()),
                pa.field("kind", pa.string()),
            ]
        )
        self._table = self._db.create_table(TABLE_NAME, schema=schema)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Số chunk và số vector không khớp")
        if not chunks:
            return
        if self._table is None:
            self._create(len(vectors[0]))
        self._table.add(
            [
                {
                    "id": c.id,
                    "vector": v,
                    "text": c.text,
                    "source": c.source,
                    "heading": c.heading,
                    "page": c.page,
                    "chunk_index": c.index,
                    "kind": c.kind,
                }
                for c, v in zip(chunks, vectors)
            ]
        )

    def delete_source(self, source: str) -> None:
        if self._table is not None:
            self._table.delete(f"source = {_sql_quote(source)}")

    def search(self, vector: list[float], limit: int) -> list[SearchResult]:
        if self._table is None or limit < 1:
            return []
        rows = self._table.search(vector).metric("cosine").limit(limit).to_list()
        return [
            SearchResult(
                chunk_id=r.get("id", ""),
                text=r["text"],
                source=r["source"],
                score=1.0 - float(r["_distance"]),
                vector_score=1.0 - float(r["_distance"]),
                heading=r.get("heading") or "",
                page=r.get("page"),
                kind=r.get("kind") or "text",
            )
            for r in rows
        ]
