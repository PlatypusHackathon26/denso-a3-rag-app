"""RAGPipeline: ingest + hybrid retrieval + grounded generation, chạy cục bộ."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import ollama

from .chunker import chunk_document
from .config import Settings, load_settings
from .loader import IMAGE_SUFFIXES, discover_files, file_sha256, load_document
from .models import Chunk, Document, RAGAnswer, SearchResult, Section
from .ollama_client import OllamaEmbedder, OllamaError, OllamaLLM, make_client
from .prompts import NO_ANSWER, SYSTEM_PROMPT, build_user_prompt, extract_citations
from .retrieval import hybrid_score, lexical_score
from .store import META_FILE, IndexMeta, VectorStore

Progress = Callable[[str], None]


class IndexMissingError(RuntimeError):
    pass


class IndexMismatchError(RuntimeError):
    pass


@dataclass
class IngestReport:
    rebuilt: bool = False
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped_empty: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks_added: int = 0


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: ollama.Client | None = None,
        embedder=None,
        llm=None,
        store: VectorStore | None = None,
        reranker=None,
        vision_llm=None,
    ):
        self.settings = s = settings or load_settings()
        if embedder is None or llm is None:
            client = client or make_client(s.ollama_host, s.allow_remote_ollama)
        self.client = client
        self.embedder = embedder or OllamaEmbedder(
            client,
            s.embed_model,
            s.embed_query_prefix,
            s.embed_doc_prefix,
            host=s.ollama_host,
        )
        self.llm = llm or OllamaLLM(
            client,
            s.llm_model,
            temperature=s.llm_temperature,
            num_ctx=s.llm_num_ctx,
            think=s.llm_think,
            host=s.ollama_host,
        )
        self.vision_llm = vision_llm
        if self.vision_llm is None and s.vision_enabled:
            if not s.vision_model:
                raise ValueError("VISION_ENABLED=true nhưng VISION_MODEL đang trống")
            if client is None:
                raise ValueError("VISION_ENABLED=true cần Ollama client")
            self.vision_llm = OllamaLLM(
                client,
                s.vision_model,
                temperature=0.0,
                num_ctx=s.llm_num_ctx,
                think=False,
                host=s.ollama_host,
            )
        self.store = store or VectorStore(s.db_dir)
        if reranker is None and s.rerank:
            from .reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(s.rerank_model)
        self.reranker = reranker

    def _signature(self) -> dict:
        s = self.settings
        return {
            "embed_model": s.embed_model,
            "embed_doc_prefix": s.embed_doc_prefix,
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
            "parser": s.parser,
            "vision_enabled": s.vision_enabled,
            "vision_model": s.vision_model,
            "schema": 2,
        }

    def _base_for(self, target: Path) -> Path:
        raw = self.settings.raw_dir.resolve()
        t = target.resolve()
        if t == raw or raw in t.parents:
            return raw
        return t.parent if t.is_file() else t

    def _image_document(self, path: Path, base: Path, digest: str) -> Document:
        if self.vision_llm is None:
            raise RuntimeError(
                f"{path.name}: ảnh cần VISION_ENABLED=true và VISION_MODEL để hệ thống đọc được."
            )
        prompt = (
            "Extract the visible content from this image for a retrieval system. "
            "Return concise plain text. Include exact names, codes, numbers, dates, labels, "
            "and table values. Keep table rows readable with one row per line."
        )
        extracted = self.vision_llm.vision_extract(path, prompt).strip()
        if not extracted:
            return Document(
                source=path.relative_to(base).as_posix(),
                file_hash=digest,
                sections=[],
            )
        return Document(
            source=path.relative_to(base).as_posix(),
            file_hash=digest,
            sections=[Section(text=extracted, heading=path.stem, kind="image")],
        )

    def ingest(
        self,
        path: Path | str | None = None,
        force: bool = False,
        progress: Progress | None = None,
    ) -> IngestReport:
        s = self.settings
        say = progress or (lambda _msg: None)
        target = Path(path) if path else s.raw_dir
        files = [f.resolve() for f in discover_files(target)]
        if not files:
            raise RuntimeError(
                f"Không tìm thấy tài liệu (.txt/.md/.pdf/.png/.jpg/.jpeg/.webp) nào trong {target}."
            )
        base = self._base_for(target)

        meta = IndexMeta.load(s.db_dir)
        signature = self._signature()
        rebuilt = force or meta is None or meta.signature != signature
        if rebuilt:
            self.store.reset()
            meta = IndexMeta(signature=signature)
            say("♻️  Build lại toàn bộ chỉ mục (lần đầu, đổi cấu hình hoặc --force)")
        report = IngestReport(rebuilt=rebuilt)

        seen: set[str] = set()
        for file in files:
            source = file.relative_to(base).as_posix()
            seen.add(source)
            try:
                digest = file_sha256(file)
                if meta.files.get(source) == digest:
                    report.unchanged.append(source)
                    continue

                if file.suffix.lower() in IMAGE_SUFFIXES:
                    doc = self._image_document(file, base, digest)
                else:
                    doc = load_document(file, base, s.parser, file_hash=digest)
                chunks = chunk_document(doc, s.chunk_size, s.chunk_overlap)
                if not chunks:
                    self.store.delete_source(source)
                    meta.files.pop(source, None)
                    report.skipped_empty.append(source)
                    say(f"⚠️  {source}: không có văn bản/ảnh đọc được")
                    continue

                vectors = self.embedder.embed_documents([c.embed_text for c in chunks])
                self.store.delete_source(source)
                self.store.add(chunks, vectors)
                (report.updated if source in meta.files else report.added).append(source)
                meta.files[source] = digest
                report.chunks_added += len(chunks)
                meta.save(s.db_dir)
                say(f"📄 {source}: {len(chunks)} chunk")
            except OllamaError:
                meta.save(s.db_dir)
                raise
            except Exception as exc:  # noqa: BLE001
                report.failed[source] = f"{type(exc).__name__}: {exc}"
                say(f"❌ {source}: {exc}")

        if path is None:
            for source in [x for x in meta.files if x not in seen]:
                self.store.delete_source(source)
                del meta.files[source]
                report.removed.append(source)
        meta.save(s.db_dir)
        return report

    def reset(self) -> None:
        self.store.reset()
        (self.settings.db_dir / META_FILE).unlink(missing_ok=True)

    def stats(self) -> dict:
        meta = IndexMeta.load(self.settings.db_dir)
        return {
            "files": len(meta.files) if meta else 0,
            "chunks": self.store.count(),
            "embed_model": meta.signature.get("embed_model") if meta else None,
            "parser": meta.signature.get("parser") if meta else None,
            "indexed": bool(meta and self.store.count()),
        }

    def _require_index(self) -> None:
        meta = IndexMeta.load(self.settings.db_dir)
        if meta is None or self.store.count() == 0:
            raise IndexMissingError("Chưa có chỉ mục. Hãy chạy trước: localrag ingest")
        built_with = meta.signature.get("embed_model")
        if built_with != self.settings.embed_model:
            raise IndexMismatchError(
                f"Chỉ mục được build bằng '{built_with}' nhưng EMBED_MODEL hiện tại là "
                f"'{self.settings.embed_model}'. Chạy: localrag ingest (sẽ build lại)."
            )

    def _rank_hybrid(self, question: str, results: list[SearchResult]) -> list[SearchResult]:
        s = self.settings
        for r in results:
            base = r.vector_score if r.vector_score is not None else r.score
            r.lexical_score = lexical_score(
                question,
                "\n".join(x for x in [r.heading, r.text] if x),
            )
            r.score = hybrid_score(base, r.lexical_score, s.vector_weight, s.lexical_weight)
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _limit_source_repetition(self, results: list[SearchResult]) -> list[SearchResult]:
        cap = self.settings.max_chunks_per_source
        counts: dict[str, int] = {}
        selected: list[SearchResult] = []
        deferred: list[SearchResult] = []
        for result in results:
            n = counts.get(result.source, 0)
            if n < cap:
                selected.append(result)
                counts[result.source] = n + 1
            else:
                deferred.append(result)
        return (selected + deferred)[: self.settings.top_k]

    def search(self, question: str, apply_threshold: bool = True) -> list[SearchResult]:
        s = self.settings
        self._require_index()
        vector = self.embedder.embed_query(question)
        candidate_limit = max(s.candidate_k, s.top_k * s.rerank_candidate_factor if self.reranker else s.top_k)
        results = self.store.search(vector, candidate_limit)
        if apply_threshold:
            results = [r for r in results if (r.vector_score or r.score) >= s.min_score]
        if not results:
            return []
        results = self._rank_hybrid(question, results)
        if self.reranker is not None:
            results = self.reranker.rerank(question, results)
        return self._limit_source_repetition(results)

    def answer(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
    ) -> RAGAnswer:
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        results = self.search(question)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        if not results:
            latency_ms = (time.perf_counter() - started) * 1000
            return RAGAnswer(
                NO_ANSWER,
                [],
                grounded=False,
                latency_ms=latency_ms,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                citations=[],
            )
        generation_started = time.perf_counter()
        text = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, results, history))
        generation_ms = (time.perf_counter() - generation_started) * 1000
        citations = extract_citations(text, len(results))
        citation_valid = bool(citations) and all(1 <= x <= len(results) for x in citations)
        latency_ms = (time.perf_counter() - started) * 1000
        return RAGAnswer(
            text,
            results,
            grounded=True,
            latency_ms=latency_ms,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            citations=citations,
            citation_valid=citation_valid,
        )

    def stream_answer(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
    ) -> tuple[list[SearchResult], Iterator[str]]:
        results = self.search(question)
        if not results:
            return [], iter([NO_ANSWER])
        tokens = self.llm.stream(SYSTEM_PROMPT, build_user_prompt(question, results, history))
        return results, tokens
