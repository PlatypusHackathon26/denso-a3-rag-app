"""CLI: doctor | ingest | ask | chat | search | eval | stats | reset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings, load_settings
from .evaluation import LLMJudge, load_eval_items, run_eval, save_report, summarize
from .ollama_client import OllamaError, OllamaLLM, PrivacyError, ensure_local_host, make_client, missing_models
from .pipeline import IndexMismatchError, IndexMissingError, RAGPipeline
from .store import IndexMeta

DEFAULT_EVAL_FILE = "eval/questions.json"


def _print_sources(sources) -> None:
    if not sources:
        return
    print("\n📚 Nguồn:")
    for i, r in enumerate(sources, start=1):
        head = f" — {r.heading}" if r.heading else ""
        rerank = f", rerank {r.rerank_score:.2f}" if r.rerank_score is not None else ""
        lexical = f", lexical {r.lexical_score:.2f}" if r.lexical_score is not None else ""
        print(
            f"  [{i}] {r.label}{head}  "
            f"(hybrid {r.score:.3f}, vector {r.vector_score:.3f}{lexical}{rerank})"
        )


def cmd_doctor(settings: Settings, _args) -> int:
    print("🩺 Kiểm tra môi trường\n")
    ensure_local_host(settings.ollama_host, settings.allow_remote_ollama)
    scope = "cục bộ" if not settings.allow_remote_ollama else "CHO PHÉP máy khác"
    print(f"• Ollama host : {settings.ollama_host}  ({scope})")
    print(f"• LLM         : {settings.llm_model}  (num_ctx={settings.llm_num_ctx}, think={settings.llm_think})")
    print(f"• Embedding   : {settings.embed_model}")
    print(f"• Retrieval   : top_k={settings.top_k}, candidates={settings.candidate_k}, hybrid={settings.vector_weight:.2f}/{settings.lexical_weight:.2f}")
    print(f"• Parser      : {settings.parser} | rerank: {settings.rerank} | vision: {settings.vision_enabled}")

    client = make_client(settings.ollama_host, settings.allow_remote_ollama)
    wanted = [settings.llm_model, settings.embed_model]
    if settings.vision_enabled:
        wanted.append(settings.vision_model)
    if settings.judge != settings.llm_model:
        wanted.append(settings.judge)
    missing = missing_models(client, wanted, settings.ollama_host)
    for name in dict.fromkeys(wanted):
        print(f"  {'❌ thiếu' if name in missing else '✅ có   '}  {name}")
    for name in missing:
        print(f"     -> ollama pull {name}")

    meta = IndexMeta.load(settings.db_dir)
    if meta is None:
        print("\n• Chỉ mục: chưa có (chạy: localrag ingest)")
    else:
        print(f"\n• Chỉ mục: {len(meta.files)} file, build bằng {meta.signature.get('embed_model')}")
    return 1 if missing else 0


def cmd_ingest(settings: Settings, args) -> int:
    pipeline = RAGPipeline(settings)
    report = pipeline.ingest(args.path, force=args.force, progress=print)
    print(
        f"\n✅ Xong: +{len(report.added)} mới, ~{len(report.updated)} cập nhật, "
        f"={len(report.unchanged)} không đổi, -{len(report.removed)} đã xoá, "
        f"{report.chunks_added} chunk."
    )
    if report.failed:
        print(f"⚠️  {len(report.failed)} file lỗi (xem log ở trên).")
        return 1
    return 0


def _ask_once(pipeline: RAGPipeline, question: str, stream: bool) -> None:
    if stream:
        sources, tokens = pipeline.stream_answer(question)
        print("\n🤖 ", end="", flush=True)
        for token in tokens:
            print(token, end="", flush=True)
        print()
    else:
        result = pipeline.answer(question)
        sources = result.sources
        print(f"\n🤖 {result.answer}")
        if result.latency_ms is not None:
            print(f"\n⏱  {result.latency_ms:.0f} ms (retrieval {result.retrieval_ms:.0f} ms, generation {result.generation_ms:.0f} ms)")
    _print_sources(sources)


def cmd_ask(settings: Settings, args) -> int:
    _ask_once(RAGPipeline(settings), " ".join(args.question), stream=not args.no_stream)
    return 0


def cmd_chat(settings: Settings, _args) -> int:
    pipeline = RAGPipeline(settings)
    history: list[tuple[str, str]] = []
    print("💬 Hỏi đáp trên tài liệu cục bộ (gõ 'exit' để thoát). Có nhớ 4 lượt gần nhất.")
    while True:
        try:
            question = input("\n❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit", "q", "thoat"}:
            return 0
        if not question:
            continue
        result = pipeline.answer(question, history=history)
        print(f"\n🤖 {result.answer}")
        _print_sources(result.sources)
        history.append((question, result.answer))
        history = history[-4:]


def cmd_search(settings: Settings, args) -> int:
    pipeline = RAGPipeline(settings)
    results = pipeline.search(" ".join(args.question), apply_threshold=False)
    if not results:
        print("Không có kết quả.")
        return 0
    print(f"(ngưỡng vector hiện tại RAG_MIN_SCORE={settings.min_score})\n")
    for i, r in enumerate(results, start=1):
        mark = "✅" if (r.vector_score or 0) >= settings.min_score else "⛔"
        snippet = " ".join(r.text.split())[:220]
        print(
            f"{mark} [{i}] hybrid={r.score:.3f} vector={r.vector_score:.3f} "
            f"lexical={r.lexical_score:.3f}  {r.label}\n     {snippet}…"
        )
    return 0


def cmd_eval(settings: Settings, args) -> int:
    client = make_client(settings.ollama_host, settings.allow_remote_ollama)
    pipeline = RAGPipeline(settings, client=client)
    judge_llm = OllamaLLM(client, settings.judge, temperature=0.0, num_ctx=settings.llm_num_ctx, think=False, host=settings.ollama_host)
    items = load_eval_items(Path(args.file))

    def progress(n, total, res):
        icon = {True: "✅", False: "❌", None: "❓"}[res.correct]
        print(f"{icon} [{n}/{total}] {res.question}")

    results = run_eval(pipeline, items, judge=LLMJudge(judge_llm), progress=progress)
    summary = summarize(results)
    print("\n===== Kết quả =====")
    if summary["accuracy"] is not None:
        print(f"Đúng: {summary['correct']}/{len([r for r in results if r.correct is not None])} ({summary['accuracy']:.0%})")
    if summary["recall_at_k"] is not None:
        print(f"Retrieval: Recall@{settings.top_k}={summary['recall_at_k']:.2f}, MRR={summary['mrr']:.2f}")
    if summary["avg_latency_ms"] is not None:
        print(f"Latency: avg={summary['avg_latency_ms']:.0f} ms, p95={summary['p95_latency_ms']:.0f} ms")
    print(f"Citation rate: {summary['citation_rate']:.0%}")
    print(f"Câu không có evidence vượt ngưỡng: {summary['ungrounded']}")
    save_report(results, Path(args.output))
    print(f"Báo cáo chi tiết: {args.output}")
    return 0


def cmd_stats(settings: Settings, _args) -> int:
    stats = RAGPipeline(settings).stats()
    print("📊 Index stats")
    for key, value in stats.items():
        print(f"• {key}: {value}")
    return 0


def cmd_reset(settings: Settings, _args) -> int:
    RAGPipeline(settings, embedder=_Unused(), llm=_Unused(), vision_llm=_Unused()).reset()
    print("🗑️  Đã xoá chỉ mục.")
    return 0


class _Unused:
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localrag", description="Local Enterprise RAG")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Kiểm tra Ollama, model và chỉ mục")

    p = sub.add_parser("ingest", help="Đồng bộ tài liệu vào chỉ mục")
    p.add_argument("-p", "--path")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("ask", help="Hỏi một câu")
    p.add_argument("question", nargs="+")
    p.add_argument("--no-stream", action="store_true")
    sub.add_parser("chat", help="Hỏi đáp liên tục với 4 lượt nhớ gần nhất")

    p = sub.add_parser("search", help="Debug hybrid retrieval")
    p.add_argument("question", nargs="+")

    p = sub.add_parser("eval", help="Đánh giá answer + retrieval + latency")
    p.add_argument("-f", "--file", default=DEFAULT_EVAL_FILE)
    p.add_argument("-o", "--output", default="eval/reports/latest.json")
    sub.add_parser("stats", help="Xem thống kê index")
    sub.add_parser("reset", help="Xoá toàn bộ chỉ mục")
    return parser


COMMANDS = {
    "doctor": cmd_doctor,
    "ingest": cmd_ingest,
    "ask": cmd_ask,
    "chat": cmd_chat,
    "search": cmd_search,
    "eval": cmd_eval,
    "stats": cmd_stats,
    "reset": cmd_reset,
}


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](load_settings(), args)
    except (OllamaError, PrivacyError, IndexMissingError, IndexMismatchError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
