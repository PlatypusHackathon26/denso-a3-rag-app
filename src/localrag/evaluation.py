"""Đánh giá retrieval + answer accuracy + latency bằng LLM judge cục bộ."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .pipeline import RAGPipeline

JUDGE_SYSTEM = """You grade whether a response answers a question correctly from the supplied evidence.
Return <reasoning>...</reasoning> and then <result>true</result> or <result>false</result>.
A response is correct when it answers the question accurately; wording does not have to match the reference.
Do not use outside knowledge to rescue an unsupported answer.
"""


def extract_tag(text: str, tag: str) -> str | None:
    start_tag, end_tag = f"<{tag}>", f"</{tag}>"
    start = text.find(start_tag)
    if start == -1:
        return None
    end = text.find(end_tag, start + len(start_tag))
    if end == -1:
        return None
    return text[start + len(start_tag) : end].strip()


class LLMJudge:
    def __init__(self, llm):
        self._llm = llm

    def __call__(self, question: str, response: str, expected: str) -> tuple[bool | None, str]:
        prompt = (
            f"<question>\n{question}\n</question>\n"
            f"<response>\n{response}\n</response>\n"
            f"<expected_answer>\n{expected}\n</expected_answer>"
        )
        raw = self._llm.generate(JUDGE_SYSTEM, prompt)
        verdict = extract_tag(raw, "result")
        reasoning = extract_tag(raw, "reasoning") or raw[:300]
        if verdict is None:
            return None, f"Judge không trả về <result>: {raw[:300]}"
        return verdict.strip().lower() == "true", reasoning


def _matches(source: str, expected: str) -> bool:
    return source == expected or source.endswith("/" + expected)


def retrieval_metrics(retrieved: list[str], expected: list[str]) -> tuple[float, float]:
    if not expected:
        raise ValueError("expected rỗng")
    found = sum(any(_matches(r, e) for r in retrieved) for e in expected)
    rank = next(
        (i for i, r in enumerate(retrieved, 1) if any(_matches(r, e) for e in expected)),
        None,
    )
    return found / len(expected), (1.0 / rank if rank else 0.0)


@dataclass
class EvalResult:
    question: str
    expected: str
    response: str
    grounded: bool
    retrieved: list[str] = field(default_factory=list)
    correct: bool | None = None
    reasoning: str = ""
    recall: float | None = None
    rr: float | None = None
    latency_ms: float | None = None
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    citations: list[int] = field(default_factory=list)
    citation_valid: bool = True


def load_eval_items(path: Path) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list) or not all("question" in i and "answer" in i for i in items):
        raise ValueError("File đánh giá phải là danh sách object có 'question' và 'answer'")
    return items


def run_eval(
    pipeline: RAGPipeline,
    items: list[dict],
    judge: Callable[[str, str, str], tuple[bool | None, str]] | None = None,
    progress: Callable[[int, int, EvalResult], None] | None = None,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for n, item in enumerate(items, start=1):
        ans = pipeline.answer(item["question"])
        res = EvalResult(
            question=item["question"],
            expected=item["answer"],
            response=ans.answer,
            grounded=ans.grounded,
            retrieved=[r.source for r in ans.sources],
            latency_ms=ans.latency_ms,
            retrieval_ms=ans.retrieval_ms,
            generation_ms=ans.generation_ms,
            citations=ans.citations,
            citation_valid=ans.citation_valid,
        )
        if item.get("sources"):
            res.recall, res.rr = retrieval_metrics(res.retrieved, item["sources"])
        if judge is not None:
            res.correct, res.reasoning = judge(res.question, res.response, res.expected)
        results.append(res)
        if progress:
            progress(n, len(items), res)
    return results


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def summarize(results: list[EvalResult]) -> dict:
    n = len(results)
    judged = [r for r in results if r.correct is not None]
    labeled = [r for r in results if r.recall is not None]
    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    retrieval = [r.retrieval_ms for r in results if r.retrieval_ms is not None]
    generation = [r.generation_ms for r in results if r.generation_ms is not None]
    return {
        "questions": n,
        "correct": sum(bool(r.correct) for r in judged),
        "judge_failures": sum(r.correct is None for r in results),
        "accuracy": (sum(bool(r.correct) for r in judged) / len(judged)) if judged else None,
        "ungrounded": sum(not r.grounded for r in results),
        "labeled_questions": len(labeled),
        "recall_at_k": (sum(r.recall for r in labeled) / len(labeled)) if labeled else None,
        "mrr": (sum(r.rr for r in labeled) / len(labeled)) if labeled else None,
        "citation_rate": (
            sum(bool(r.citations) for r in results) / n if n else None
        ),
        "avg_latency_ms": statistics.mean(latencies) if latencies else None,
        "p95_latency_ms": _p95(latencies),
        "avg_retrieval_ms": statistics.mean(retrieval) if retrieval else None,
        "avg_generation_ms": statistics.mean(generation) if generation else None,
    }


def save_report(results: list[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summarize(results), "results": [asdict(r) for r in results]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
