import json

import pytest

from localrag.evaluation import (
    EvalResult,
    LLMJudge,
    extract_tag,
    load_eval_items,
    retrieval_metrics,
    run_eval,
    save_report,
    summarize,
)
from conftest import FakeLLM


def test_extract_tag():
    assert extract_tag("x <result> true </result> y", "result") == "true"
    assert extract_tag("không có thẻ", "result") is None
    assert extract_tag("<result>chưa đóng", "result") is None


def test_retrieval_metrics():
    assert retrieval_metrics(["a.pdf", "b.pdf"], ["b.pdf"]) == (1.0, 0.5)
    assert retrieval_metrics(["a.pdf"], ["b.pdf"]) == (0.0, 0.0)
    assert retrieval_metrics(["x/a.pdf", "b.pdf"], ["a.pdf", "b.pdf"]) == (1.0, 1.0)
    assert retrieval_metrics(["a.pdf"], ["a.pdf", "b.pdf"])[0] == 0.5
    with pytest.raises(ValueError):
        retrieval_metrics(["a"], [])


def test_judge_parses_verdicts():
    judge = LLMJudge(FakeLLM("<reasoning>khớp</reasoning><result>true</result>"))
    assert judge("q", "r", "e") == (True, "khớp")
    judge = LLMJudge(FakeLLM("<reasoning>sai</reasoning><result>False</result>"))
    assert judge("q", "r", "e")[0] is False
    ok, why = LLMJudge(FakeLLM("lan man không có thẻ"))("q", "r", "e")
    assert ok is None and "không trả về" in why


def test_run_eval_end_to_end(settings, make_pipeline, tmp_path):
    (settings.raw_dir / "pho.md").write_text("# Món\nPhở bò nấu từ xương bò hầm.", encoding="utf-8")
    (settings.raw_dir / "rag.md").write_text("# RAG\nRAG là truy xuất rồi sinh câu trả lời.", encoding="utf-8")
    pipeline = make_pipeline()
    pipeline.ingest()
    items = [
        {"question": "phở nấu từ gì xương bò", "answer": "xương bò", "sources": ["pho.md"]},
        {"question": "zzzz qqqq", "answer": "không có"},
    ]
    results = run_eval(pipeline, items, judge=lambda q, r, e: (True, "ok"))
    assert results[0].recall == 1.0 and results[0].rr == 1.0
    assert results[1].recall is None and results[1].grounded is False
    summary = summarize(results)
    assert summary["questions"] == 2 and summary["labeled_questions"] == 1
    assert summary["recall_at_k"] == 1.0 and summary["ungrounded"] == 1
    out = tmp_path / "r.json"
    save_report(results, out)
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["questions"] == 2


def test_summarize_counts_judge_failures():
    rs = [EvalResult("q", "e", "r", True, correct=None), EvalResult("q", "e", "r", True, correct=True)]
    s = summarize(rs)
    assert s["judge_failures"] == 1 and s["accuracy"] == 1.0


def test_load_eval_items_validates(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_items(bad)
