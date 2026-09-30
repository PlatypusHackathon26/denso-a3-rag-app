"""Chạy pipeline thật (pypdf + LanceDB) trên PDF mẫu, chỉ thay Ollama bằng bản giả."""

from dataclasses import replace
from pathlib import Path

import pytest

from conftest import FakeEmbedder, FakeLLM
from localrag.pipeline import RAGPipeline

RAW = Path(__file__).parents[1] / "data" / "raw"
pytestmark = pytest.mark.skipif(not list(RAW.glob("*.pdf")), reason="thiếu PDF mẫu")


@pytest.fixture
def sample_pipeline(settings):
    s = replace(settings, raw_dir=RAW, chunk_size=800, chunk_overlap=120, top_k=5, min_score=0.1)
    pipeline = RAGPipeline(s, embedder=FakeEmbedder(), llm=FakeLLM())
    pipeline.ingest()
    return pipeline


@pytest.mark.parametrize(
    "question, expected_source",
    [
        ("ABN Golden Swan Bakery", "sl_business_directory.pdf"),
        ("Lagoon Breeze Hotel opened 1962", "sl_booklet.pdf"),
        ("recycling collection frequency", "sl_service_guide.pdf"),
    ],
)
def test_keyword_queries_hit_expected_pdf(sample_pipeline, question, expected_source):
    sources = [r.source for r in sample_pipeline.search(question)]
    assert expected_source in sources[:3]


def test_results_carry_page_numbers(sample_pipeline):
    assert all(r.page for r in sample_pipeline.search("Golden Swan Mine"))
