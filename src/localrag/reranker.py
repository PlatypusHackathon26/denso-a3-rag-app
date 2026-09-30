"""Rerank cục bộ bằng cross-encoder (tuỳ chọn). Thay thế Cohere của project mẫu."""

from __future__ import annotations

from .models import SearchResult


class CrossEncoderReranker:
    """Chấm lại (câu hỏi, đoạn) bằng cross-encoder chạy trên máy.

    Model được tải từ Hugging Face ở lần chạy đầu. Sau đó có thể chạy offline
    bằng cách đặt HF_HUB_OFFLINE=1.
    """

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - phụ thuộc tuỳ chọn
            raise RuntimeError(
                'RERANK=true cần cài thêm: pip install -e ".[rerank]"'
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results
        pairs = [
            (query, f"{r.heading}\n{r.text}" if r.heading else r.text) for r in results
        ]
        for r, s in zip(results, self._model.predict(pairs)):
            r.rerank_score = float(s)
        return sorted(results, key=lambda r: r.rerank_score or 0.0, reverse=True)
