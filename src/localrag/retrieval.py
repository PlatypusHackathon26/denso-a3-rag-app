"""Các hàm scoring bổ sung cho vector retrieval: exact term + number aware hybrid ranking."""

from __future__ import annotations

import re
from collections import Counter

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_DIGIT_RE = re.compile(r"\d+")


def _tokens(text: str) -> list[str]:
    return [x.casefold() for x in _TOKEN_RE.findall(text)]


def _compact_numbers(text: str) -> set[str]:
    return {re.sub(r"\D", "", x) for x in _DIGIT_RE.findall(text) if re.search(r"\d", x)}


def lexical_score(query: str, text: str) -> float:
    """Điểm 0..1; ưu tiên token hiếm, cụm từ và chuỗi số exact.

    Đây không phải BM25 đầy đủ, nhưng cực rẻ và hữu ích cho mã hàng, ABN,
    số điện thoại, part-number, tên viết tắt và địa chỉ.
    """
    q = _tokens(query)
    d = _tokens(text)
    if not q or not d:
        return 0.0

    q_counter = Counter(q)
    d_set = set(d)
    overlap = sum(1 for token in q_counter if token in d_set)
    token_score = overlap / max(1, len(q_counter))

    query_norm = " ".join(q)
    text_norm = " ".join(d)
    phrase_bonus = 1.0 if len(q) >= 2 and query_norm in text_norm else 0.0

    q_nums = _compact_numbers(query)
    d_nums = _compact_numbers(text)
    number_bonus = len(q_nums & d_nums) / len(q_nums) if q_nums else 0.0

    return min(1.0, 0.55 * token_score + 0.25 * phrase_bonus + 0.20 * number_bonus)


def hybrid_score(vector: float, lexical: float, vector_weight: float, lexical_weight: float) -> float:
    total = vector_weight + lexical_weight
    return (vector_weight * vector + lexical_weight * lexical) / total
