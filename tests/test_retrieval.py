import pytest

from localrag.retrieval import hybrid_score, lexical_score


def test_lexical_score_matches_exact_identifier():
    q = "ABN 92 345 678 901"
    text = "Golden Swan Bakery — ABN 92 345 678 901 — Mon-Sun 6:00-17:00"
    assert lexical_score(q, text) > 0.5


def test_lexical_score_rejects_unrelated_text():
    assert lexical_score("police phone number", "Golden Swan Bakery opening hours") < 0.5


def test_hybrid_score_is_weighted_average():
    assert hybrid_score(0.8, 1.0, 0.8, 0.2) == pytest.approx(0.84)
