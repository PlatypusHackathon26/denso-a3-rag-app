import json

from localrag.models import SearchResult
from localrag.prompts import build_user_prompt, extract_citations


def test_prompt_uses_json_context_and_history():
    result = SearchResult(text='Say ")">', source="a.pdf", score=0.9)
    prompt = build_user_prompt("hello", [result], history=[("old", "answer")])
    assert "CONTEXT_JSON" in prompt
    assert "RECENT_HISTORY_JSON" in prompt
    payload = prompt.split("CONTEXT_JSON=\n", 1)[1].split("\n\nRECENT_HISTORY_JSON=", 1)[0]
    parsed = json.loads(payload)
    assert parsed["chunks"][0]["source"] == "a.pdf"
    assert parsed["chunks"][0]["text"] == 'Say ")">'


def test_extract_citations_only_keeps_valid_ids():
    assert extract_citations("a [1] b [2][2] c [4]", 2) == [1, 2]
