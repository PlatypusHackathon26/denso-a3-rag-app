import pytest

from localrag.chunker import chunk_document, split_text
from localrag.models import Document, Section


def test_short_text_is_single_chunk():
    assert split_text("Xin chào thế giới", 200, 20) == ["Xin chào thế giới"]


def test_empty_text_has_no_chunks():
    assert split_text("  \n\n  ", 200, 20) == []


def test_chunks_never_exceed_size_and_cover_all_words():
    words = [f"từ{i}" for i in range(400)]
    text = " ".join(words)
    chunks = split_text(text, 200, 30)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)
    joined = " ".join(chunks)
    assert all(w in joined for w in words)


def test_no_word_is_cut_in_the_middle():
    text = " ".join(["abcdefghij"] * 200)
    for chunk in split_text(text, 120, 20):
        assert set(chunk.split()) == {"abcdefghij"}


def test_overlap_repeats_tail_of_previous_chunk():
    paras = "\n\n".join(" ".join(f"w{i}_{j}" for j in range(12)) for i in range(10))
    chunks = split_text(paras, 250, 60)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.split()[-1] in nxt.split()[:12]


def test_paragraphs_are_kept_whole_when_they_fit():
    chunks = split_text("Đoạn A ngắn.\n\nĐoạn B ngắn.", 200, 20)
    assert chunks == ["Đoạn A ngắn.\n\nĐoạn B ngắn."]


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)


def test_chunk_document_keeps_heading_and_page():
    doc = Document(
        source="a.md",
        file_hash="h",
        sections=[Section("thân 1", heading="Mục 1"), Section("thân 2", page=7)],
    )
    chunks = chunk_document(doc, 200, 20)
    assert [c.index for c in chunks] == [0, 1]
    assert chunks[0].heading == "Mục 1" and chunks[0].embed_text.startswith("Mục 1\n")
    assert chunks[1].page == 7


def test_chunk_id_is_stable_and_content_sensitive():
    doc = Document("a.md", "h", [Section("nội dung")])
    a = chunk_document(doc, 200, 20)[0]
    b = chunk_document(doc, 200, 20)[0]
    other = chunk_document(Document("b.md", "h", [Section("nội dung")]), 200, 20)[0]
    assert a.id == b.id != other.id
