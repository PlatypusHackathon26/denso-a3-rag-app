from pathlib import Path

from localrag.loader import discover_files, load_document, split_markdown_sections


def test_markdown_sections_carry_headings():
    text = "Lời mở đầu\n\n# RAG là gì\nNội dung A\n\n## Vì sao cần\nNội dung B"
    sections = split_markdown_sections(text)
    assert [(s.heading, s.text) for s in sections] == [
        ("", "Lời mở đầu"),
        ("RAG là gì", "Nội dung A"),
        ("Vì sao cần", "Nội dung B"),
    ]


def test_plain_text_is_one_section():
    assert len(split_markdown_sections("chỉ có văn bản")) == 1
    assert split_markdown_sections("   ") == []


def test_discover_skips_hidden_and_unsupported(tmp_path: Path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.exe").write_text("x")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "c.md").write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.txt").write_text("x", encoding="utf-8")
    names = [p.relative_to(tmp_path).as_posix() for p in discover_files(tmp_path)]
    assert names == ["a.md", "sub/d.txt"]


def test_load_document_uses_posix_relative_source_and_strips_bom(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    f = sub / "n.md"
    f.write_bytes("\ufeff# Tiêu đề\nnội dung".encode("utf-8"))
    doc = load_document(f, tmp_path)
    assert doc.source == "sub/n.md"
    assert doc.sections[0].heading == "Tiêu đề"


def test_sample_pdf_has_page_numbers():
    pdf = Path(__file__).parents[1] / "data/raw/sl_booklet.pdf"
    if not pdf.exists():
        return
    doc = load_document(pdf, pdf.parent)
    assert doc.sections and all(s.page for s in doc.sections)
