"""LOAD: đọc tài liệu thô thành Document gồm các Section."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import Document, Section

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | IMAGE_SUFFIXES | {".pdf"}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def discover_files(root: Path) -> list[Path]:
    """Danh sách file được hỗ trợ, đệ quy, bỏ file ẩn, thứ tự ổn định."""
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_extracted_text(text: str) -> str:
    """Làm sạch PDF text nhưng giữ xuống dòng cần thiết cho đọc hiểu."""
    text = text.replace("\u00a0", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def split_markdown_sections(text: str) -> list[Section]:
    """Tách văn bản theo heading Markdown."""
    sections: list[Section] = []
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        body = text.strip()
        return [Section(text=body)] if body else []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(text=preamble))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            sections.append(Section(text=body, heading=m.group(2)))
    return sections


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _read_pdf_pypdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections: list[Section] = []
    for number, page in enumerate(reader.pages, start=1):
        text = normalize_extracted_text(page.extract_text() or "")
        if text:
            sections.append(Section(text=text, page=number))
    return sections


def _read_pdf_docling(path: Path) -> list[Section]:
    """Docling -> Markdown cho PDF có bảng/layout phức tạp.

    Docling export không phải lúc nào giữ được page provenance; khi bật parser này,
    citations theo heading/source vẫn giữ, còn page có thể không có giá trị.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'PARSER=docling cần cài thêm: pip install -e ".[docling]"'
        ) from exc

    markdown = DocumentConverter().convert(str(path)).document.export_to_markdown()
    return split_markdown_sections(markdown)


def load_document(
    path: Path, base: Path, parser: str = "pypdf", file_hash: str | None = None
) -> Document:
    """Đọc text/PDF. Ảnh được xử lý bởi VisionAnalyzer ở pipeline."""
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        sections = split_markdown_sections(_read_text_file(path))
    elif suffix == ".pdf":
        sections = _read_pdf_docling(path) if parser == "docling" else _read_pdf_pypdf(path)
    elif suffix in IMAGE_SUFFIXES:
        raise ValueError("File ảnh cần VISION_ENABLED=true và VISION_MODEL để ingest.")
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {path.suffix}")

    return Document(
        source=path.relative_to(base).as_posix(),
        file_hash=file_hash or file_sha256(path),
        sections=sections,
    )
