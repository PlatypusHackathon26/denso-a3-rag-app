"""CHUNK: chia Section thành các chunk theo đoạn văn, có overlap, không cắt giữa từ."""

from __future__ import annotations

import re

from .models import Chunk, Document

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Cắt một đoạn quá dài thành các mảnh <= size, ưu tiên cắt ở xuống dòng rồi tới khoảng trắng."""
    if len(text) <= size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            lo = start + size // 2
            cut = text.rfind("\n", lo, end)
            if cut == -1:
                cut = text.rfind(" ", lo, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = _snap_to_word_start(text, max(end - overlap, start + 1), end)
    return pieces


def _snap_to_word_start(text: str, pos: int, limit: int) -> int:
    """Đẩy `pos` tới đầu từ kế tiếp để chunk sau không bắt đầu giữa từ."""
    if pos == 0 or text[pos - 1].isspace():
        return pos
    for i in range(pos, limit):
        if text[i].isspace():
            return i + 1
    return pos  # token dài hơn cả vùng overlap: đành giữ nguyên


def _tail(text: str, overlap: int) -> str:
    """Phần đuôi ~overlap ký tự của chunk trước, cắt ở ranh giới từ."""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    if 0 <= space < len(tail) - 1:
        tail = tail[space + 1 :]
    return tail.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Gom các đoạn văn vào chunk tới khi đầy; chunk kế tiếp mở đầu bằng phần đuôi của chunk trước."""
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap phải nằm trong [0, chunk_size)")
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    pieces = [
        piece for p in paragraphs for piece in _hard_split(p, chunk_size, overlap)
    ]

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        candidate = f"{buf}\n\n{piece}" if buf else piece
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        chunks.append(buf)
        tail = _tail(buf, overlap)
        with_tail = f"{tail}\n\n{piece}" if tail else piece
        buf = with_tail if len(with_tail) <= chunk_size else piece
    if buf:
        chunks.append(buf)
    return chunks


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        for text in split_text(section.text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    source=doc.source,
                    index=len(chunks),
                    text=text,
                    heading=section.heading,
                    page=section.page,
                    kind=section.kind,
                )
            )
    return chunks
