from __future__ import annotations

import re

from app.schemas import DocumentChunk, ParsedPage
from app.utils import split_words


def _split_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    segments = re.split(r"\n\s*\n", normalized)
    cleaned = [" ".join(split_words(segment)) for segment in segments]
    return [segment for segment in cleaned if segment]


def _word_windows(words: list[str], max_words: int, overlap_words: int) -> list[str]:
    if not words:
        return []

    windows: list[str] = []
    start = 0
    stride = max(max_words - overlap_words, 1)
    while start < len(words):
        end = min(start + max_words, len(words))
        windows.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride
    return windows


def chunk_pages(
    pages: list[ParsedPage],
    max_words: int = 700,
    overlap_words: int = 120,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in pages:
        paragraphs = _split_paragraphs(page.text)
        if not paragraphs:
            continue

        buffer_words: list[str] = []
        current_page_start = page.page_number

        def flush_buffer() -> None:
            nonlocal buffer_words, chunk_index, current_page_start
            if not buffer_words:
                return

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{page.document_name}-chunk-{chunk_index}",
                    document_name=page.document_name,
                    page_start=current_page_start,
                    page_end=page.page_number,
                    chunk_index=chunk_index,
                    text=" ".join(buffer_words),
                )
            )
            chunk_index += 1
            buffer_words = buffer_words[-overlap_words:] if overlap_words else []
            current_page_start = page.page_number

        for paragraph in paragraphs:
            paragraph_words = split_words(paragraph)
            if len(paragraph_words) > max_words:
                if buffer_words:
                    flush_buffer()
                    buffer_words = []
                for window_text in _word_windows(paragraph_words, max_words, overlap_words):
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{page.document_name}-chunk-{chunk_index}",
                            document_name=page.document_name,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            chunk_index=chunk_index,
                            text=window_text,
                        )
                    )
                    chunk_index += 1
                current_page_start = page.page_number
                continue

            candidate = buffer_words + paragraph_words
            if len(candidate) <= max_words:
                buffer_words = candidate
            else:
                flush_buffer()
                buffer_words = paragraph_words

        if buffer_words:
            flush_buffer()

    return chunks
