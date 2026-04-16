from __future__ import annotations

from pathlib import Path

import fitz
from loguru import logger

from app.schemas import ParsedPage
from app.utils import normalize_whitespace


def _clean_page_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    cleaned_lines = [normalize_whitespace(line) if line.strip() else "" for line in lines]
    return "\n".join(cleaned_lines).strip()


def parse_pdf(pdf_path: Path) -> list[ParsedPage]:
    logger.info("Parsing PDF: {}", pdf_path.name)
    parsed_pages: list[ParsedPage] = []

    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            text = _clean_page_text(page.get_text("text"))
            if not text:
                continue
            parsed_pages.append(
                ParsedPage(
                    document_name=pdf_path.name,
                    page_number=index,
                    text=text,
                )
            )

    return parsed_pages


def parse_uploaded_pdfs(pdf_paths: list[Path]) -> list[ParsedPage]:
    pages: list[ParsedPage] = []
    for path in pdf_paths:
        pages.extend(parse_pdf(path))
    return pages
