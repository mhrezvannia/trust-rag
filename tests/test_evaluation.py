"""PDF ingestion and the lightweight evaluation loop, run on small PDFs generated in the test."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from conftest import FakeEmbeddingService

from app import evaluation
from app.evaluation import EvaluationExample, run_lightweight_evaluation
from app.ingestion import parse_pdf, parse_uploaded_pdfs
from app.notebook_data import DEFAULT_EVAL_SET

PAGE_TEXTS = [
    "Landmark points for the Nystrom approximation are selected using approximate leverage scores.",
    "",
    "Limitations: the method assumes a fixed kernel bandwidth and was evaluated on two datasets only.",
]


def _write_pdf(path: Path, page_texts: list[str]) -> Path:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_textbox(fitz.Rect(72, 72, 540, 770), text, fontsize=11)
    document.save(path)
    document.close()
    return path


@pytest.fixture
def paper_pdf(tmp_path: Path) -> Path:
    return _write_pdf(tmp_path / "paper.pdf", PAGE_TEXTS)


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "EmbeddingService", FakeEmbeddingService)


def test_parse_pdf_skips_blank_pages_and_keeps_page_numbers(paper_pdf: Path) -> None:
    pages = parse_pdf(paper_pdf)

    assert [page.page_number for page in pages] == [1, 3]
    assert {page.document_name for page in pages} == {"paper.pdf"}
    assert "leverage scores" in " ".join(pages[0].text.split())
    assert pages[1].text.startswith("Limitations:")


def test_parse_uploaded_pdfs_concatenates_documents(tmp_path: Path, paper_pdf: Path) -> None:
    other = _write_pdf(tmp_path / "other.pdf", ["Another document."])

    pages = parse_uploaded_pdfs([paper_pdf, other])

    assert [(page.document_name, page.page_number) for page in pages] == [
        ("paper.pdf", 1),
        ("paper.pdf", 3),
        ("other.pdf", 1),
    ]


@pytest.mark.usefixtures("fake_embeddings")
def test_lightweight_evaluation_reports_one_row_per_question(paper_pdf: Path) -> None:
    examples = [
        EvaluationExample(question="How are landmark points selected?", expected_keyword="leverage"),
        EvaluationExample(question="Which limitations are discussed?", expected_keyword="quantum"),
        EvaluationExample(question="What problem is the document solving?"),
    ]

    frame = run_lightweight_evaluation([paper_pdf], examples)

    assert list(frame.columns) == [
        "question",
        "abstained",
        "confidence",
        "top_score",
        "citation_count",
        "keyword_present",
        "answer",
    ]
    assert frame["question"].tolist() == [example.question for example in examples]
    assert frame["confidence"].isin(["High", "Medium", "Low"]).all()
    assert frame["top_score"].between(0.0, 1.0).all()
    # Abstained answers carry no citations; answered ones cite the evidence (mock provider: one citation).
    for row in frame.itertuples():
        assert row.citation_count == (0 if row.abstained else 1)
    # keyword_present is None without an expected keyword, otherwise a case-insensitive substring check.
    assert frame.loc[2, "keyword_present"] is None
    assert frame.loc[1, "keyword_present"] is False
    first = frame.loc[0]
    assert first["keyword_present"] is (not first["abstained"] and "leverage" in first["answer"].lower())


@pytest.mark.usefixtures("fake_embeddings")
def test_lightweight_evaluation_runs_default_question_set(paper_pdf: Path) -> None:
    frame = run_lightweight_evaluation([paper_pdf], DEFAULT_EVAL_SET)

    assert len(frame) == len(DEFAULT_EVAL_SET)
    assert frame["keyword_present"].isna().all()


@pytest.mark.usefixtures("fake_embeddings")
def test_lightweight_evaluation_rejects_pdfs_without_text(tmp_path: Path) -> None:
    blank = _write_pdf(tmp_path / "blank.pdf", ["", ""])

    with pytest.raises(ValueError, match="No extractable text"):
        run_lightweight_evaluation([blank], DEFAULT_EVAL_SET)
