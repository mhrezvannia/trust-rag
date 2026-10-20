"""Headless smoke test of the Streamlit app (mock LLM, fake embedder, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeEmbeddingService

from app.schemas import DocumentChunk
from app.vectorstore import FaissVectorStore

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

MAIN_SCRIPT = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


def _indexed_store(embedder: FakeEmbeddingService) -> FaissVectorStore:
    texts = [
        "Landmark points are selected using approximate leverage scores for the Nystrom approximation.",
        "We thank the reviewers for their helpful comments.",
    ]
    chunks = [
        DocumentChunk(
            chunk_id=f"paper.pdf-chunk-{i}",
            document_name="paper.pdf",
            page_start=i + 1,
            page_end=i + 1,
            chunk_index=i,
            text=text,
        )
        for i, text in enumerate(texts)
    ]
    store = FaissVectorStore()
    store.build(chunks, embedder.embed_chunks(chunks))
    return store


def test_app_renders_without_documents() -> None:
    app = AppTest.from_file(MAIN_SCRIPT, default_timeout=60).run()

    assert not app.exception
    assert app.title[0].value == "TrustRAG"
    answer_button = next(button for button in app.button if button.label == "Answer Question")
    assert answer_button.disabled


def test_app_answers_a_question_over_an_existing_index() -> None:
    embedder = FakeEmbeddingService()
    app = AppTest.from_file(MAIN_SCRIPT, default_timeout=60)
    app.session_state["vectorstore"] = _indexed_store(embedder)
    app.session_state["embedding_service"] = embedder
    app.run()

    app.text_input(key="question_input").input("How are landmark points selected?").run()
    next(button for button in app.button if button.label == "Answer Question").click().run()

    assert not app.exception
    subheaders = [element.value for element in app.subheader]
    assert "4. Answer" in subheaders
    assert any("[Source: paper.pdf, p. 1, chunk paper.pdf-chunk-0]" in md.value for md in app.markdown)
