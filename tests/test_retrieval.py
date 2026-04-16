import numpy as np
import pytest

from app.schemas import DocumentChunk
from app.vectorstore import FaissVectorStore


def test_vectorstore_search_returns_formatted_results() -> None:
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_name="doc.pdf",
            page_start=1,
            page_end=1,
            chunk_index=0,
            text="nystrom approximation uses landmarks",
        ),
        DocumentChunk(
            chunk_id="c2",
            document_name="doc.pdf",
            page_start=2,
            page_end=2,
            chunk_index=1,
            text="unrelated appendix material",
        ),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    query = np.array([1.0, 0.0], dtype="float32")

    store = FaissVectorStore()
    store.build(chunks, embeddings)
    results = store.search(query, top_k=2)

    assert results[0].chunk_id == "c1"
    assert results[0].document_name == "doc.pdf"
    assert 0.0 <= results[0].score <= 1.0
    assert len(results) == 2


def test_vectorstore_rejects_empty_build_inputs() -> None:
    store = FaissVectorStore()

    with pytest.raises(ValueError, match="empty chunks or embeddings"):
        store.build([], np.empty((0, 0), dtype="float32"))
