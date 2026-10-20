import numpy as np
import pytest
from conftest import make_result

from app.retrieval import RetrievalEngine, _rerank_results
from app.schemas import DocumentChunk
from app.vectorstore import FaissVectorStore


def _chunk(chunk_id: str, text: str, page: int = 1) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_name="doc.pdf",
        page_start=page,
        page_end=page,
        chunk_index=int(chunk_id.lstrip("c")),
        text=text,
    )


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


def test_vectorstore_rejects_mismatched_inputs() -> None:
    store = FaissVectorStore()

    with pytest.raises(ValueError, match="counts must match"):
        store.build([_chunk("c1", "text")], np.ones((2, 3), dtype="float32"))


def test_vectorstore_maps_cosine_to_unit_interval() -> None:
    store = FaissVectorStore()
    store.build(
        [_chunk("c1", "same"), _chunk("c2", "orthogonal"), _chunk("c3", "opposite")],
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype="float32"),
    )

    results = store.search(np.array([1.0, 0.0], dtype="float32"), top_k=3)

    # score = (cosine + 1) / 2
    assert [(r.chunk_id, r.score) for r in results] == [("c1", 1.0), ("c2", 0.5), ("c3", 0.0)]


def test_vectorstore_search_handles_empty_index_and_large_top_k() -> None:
    store = FaissVectorStore()
    assert store.search(np.array([1.0, 0.0], dtype="float32")) == []

    store.build([_chunk("c1", "only")], np.array([[1.0, 0.0]], dtype="float32"))
    results = store.search(np.array([1.0, 0.0], dtype="float32"), top_k=10)
    assert [r.chunk_id for r in results] == ["c1"]


def test_vectorstore_save_and_load_roundtrip(tmp_path) -> None:
    store = FaissVectorStore()
    store.build(
        [_chunk("c1", "alpha"), _chunk("c2", "beta")],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
    )
    store.save(tmp_path / "index.faiss", tmp_path / "meta.json")

    loaded = FaissVectorStore()
    loaded.load(tmp_path / "index.faiss", tmp_path / "meta.json")

    assert loaded.size == 2
    assert loaded.search(np.array([0.0, 1.0], dtype="float32"), top_k=1)[0].chunk_id == "c2"


def test_rerank_promotes_lexical_overlap_over_small_dense_margin() -> None:
    results = [
        make_result("c1", "General background on kernel methods.", score=0.70),
        make_result("c2", "Landmark selection uses leverage scores.", score=0.66),
    ]

    reranked = _rerank_results("How does landmark selection work?", results)

    assert [r.chunk_id for r in reranked] == ["c2", "c1"]
    # Query terms {landmark, selection, work}: 2/3 overlap. "landmark" and "work" activate the
    # "method" and "future" intent groups (10 terms), of which 1 ("landmark") occurs in the text.
    assert reranked[0].score == pytest.approx(0.72 * 0.66 + 0.18 * 2 / 3 + 0.10 * 1 / 10, abs=1e-4)
    assert reranked[1].score == pytest.approx(0.72 * 0.70, abs=1e-4)


def test_rerank_applies_section_phrase_boost() -> None:
    results = [
        make_result("c1", "Results on the benchmark are reported here.", score=0.60),
        make_result("c2", "In future work we will extend the model.", score=0.55),
    ]

    reranked = _rerank_results("What future work is proposed?", results)

    assert reranked[0].chunk_id == "c2"
    assert reranked[0].score > reranked[1].score + 0.12


def test_rerank_scores_are_capped_and_sorted() -> None:
    results = [
        make_result("c1", "limitation future work conclusion", score=1.0),
        make_result("c2", "nothing relevant", score=0.2),
    ]

    reranked = _rerank_results("limitation future work conclusion", results)

    assert reranked[0].score == 1.0
    assert [r.score for r in reranked] == sorted((r.score for r in reranked), reverse=True)


def test_retrieval_engine_ranks_relevant_chunk_first(fake_embedder) -> None:
    chunks = [
        _chunk("c0", "The appendix lists hyperparameters for all baselines."),
        _chunk("c1", "Landmark points are selected using approximate leverage scores."),
        _chunk("c2", "We thank the reviewers for helpful comments."),
        _chunk("c3", "Training time and accuracy results on the benchmark dataset."),
    ]
    store = FaissVectorStore()
    store.build(chunks, fake_embedder.embed_chunks(chunks))
    engine = RetrievalEngine(vectorstore=store, embedding_service=fake_embedder)

    results = engine.retrieve("How are landmark points selected?", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].score >= results[1].score
