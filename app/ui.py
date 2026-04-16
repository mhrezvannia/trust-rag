from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from app.answering import ABSTENTION_MESSAGE, answer_question
from app.chunking import chunk_pages
from app.config import DATA_DIR, get_settings
from app.embeddings import EmbeddingService
from app.ingestion import parse_uploaded_pdfs
from app.llm import LLMProviderError, build_provider
from app.retrieval import RetrievalEngine
from app.sample_questions import SAMPLE_QUESTIONS
from app.schemas import AnswerResult
from app.vectorstore import FaissVectorStore


def _confidence_badge(level: str) -> str:
    colors = {"High": "green", "Medium": "orange", "Low": "red"}
    return f":{colors.get(level, 'gray')}[{level}]"


def _persist_uploads(files) -> list[Path]:
    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for file in files:
        target = upload_dir / file.name
        target.write_bytes(file.getbuffer())
        paths.append(target)
    return paths


def _provider_name(provider: Any) -> str:
    return type(provider).__name__.replace("Provider", "")


def _provider_error_help(provider_name: str) -> str:
    if provider_name == "Gemini":
        return "Gemini is temporarily unavailable due to high demand. Wait a bit and retry, or switch to `mock` for retrieval/UI testing."
    if provider_name == "Ollama":
        return "Try lowering `OLLAMA_NUM_CTX`, closing other memory-heavy apps, switching providers, or using a smaller local model."
    if provider_name == "OpenAI":
        return "Check your API key, model name, network access, and account quota, then retry."
    return "Check the provider configuration and try again."


def _initialize_state() -> None:
    settings = get_settings()
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = FaissVectorStore()
    provider_signature = (
        settings.llm_provider,
        settings.gemini_model,
        bool(settings.gemini_api_key),
        settings.openai_model,
        bool(settings.openai_api_key),
        settings.ollama_model,
        settings.ollama_num_ctx,
        settings.ollama_num_predict,
    )
    if st.session_state.get("provider_signature") != provider_signature:
        st.session_state.provider = build_provider()
        st.session_state.provider_signature = provider_signature


def _get_embedding_service() -> EmbeddingService:
    if "embedding_service" not in st.session_state:
        st.session_state.embedding_service = EmbeddingService()
    return st.session_state.embedding_service


def _get_retrieval_engine() -> RetrievalEngine:
    if "retrieval_engine" not in st.session_state:
        st.session_state.retrieval_engine = RetrievalEngine(
            st.session_state.vectorstore,
            _get_embedding_service(),
        )
    return st.session_state.retrieval_engine


def render_app() -> None:
    settings = get_settings()
    st.set_page_config(page_title="TrustRAG", page_icon=":books:", layout="wide")
    _initialize_state()

    st.title("TrustRAG")
    st.caption(
        "A trustworthy retrieval-augmented QA system for technical and academic PDFs. "
        "It retrieves evidence, cites sources, estimates confidence, and abstains when support is weak."
    )

    with st.sidebar:
        st.subheader("Sample Questions")
        for sample in SAMPLE_QUESTIONS:
            if st.button(sample, use_container_width=True):
                st.session_state["question_input"] = sample

        with st.expander("Developer Panel", expanded=False):
            st.json(
                {
                    "llm_provider": _provider_name(st.session_state.provider),
                    "embedding_model": settings.embedding_model,
                    "gemini_model": settings.gemini_model,
                    "openai_model": settings.openai_model,
                    "ollama_model": settings.ollama_model,
                    "ollama_num_ctx": settings.ollama_num_ctx,
                    "has_gemini_key": bool(settings.gemini_api_key),
                    "has_openai_key": bool(settings.openai_api_key),
                    "indexed_chunks": st.session_state.vectorstore.size,
                }
            )

    upload_col, status_col = st.columns([1.4, 1])
    with upload_col:
        st.subheader("1. Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        build_clicked = st.button("Parse and Index Documents", disabled=not uploaded_files)

    with status_col:
        st.subheader("2. Index Status")
        st.metric("Indexed chunks", st.session_state.vectorstore.size)
        st.metric("LLM provider", _provider_name(st.session_state.provider))

    if build_clicked and uploaded_files:
        with st.spinner("Building TrustRAG index..."):
            pdf_paths = _persist_uploads(uploaded_files)
            pages = parse_uploaded_pdfs(pdf_paths)
            if not pages:
                st.session_state.index_summary = None
                st.session_state.last_result = None
                st.warning("No extractable text was found in the uploaded PDFs.")
                return

            chunks = chunk_pages(
                pages,
                max_words=settings.chunk_max_words,
                overlap_words=settings.chunk_overlap_words,
            )
            if not chunks:
                st.session_state.index_summary = None
                st.session_state.last_result = None
                st.warning("Parsed pages were empty after chunking. Try a text-based PDF.")
                return

            embedding_service = _get_embedding_service()
            embeddings = embedding_service.embed_chunks(chunks)
            st.session_state.vectorstore.build(chunks=chunks, embeddings=embeddings)
            st.session_state.retrieval_engine = RetrievalEngine(
                st.session_state.vectorstore,
                embedding_service,
            )
            st.session_state.index_summary = {
                "documents": len(pdf_paths),
                "pages": len(pages),
                "chunks": len(chunks),
            }

    if st.session_state.get("index_summary"):
        st.success(
            "Indexed {documents} document(s), {pages} page(s), and {chunks} chunk(s).".format(
                **st.session_state.index_summary
            )
        )

    st.subheader("3. Ask a Question")
    question = st.text_input(
        "Ask a grounded question about the uploaded documents",
        key="question_input",
        placeholder="What does the paper say about landmark selection in the Nystrom approximation?",
    )
    ask_clicked = st.button(
        "Answer Question",
        disabled=st.session_state.vectorstore.size == 0 or not question,
    )

    if ask_clicked and question:
        evidence = _get_retrieval_engine().retrieve(
            query=question,
            top_k=settings.retrieval_top_k,
        )
        try:
            st.session_state.last_result = answer_question(
                question=question,
                evidence=evidence,
                provider=st.session_state.provider,
            )
            st.session_state.provider_error = ""
        except LLMProviderError as exc:
            st.session_state.provider_error = str(exc)
            st.session_state.last_result = None

    provider_error = st.session_state.get("provider_error", "")
    if provider_error:
        active_provider_name = _provider_name(st.session_state.provider)
        st.error(provider_error)
        st.info(_provider_error_help(active_provider_name))

    result: AnswerResult | None = st.session_state.get("last_result")
    if not result:
        return

    st.subheader("4. Answer")
    if result.abstained:
        st.warning(ABSTENTION_MESSAGE)
    st.write(result.answer)

    answer_col, trust_col = st.columns([2, 1])
    with answer_col:
        st.subheader("5. Supporting Evidence")
        if result.citations:
            st.markdown("\n".join(f"- `{citation}`" for citation in result.citations))
        else:
            st.caption("No citations are shown because the system abstained.")

        with st.expander("View Retrieved Evidence", expanded=True):
            for item in result.evidence:
                st.markdown(
                    f"**{item.document_name}** | p. {item.page_start}-{item.page_end} | "
                    f"score={item.score:.3f} | chunk=`{item.chunk_id}`"
                )
                st.write(item.text)
                st.divider()

    with trust_col:
        st.subheader("6. Confidence / Trust Signal")
        st.markdown(_confidence_badge(result.confidence.level))
        st.metric("Numeric score", f"{result.confidence.numeric_score:.2f}")
        if result.confidence.level == "Low":
            st.warning("Low confidence: evidence may be insufficient.")
        st.markdown("**Short rationale**")
        st.write(result.reasoning_summary)
        st.markdown("**Heuristic signals**")
        for reason in result.confidence.reasons:
            st.caption(f"- {reason}")
