from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.answering import answer_question
from app.chunking import chunk_pages
from app.config import get_settings
from app.embeddings import EmbeddingService
from app.ingestion import parse_uploaded_pdfs
from app.llm import build_provider
from app.retrieval import RetrievalEngine
from app.vectorstore import FaissVectorStore


@dataclass
class EvaluationExample:
    question: str
    expected_keyword: str = ""


def run_lightweight_evaluation(pdf_paths: list[Path], examples: list[EvaluationExample]) -> pd.DataFrame:
    settings = get_settings()
    pages = parse_uploaded_pdfs(pdf_paths)
    chunks = chunk_pages(
        pages,
        max_words=settings.chunk_max_words,
        overlap_words=settings.chunk_overlap_words,
    )
    embedding_service = EmbeddingService()
    vectorstore = FaissVectorStore()
    vectorstore.build(chunks=chunks, embeddings=embedding_service.embed_chunks(chunks))
    retrieval_engine = RetrievalEngine(vectorstore=vectorstore, embedding_service=embedding_service)
    provider = build_provider()

    rows = []
    for example in examples:
        evidence = retrieval_engine.retrieve(example.question, top_k=settings.retrieval_top_k)
        result = answer_question(example.question, evidence=evidence, provider=provider)
        rows.append(
            {
                "question": example.question,
                "abstained": result.abstained,
                "confidence": result.confidence.level,
                "top_score": evidence[0].score if evidence else 0.0,
                "citation_count": len(result.citations),
                "keyword_present": (
                    example.expected_keyword.lower() in result.answer.lower()
                    if example.expected_keyword
                    else None
                ),
                "answer": result.answer,
            }
        )
    return pd.DataFrame(rows)
