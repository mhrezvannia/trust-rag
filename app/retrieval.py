from __future__ import annotations

import re

from app.embeddings import EmbeddingService
from app.schemas import RetrievalResult
from app.vectorstore import FaissVectorStore


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",
}
INTENT_TERMS = {
    "future": {"future", "work", "direction", "directions"},
    "limitation": {"limitation", "limitations", "drawback", "drawbacks", "constraint", "constraints"},
    "result": {"result", "results", "accuracy", "performance", "metric", "metrics"},
    "dataset": {"dataset", "data", "samples", "preprocessing"},
    "method": {"method", "framework", "approach", "algorithm", "nystrom", "landmark"},
}


def _tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    }


def _infer_intent_terms(query_terms: set[str]) -> set[str]:
    inferred: set[str] = set()
    for terms in INTENT_TERMS.values():
        if query_terms & terms:
            inferred |= terms
    return inferred


def _rerank_results(query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
    query_terms = _tokenize(query)
    intent_terms = _infer_intent_terms(query_terms)

    reranked: list[RetrievalResult] = []
    for item in results:
        text_terms = _tokenize(item.text)
        lexical_overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
        intent_overlap = len(intent_terms & text_terms) / max(len(intent_terms), 1) if intent_terms else 0.0

        phrase_boost = 0.0
        text_lower = item.text.lower()
        if "future work" in query.lower() and "future work" in text_lower:
            phrase_boost += 0.12
        if "limitation" in query.lower() and "limitation" in text_lower:
            phrase_boost += 0.10
        if "conclusion" in query.lower() and "conclusion" in text_lower:
            phrase_boost += 0.08

        adjusted_score = (
            0.72 * item.score
            + 0.18 * lexical_overlap
            + 0.10 * intent_overlap
            + phrase_boost
        )
        reranked.append(item.model_copy(update={"score": round(min(adjusted_score, 1.0), 4)}))

    reranked.sort(key=lambda result: result.score, reverse=True)
    return reranked


class RetrievalEngine:
    def __init__(self, vectorstore: FaissVectorStore, embedding_service: EmbeddingService) -> None:
        self.vectorstore = vectorstore
        self.embedding_service = embedding_service

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_embedding = self.embedding_service.embed_query(query)
        initial_results = self.vectorstore.search(query_embedding=query_embedding, top_k=max(top_k * 3, top_k))
        reranked = _rerank_results(query=query, results=initial_results)
        return reranked[:top_k]
