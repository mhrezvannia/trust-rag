"""Shared fixtures: every test runs offline with the mock LLM and a fake embedder."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from app.config import get_settings
from app.schemas import DocumentChunk, RetrievalResult

_ENV_DEFAULTS = {
    "LLM_PROVIDER": "mock",
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": "",
}


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch: pytest.MonkeyPatch):
    """Pin provider settings so a developer's local .env cannot leak into tests."""
    for key, value in _ENV_DEFAULTS.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeEmbeddingService:
    """Deterministic bag-of-words hashing embedder (no model download).

    Texts sharing more tokens get higher cosine similarity, which is enough to
    exercise ranking logic without sentence-transformers.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model_name = "fake-hashing-embedder"

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype="float32")
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_chunks(self, chunks: list[DocumentChunk]) -> np.ndarray:
        if not chunks:
            return np.empty((0, 0), dtype="float32")
        return np.stack([self._embed(chunk.text) for chunk in chunks]).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed(query)


@pytest.fixture
def fake_embedder() -> FakeEmbeddingService:
    return FakeEmbeddingService()


def make_result(
    chunk_id: str = "c1",
    text: str = "evidence text",
    score: float = 0.8,
    page: int = 1,
    document_name: str = "paper.pdf",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_name=document_name,
        page_start=page,
        page_end=page,
        text=text,
        score=score,
    )
