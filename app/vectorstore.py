from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.schemas import DocumentChunk, RetrievalResult


class FaissVectorStore:
    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[DocumentChunk] = []

    @property
    def size(self) -> int:
        return len(self.chunks)

    def build(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match.")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        if not chunks or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
            raise ValueError("Cannot build a vector index from empty chunks or embeddings.")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))
        self.chunks = list(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        if self.index is None or not self.chunks:
            return []

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        scores, indices = self.index.search(query_embedding.astype("float32"), top_k)
        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            normalized_score = float(max(min((score + 1.0) / 2.0, 1.0), 0.0))
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    document_name=chunk.document_name,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    score=normalized_score,
                )
            )
        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        if self.index is None:
            raise ValueError("Vector index has not been built.")
        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(
            json.dumps([chunk.model_dump() for chunk in self.chunks], indent=2),
            encoding="utf-8",
        )

    def load(self, index_path: Path, metadata_path: Path) -> None:
        self.index = faiss.read_index(str(index_path))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.chunks = [DocumentChunk(**item) for item in payload]
