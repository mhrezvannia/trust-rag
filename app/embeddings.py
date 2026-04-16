from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.config import CACHE_DIR, get_settings
from app.schemas import DocumentChunk
from app.utils import ensure_dir, stable_text_hash


class EmbeddingService:
    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.cache_dir = ensure_dir(CACHE_DIR / "embeddings")

    def _cache_paths(self, chunks: list[DocumentChunk]) -> tuple[Path, Path]:
        digest = stable_text_hash(
            self.model_name + "||" + "||".join(f"{chunk.chunk_id}:{chunk.text}" for chunk in chunks)
        )
        return self.cache_dir / f"{digest}.npy", self.cache_dir / f"{digest}.json"

    def embed_chunks(self, chunks: list[DocumentChunk]) -> np.ndarray:
        if not chunks:
            return np.empty((0, 0), dtype="float32")

        array_path, meta_path = self._cache_paths(chunks)
        if array_path.exists() and meta_path.exists():
            logger.info("Loading cached embeddings from {}", array_path.name)
            return np.load(array_path)

        embeddings = self.model.encode(
            [chunk.text for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        np.save(array_path, embeddings)
        meta_path.write_text(
            json.dumps({"model": self.model_name, "chunk_count": len(chunks)}, indent=2),
            encoding="utf-8",
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        return embedding[0]
