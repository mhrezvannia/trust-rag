from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, TypeVar

from loguru import logger


T = TypeVar("T")


def configure_logging(level: str) -> None:
    logger.remove()
    logger.add(lambda message: print(message, end=""), level=level)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_words(text: str) -> list[str]:
    return [token for token in re.findall(r"\S+", text) if token]


def batched(items: list[T], size: int) -> Iterable[list[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
