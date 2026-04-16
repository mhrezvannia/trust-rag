from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas import ProviderName


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "indices"

load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    llm_provider: ProviderName = Field(default="gemini", alias="LLM_PROVIDER")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="gemma3:4b", alias="OLLAMA_MODEL")
    ollama_num_ctx: int = Field(default=1024, alias="OLLAMA_NUM_CTX")
    ollama_num_predict: int = Field(default=256, alias="OLLAMA_NUM_PREDICT")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    chunk_max_words: int = 700
    chunk_overlap_words: int = 120
    retrieval_top_k: int = 5
    min_retrieval_score: float = 0.30
    strong_retrieval_score: float = 0.55
    answer_max_context_chars: int = 12000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return Settings()
