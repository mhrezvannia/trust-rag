from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ParsedPage(BaseModel):
    document_name: str
    page_number: int
    text: str


class DocumentChunk(BaseModel):
    chunk_id: str
    document_name: str
    page_start: int
    page_end: int
    chunk_index: int
    text: str


class RetrievalResult(BaseModel):
    chunk_id: str
    document_name: str
    page_start: int
    page_end: int
    text: str
    score: float = Field(ge=0.0, le=1.0)


class ConfidenceResult(BaseModel):
    level: Literal["High", "Medium", "Low"]
    numeric_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str]
    should_abstain: bool


class AnswerResult(BaseModel):
    answer: str
    citations: list[str]
    confidence: ConfidenceResult
    evidence: list[RetrievalResult]
    abstained: bool
    reasoning_summary: str


class LLMAnswerDraft(BaseModel):
    answer: str
    citations: list[str]
    reasoning_summary: str
    abstain: bool = False


ProviderName = Literal["gemini", "ollama", "openai", "mock"]
