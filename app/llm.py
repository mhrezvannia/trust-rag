from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from urllib import request
from urllib.error import HTTPError, URLError

from openai import OpenAI
from pydantic import ValidationError

from app.config import get_settings
from app.prompts import SYSTEM_PROMPT
from app.schemas import LLMAnswerDraft, RetrievalResult


class LLMProvider(ABC):
    @abstractmethod
    def answer(self, question: str, evidence: list[RetrievalResult], prompt: str) -> LLMAnswerDraft:
        raise NotImplementedError


class LLMProviderError(RuntimeError):
    """Raised when the selected LLM provider cannot complete a request."""


def _parse_json_response(content: str) -> LLMAnswerDraft:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    if not cleaned:
        raise LLMProviderError("The model returned an empty response.")

    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            cleaned = match.group(0)

    try:
        return LLMAnswerDraft(**json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Failed to parse model output: {content}") from exc


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.openai_model
        self.client = OpenAI(api_key=settings.openai_api_key)

    def answer(self, question: str, evidence: list[RetrievalResult], prompt: str) -> LLMAnswerDraft:
        response = self.client.responses.create(
            model=self.model,
            temperature=0,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return _parse_json_response(response.output_text.strip())


class GeminiProvider(LLMProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model
        self.base_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )

    def answer(self, question: str, evidence: list[RetrievalResult], prompt: str) -> LLMAnswerDraft:
        payload = json.dumps(
            {
                "systemInstruction": {
                    "parts": [
                        {
                            "text": SYSTEM_PROMPT,
                        }
                    ]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt,
                            }
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                },
            }
        ).encode("utf-8")
        req = request.Request(
            url=self.base_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = ""
            if exc.fp is not None:
                try:
                    error_body = exc.fp.read().decode("utf-8", errors="replace")
                    detail = f" Details: {error_body}"
                except Exception:
                    detail = ""
            raise LLMProviderError(
                f"Gemini request failed with HTTP {exc.code}.{detail}"
            ) from exc
        except URLError as exc:
            raise LLMProviderError(
                "Could not reach the Gemini API. Check your network access and API key."
            ) from exc

        candidates = body.get("candidates", [])
        if not candidates:
            raise LLMProviderError(f"Gemini returned no candidates. Response: {body}")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        return _parse_json_response(text)


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.num_ctx = settings.ollama_num_ctx
        self.num_predict = settings.ollama_num_predict

    def answer(self, question: str, evidence: list[RetrievalResult], prompt: str) -> LLMAnswerDraft:
        payload = json.dumps(
            {
                "model": self.model,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
            }
        ).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = ""
            if exc.fp is not None:
                try:
                    error_body = exc.fp.read().decode("utf-8", errors="replace")
                    detail = f" Details: {error_body}"
                except Exception:
                    detail = ""
            raise LLMProviderError(
                f"Ollama request failed with HTTP {exc.code}.{detail}"
            ) from exc
        except URLError as exc:
            raise LLMProviderError(
                f"Could not reach Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from exc

        return _parse_json_response(body.get("response", "").strip())


class MockProvider(LLMProvider):
    def answer(self, question: str, evidence: list[RetrievalResult], prompt: str) -> LLMAnswerDraft:
        settings = get_settings()
        if not evidence or evidence[0].score < settings.min_retrieval_score:
            return LLMAnswerDraft(
                answer="I do not have sufficient evidence in the uploaded documents to answer this confidently.",
                citations=[],
                reasoning_summary="Retrieved evidence was missing or too weak to support a grounded answer.",
                abstain=True,
            )

        top = evidence[0]
        citation = f"[Source: {top.document_name}, p. {top.page_start}, chunk {top.chunk_id}]"
        return LLMAnswerDraft(
            answer=f"Based on the retrieved evidence, the documents indicate: {top.text[:280]}",
            citations=[citation],
            reasoning_summary="The answer is grounded in the top retrieved chunk.",
            abstain=False,
        )


def build_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        return GeminiProvider()
    if settings.llm_provider == "ollama":
        return OllamaProvider()
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAIProvider()
    return MockProvider()
