"""Provider layer tests. Network calls are replaced by stubs; nothing leaves the machine."""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pytest
from conftest import make_result
from openai import OpenAIError

from app import llm
from app.config import get_settings
from app.llm import (
    GeminiProvider,
    LLMProviderError,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    _parse_json_response,
    build_provider,
)

VALID_JSON = '{"answer": "A", "citations": ["[Source: p.pdf, p. 1, chunk c1]"], "reasoning_summary": "R", "abstain": false}'


def _use_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


class _FakeHTTPResponse:
    def __init__(self, body: dict) -> None:
        self._raw = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


# --- JSON parsing ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        VALID_JSON,
        f"```json\n{VALID_JSON}\n```",
        f"Here is the answer:\n{VALID_JSON}\nThanks.",
    ],
)
def test_parse_json_response_accepts_plain_fenced_and_wrapped_json(content: str) -> None:
    draft = _parse_json_response(content)

    assert draft.answer == "A"
    assert draft.citations == ["[Source: p.pdf, p. 1, chunk c1]"]
    assert draft.abstain is False


@pytest.mark.parametrize("content", ["", "```\n```", "not json at all", '{"answer": "missing fields"}'])
def test_parse_json_response_raises_provider_error_on_bad_output(content: str) -> None:
    # The UI only catches LLMProviderError, so malformed output must not surface as another type.
    with pytest.raises(LLMProviderError):
        _parse_json_response(content)


# --- provider selection ----------------------------------------------------------------


def test_build_provider_falls_back_to_mock_without_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_settings(monkeypatch, LLM_PROVIDER="gemini", GEMINI_API_KEY="")
    assert isinstance(build_provider(), MockProvider)

    _use_settings(monkeypatch, LLM_PROVIDER="openai", OPENAI_API_KEY="")
    assert isinstance(build_provider(), MockProvider)

    _use_settings(monkeypatch, LLM_PROVIDER="mock")
    assert isinstance(build_provider(), MockProvider)


def test_build_provider_selects_configured_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_settings(monkeypatch, LLM_PROVIDER="gemini", GEMINI_API_KEY="test-key")
    assert isinstance(build_provider(), GeminiProvider)

    _use_settings(monkeypatch, LLM_PROVIDER="ollama")
    assert isinstance(build_provider(), OllamaProvider)

    _use_settings(monkeypatch, LLM_PROVIDER="openai", OPENAI_API_KEY="test-key")
    assert isinstance(build_provider(), OpenAIProvider)


# --- Gemini ----------------------------------------------------------------------------


def test_gemini_provider_sends_prompt_and_parses_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_settings(monkeypatch, LLM_PROVIDER="gemini", GEMINI_API_KEY="test-key", GEMINI_MODEL="gemini-test")
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse({"candidates": [{"content": {"parts": [{"text": VALID_JSON}]}}]})

    monkeypatch.setattr(llm.request, "urlopen", fake_urlopen)

    draft = GeminiProvider().answer("Q?", [make_result()], prompt="PROMPT")

    assert draft.answer == "A"
    assert captured["url"].endswith("/models/gemini-test:generateContent")
    assert captured["headers"]["X-goog-api-key"] == "test-key"
    assert captured["payload"]["contents"][0]["parts"][0]["text"] == "PROMPT"
    assert captured["payload"]["generationConfig"]["temperature"] == 0


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("https://x", 503, "Service Unavailable", {}, io.BytesIO(b"overloaded")),
        URLError("no route"),
        TimeoutError("timed out"),
    ],
)
def test_gemini_transport_errors_become_provider_errors(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    _use_settings(monkeypatch, LLM_PROVIDER="gemini", GEMINI_API_KEY="test-key")

    def fake_urlopen(req, timeout):
        raise error

    monkeypatch.setattr(llm.request, "urlopen", fake_urlopen)

    with pytest.raises(LLMProviderError):
        GeminiProvider().answer("Q?", [make_result()], prompt="PROMPT")


def test_gemini_without_candidates_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_settings(monkeypatch, LLM_PROVIDER="gemini", GEMINI_API_KEY="test-key")
    monkeypatch.setattr(llm.request, "urlopen", lambda req, timeout: _FakeHTTPResponse({"candidates": []}))

    with pytest.raises(LLMProviderError, match="no candidates"):
        GeminiProvider().answer("Q?", [make_result()], prompt="PROMPT")


# --- Ollama ----------------------------------------------------------------------------


def test_ollama_provider_posts_generate_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_settings(monkeypatch, LLM_PROVIDER="ollama", OLLAMA_BASE_URL="http://ollama.test:11434/", OLLAMA_NUM_CTX="2048")
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse({"response": VALID_JSON})

    monkeypatch.setattr(llm.request, "urlopen", fake_urlopen)

    draft = OllamaProvider().answer("Q?", [make_result()], prompt="PROMPT")

    assert draft.answer == "A"
    assert captured["url"] == "http://ollama.test:11434/api/generate"
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["options"]["num_ctx"] == 2048


@pytest.mark.parametrize("error", [URLError("connection refused"), TimeoutError("timed out")])
def test_ollama_unreachable_becomes_provider_error(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    _use_settings(monkeypatch, LLM_PROVIDER="ollama")

    def fake_urlopen(req, timeout):
        raise error

    monkeypatch.setattr(llm.request, "urlopen", fake_urlopen)

    with pytest.raises(LLMProviderError, match="Could not reach Ollama"):
        OllamaProvider().answer("Q?", [make_result()], prompt="PROMPT")


# --- OpenAI ----------------------------------------------------------------------------


class _StubResponses:
    def __init__(self, output_text: str | None = None, error: Exception | None = None) -> None:
        self.output_text = output_text
        self.error = error
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return type("Response", (), {"output_text": self.output_text})()


def test_openai_provider_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_settings(monkeypatch, LLM_PROVIDER="openai", OPENAI_API_KEY="test-key", OPENAI_MODEL="gpt-test")
    provider = OpenAIProvider()
    stub = _StubResponses(output_text=f"  {VALID_JSON}  ")
    provider.client = type("Client", (), {"responses": stub})()

    draft = provider.answer("Q?", [make_result()], prompt="PROMPT")

    assert draft.answer == "A"
    assert stub.kwargs["model"] == "gpt-test"
    assert stub.kwargs["input"][1] == {"role": "user", "content": "PROMPT"}


def test_openai_client_errors_become_provider_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_settings(monkeypatch, LLM_PROVIDER="openai", OPENAI_API_KEY="test-key")
    provider = OpenAIProvider()
    provider.client = type("Client", (), {"responses": _StubResponses(error=OpenAIError("quota exceeded"))})()

    with pytest.raises(LLMProviderError, match="quota exceeded"):
        provider.answer("Q?", [make_result()], prompt="PROMPT")


# --- Mock ------------------------------------------------------------------------------


def test_mock_provider_cites_top_chunk_with_prompt_citation_format() -> None:
    evidence = [make_result("c7", "Landmarks are sampled by leverage.", score=0.8, page=4), make_result("c2")]

    draft = MockProvider().answer("Q?", evidence, prompt="")

    assert draft.abstain is False
    assert draft.citations == ["[Source: paper.pdf, p. 4, chunk c7]"]
    assert "Landmarks are sampled by leverage." in draft.answer


def test_mock_provider_abstains_on_missing_or_weak_evidence() -> None:
    assert MockProvider().answer("Q?", [], prompt="").abstain is True
    assert MockProvider().answer("Q?", [make_result(score=0.1)], prompt="").abstain is True
