"""Citation grounding in the prompt and abstention in the final answer."""

from __future__ import annotations

import json

import pytest
from conftest import make_result

from app.answering import ABSTENTION_MESSAGE, answer_question
from app.config import get_settings
from app.llm import LLMProvider, MockProvider
from app.prompts import build_user_prompt
from app.schemas import LLMAnswerDraft

STRONG_TEXT = (
    "Landmark points for the Nystrom approximation are selected using approximate ridge leverage scores, "
    "which bounds the kernel approximation error while keeping the number of landmarks small."
)


class StubProvider(LLMProvider):
    def __init__(self, draft: LLMAnswerDraft) -> None:
        self.draft = draft
        self.calls: list[tuple[str, str]] = []

    def answer(self, question, evidence, prompt) -> LLMAnswerDraft:
        self.calls.append((question, prompt))
        return self.draft


def _evidence_payload(prompt: str) -> list[dict]:
    start = prompt.index("Evidence:\n") + len("Evidence:\n")
    end = prompt.index("\n\nUse only directly relevant evidence")
    return json.loads(prompt[start:end])


def test_prompt_lists_each_chunk_with_its_exact_citation_string() -> None:
    evidence = [
        make_result("doc.pdf-chunk-3", "First chunk.", score=0.81234, page=2, document_name="doc.pdf"),
        make_result("doc.pdf-chunk-9", "Second chunk.", score=0.5, page=7, document_name="doc.pdf"),
    ]

    prompt = build_user_prompt("What is proposed?", evidence)
    payload = _evidence_payload(prompt)

    assert prompt.startswith("Question: What is proposed?")
    assert [item["citation"] for item in payload] == [
        "[Source: doc.pdf, p. 2, chunk doc.pdf-chunk-3]",
        "[Source: doc.pdf, p. 7, chunk doc.pdf-chunk-9]",
    ]
    assert payload[0]["score"] == 0.8123
    assert payload[1]["text"] == "Second chunk."


def test_prompt_respects_context_character_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANSWER_MAX_CONTEXT_CHARS", "15")
    get_settings.cache_clear()
    evidence = [make_result("c1", "0123456789"), make_result("c2", "abcdefghij"), make_result("c3", "never shown")]

    payload = _evidence_payload(build_user_prompt("Q?", evidence))

    assert [item["text"] for item in payload] == ["0123456789", "abcde"]


def test_grounded_draft_is_returned_with_its_citations() -> None:
    evidence = [make_result("c1", STRONG_TEXT, score=0.82), make_result("c2", STRONG_TEXT, score=0.74)]
    citations = ["[Source: paper.pdf, p. 1, chunk c1]", "[Source: paper.pdf, p. 1, chunk c2]"]
    provider = StubProvider(
        LLMAnswerDraft(
            answer="Landmark points are selected using approximate ridge leverage scores.",
            citations=citations,
            reasoning_summary="Both chunks state the selection rule.",
        )
    )

    result = answer_question("How are landmarks selected?", evidence, provider)

    assert result.abstained is False
    assert result.answer.startswith("Landmark points are selected")
    assert result.citations == citations
    assert result.confidence.level in {"Medium", "High"}
    assert result.evidence == evidence
    assert provider.calls[0][1] == build_user_prompt("How are landmarks selected?", evidence)


def test_model_abstention_replaces_answer_and_drops_citations() -> None:
    evidence = [make_result("c1", STRONG_TEXT, score=0.82)]
    provider = StubProvider(
        LLMAnswerDraft(
            answer="Some speculative answer.",
            citations=["[Source: paper.pdf, p. 1, chunk c1]"],
            reasoning_summary="Evidence is indirect.",
            abstain=True,
        )
    )

    result = answer_question("Q?", evidence, provider)

    assert result.abstained is True
    assert result.answer == ABSTENTION_MESSAGE
    assert result.citations == []
    assert result.reasoning_summary == "Evidence is indirect."


def test_confidence_gate_overrides_a_non_abstaining_model() -> None:
    # The model answers confidently, but retrieval support is below the minimum score.
    evidence = [make_result("c1", "Unrelated acknowledgements section.", score=0.2)]
    provider = StubProvider(
        LLMAnswerDraft(
            answer="The method uses quantum annealing.",
            citations=["[Source: paper.pdf, p. 1, chunk c1]"],
            reasoning_summary="Confident.",
        )
    )

    result = answer_question("Q?", evidence, provider)

    assert result.abstained is True
    assert result.answer == ABSTENTION_MESSAGE
    assert result.citations == []
    assert result.confidence.level == "Low"


def test_no_evidence_abstains_end_to_end_with_mock_provider() -> None:
    result = answer_question("Q?", [], MockProvider())

    assert result.abstained is True
    assert result.confidence.should_abstain is True
    assert result.citations == []
