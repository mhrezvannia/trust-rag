from __future__ import annotations

from app.confidence import score_confidence
from app.llm import LLMProvider
from app.prompts import build_user_prompt
from app.schemas import AnswerResult, RetrievalResult


ABSTENTION_MESSAGE = (
    "I do not have sufficient evidence in the uploaded documents to answer this confidently."
)


def answer_question(
    question: str,
    evidence: list[RetrievalResult],
    provider: LLMProvider,
) -> AnswerResult:
    prompt = build_user_prompt(question=question, evidence=evidence)
    draft = provider.answer(question=question, evidence=evidence, prompt=prompt)
    confidence = score_confidence(
        evidence=evidence,
        citations=draft.citations,
        answer=draft.answer,
    )

    abstained = draft.abstain or confidence.should_abstain
    return AnswerResult(
        answer=ABSTENTION_MESSAGE if abstained else draft.answer,
        citations=[] if abstained else draft.citations,
        confidence=confidence,
        evidence=evidence,
        abstained=abstained,
        reasoning_summary=draft.reasoning_summary,
    )
