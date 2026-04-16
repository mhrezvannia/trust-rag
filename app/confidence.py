from __future__ import annotations

import re
from statistics import mean

from app.config import get_settings
from app.schemas import ConfidenceResult, RetrievalResult


NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def _lexical_coverage(answer: str, evidence: list[RetrievalResult]) -> float:
    answer_terms = {term.lower().strip(".,:;()[]") for term in answer.split() if len(term) > 3}
    evidence_terms = {
        term.lower().strip(".,:;()[]")
        for item in evidence
        for term in item.text.split()
        if len(term) > 3
    }
    if not answer_terms:
        return 0.0
    return len(answer_terms & evidence_terms) / len(answer_terms)


def _numeric_grounding(answer: str, evidence_text: str) -> float:
    answer_numbers = {token.strip() for token in NUMBER_RE.findall(answer)}
    if not answer_numbers:
        return 0.0
    evidence_numbers = {token.strip() for token in NUMBER_RE.findall(evidence_text)}
    return len(answer_numbers & evidence_numbers) / len(answer_numbers)


def score_confidence(
    evidence: list[RetrievalResult],
    citations: list[str],
    answer: str,
) -> ConfidenceResult:
    settings = get_settings()
    if not evidence:
        return ConfidenceResult(
            level="Low",
            numeric_score=0.0,
            reasons=["No supporting evidence was retrieved."],
            should_abstain=True,
        )

    scores = [item.score for item in evidence[: min(len(evidence), 5)]]
    top_score = scores[0]
    avg_score = mean(scores)
    top_two_avg = mean(scores[: min(len(scores), 2)])
    score_gap = scores[0] - scores[1] if len(scores) > 1 else scores[0]
    supported_chunks = sum(score >= settings.min_retrieval_score for score in scores)
    direct_support_threshold = max(settings.strong_retrieval_score, 0.65)
    strong_chunks = sum(score >= direct_support_threshold for score in scores)
    top_text = evidence[0].text if evidence else ""
    coverage = _lexical_coverage(answer, evidence)
    numeric_grounding = _numeric_grounding(answer, top_text)
    single_chunk_sufficiency = (
        top_score >= 0.58 and coverage >= 0.50 and (numeric_grounding >= 0.80 or len(top_text) >= 300)
    )
    effective_required_citations = 1 if single_chunk_sufficiency else max(supported_chunks, 1)
    citation_ratio = min(len(set(citations)) / effective_required_citations, 1.0)

    numeric_score = (
        0.22 * avg_score
        + 0.20 * top_score
        + 0.15 * top_two_avg
        + 0.10 * min(score_gap + 0.2, 1.0)
        + 0.10 * min(supported_chunks / 3, 1.0)
        + 0.10 * min(strong_chunks / 2, 1.0)
        + 0.07 * citation_ratio
        + 0.08 * coverage
        + 0.08 * numeric_grounding
    )
    numeric_score = max(0.0, min(numeric_score, 1.0))

    reasons: list[str] = []
    if top_score >= settings.strong_retrieval_score:
        reasons.append("Top retrieval score is strong.")
    elif top_score >= settings.min_retrieval_score:
        reasons.append("Top retrieval score is usable but not strong.")
    else:
        reasons.append("Top retrieval score is weak.")

    if supported_chunks >= 2:
        reasons.append("Multiple chunks support the answer.")
    else:
        reasons.append("Evidence is sparse across retrieved chunks.")

    if strong_chunks >= 1:
        reasons.append("At least one chunk is directly strong enough to support a grounded answer.")
    else:
        reasons.append("No chunk stands out as directly strong evidence.")

    if numeric_grounding >= 0.80:
        reasons.append("Exact numeric details in the answer are grounded in the top evidence chunk.")
    elif numeric_grounding > 0:
        reasons.append("Some numeric details in the answer are grounded in the evidence.")

    if single_chunk_sufficiency:
        reasons.append("A single high-value chunk appears sufficient to support the main answer.")

    if citation_ratio >= 0.67:
        reasons.append("Citations cover most of the supporting evidence.")
    else:
        reasons.append("Citation coverage is limited.")

    if coverage >= 0.45:
        reasons.append("Answer terminology overlaps with the evidence.")
    else:
        reasons.append("Answer wording has limited overlap with the evidence.")

    should_abstain = (
        top_score < settings.min_retrieval_score
        or supported_chunks == 0
        or (
            strong_chunks == 0
            and not single_chunk_sufficiency
            and (top_two_avg < 0.62 or coverage < 0.40 or citation_ratio < 0.67)
        )
        or (numeric_score < 0.45 and coverage < 0.45)
    )

    if numeric_score >= 0.72 and not should_abstain:
        level = "High"
    elif numeric_score >= 0.45 and not should_abstain:
        level = "Medium"
    else:
        level = "Low"

    return ConfidenceResult(
        level=level,
        numeric_score=round(numeric_score, 4),
        reasons=reasons,
        should_abstain=should_abstain,
    )
