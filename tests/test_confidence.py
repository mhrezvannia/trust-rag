import pytest

from app.confidence import NUMBER_RE, _lexical_coverage, _numeric_grounding, score_confidence
from app.schemas import RetrievalResult


def test_confidence_low_and_abstains_on_weak_evidence() -> None:
    evidence = [
        RetrievalResult(
            chunk_id="c1",
            document_name="paper.pdf",
            page_start=4,
            page_end=4,
            text="mostly irrelevant short text",
            score=0.12,
        )
    ]

    result = score_confidence(
        evidence=evidence,
        citations=["[Source: paper.pdf, p. 4, chunk c1]"],
        answer="An unsupported answer.",
    )

    assert result.level == "Low"
    assert result.should_abstain is True


def test_confidence_high_when_evidence_is_strong_and_aligned() -> None:
    evidence = [
        RetrievalResult(
            chunk_id="c1",
            document_name="paper.pdf",
            page_start=3,
            page_end=3,
            text="The method selects landmark points using approximate leverage scores.",
            score=0.86,
        ),
        RetrievalResult(
            chunk_id="c2",
            document_name="paper.pdf",
            page_start=4,
            page_end=4,
            text="Landmark selection improves the Nystrom approximation quality and stability.",
            score=0.81,
        ),
    ]

    result = score_confidence(
        evidence=evidence,
        citations=[
            "[Source: paper.pdf, p. 3, chunk c1]",
            "[Source: paper.pdf, p. 4, chunk c2]",
        ],
        answer="The paper says landmark points are selected with approximate leverage scores to improve Nystrom approximation quality.",
    )

    assert result.level in {"High", "Medium"}
    assert result.numeric_score > 0.6
    assert result.should_abstain is False


def test_confidence_drops_for_indirect_middling_evidence() -> None:
    evidence = [
        RetrievalResult(
            chunk_id="c1",
            document_name="paper.pdf",
            page_start=2,
            page_end=2,
            text="The broader field faces scalability challenges and open research questions.",
            score=0.58,
        ),
        RetrievalResult(
            chunk_id="c2",
            document_name="paper.pdf",
            page_start=3,
            page_end=3,
            text="Related work discusses complexity and instability in other methods.",
            score=0.56,
        ),
    ]

    result = score_confidence(
        evidence=evidence,
        citations=["[Source: paper.pdf, p. 2, chunk c1]"],
        answer="The paper does not mention future work explicitly.",
    )

    assert result.level == "Low"
    assert result.should_abstain is True


def test_confidence_accepts_single_chunk_with_exact_numeric_results() -> None:
    evidence = [
        RetrievalResult(
            chunk_id="c1",
            document_name="paper.pdf",
            page_start=8,
            page_end=8,
            text=(
                "The Breast Cancer Wisconsin dataset was used. Training time was reduced from "
                "0.0057 seconds to 0.0022 seconds, about 60%. Accuracy with the approximated "
                "kernel was approximately 0.65, with loss of less than 3%."
            ),
            score=0.60,
        )
    ]

    result = score_confidence(
        evidence=evidence,
        citations=["[Source: paper.pdf, p. 8, chunk c1]"],
        answer=(
            "The paper uses the Breast Cancer Wisconsin dataset and reports training time "
            "dropping from 0.0057 seconds to 0.0022 seconds, about 60%, with an accuracy "
            "loss of less than 3%."
        ),
    )

    assert result.level in {"Medium", "High"}
    assert result.should_abstain is False


def test_no_evidence_always_abstains() -> None:
    result = score_confidence(evidence=[], citations=[], answer="Anything.")

    assert result.level == "Low"
    assert result.numeric_score == 0.0
    assert result.should_abstain is True


def test_retrieval_score_below_minimum_forces_abstention_even_with_perfect_wording() -> None:
    text = "Landmark points are selected using approximate leverage scores."
    evidence = [
        RetrievalResult(chunk_id="c1", document_name="p.pdf", page_start=1, page_end=1, text=text, score=0.29)
    ]

    result = score_confidence(evidence=evidence, citations=["[Source: p.pdf, p. 1, chunk c1]"], answer=text)

    assert result.should_abstain is True
    assert "Top retrieval score is weak." in result.reasons


def test_numeric_regex_keeps_percent_signs() -> None:
    assert NUMBER_RE.findall("about 60%, loss below 3%. and 0.0057 s, 12abc") == ["60%", "3%", "0.0057"]


def test_numeric_grounding_distinguishes_percent_from_plain_number() -> None:
    assert _numeric_grounding("Accuracy improved by 60%.", "Accuracy improved by 60%.") == 1.0
    assert _numeric_grounding("Accuracy improved by 60%.", "Training took 60 seconds.") == 0.0
    assert _numeric_grounding("No numbers here.", "60%") == 0.0


def test_lexical_coverage_counts_long_terms_case_insensitively() -> None:
    evidence = [
        RetrievalResult(
            chunk_id="c1", document_name="p.pdf", page_start=1, page_end=1, text="Kernel (approximation).", score=0.9
        )
    ]

    # Terms of <= 3 characters are ignored; "kernel" and "approximation" match, "quality" does not.
    assert _lexical_coverage("The KERNEL approximation quality.", evidence) == pytest.approx(2 / 3)
    assert _lexical_coverage("a an the", evidence) == 0.0


def test_duplicate_citations_count_once() -> None:
    evidence = [
        RetrievalResult(
            chunk_id=f"c{i}",
            document_name="p.pdf",
            page_start=i,
            page_end=i,
            text="Landmark selection improves Nystrom approximation quality.",
            score=0.62,
        )
        for i in range(3)
    ]
    answer = "Landmark selection improves Nystrom approximation quality."
    one = "[Source: p.pdf, p. 0, chunk c0]"

    duplicated = score_confidence(evidence=evidence, citations=[one, one, one], answer=answer)
    single = score_confidence(evidence=evidence, citations=[one], answer=answer)

    assert duplicated.numeric_score == single.numeric_score
    assert "Citation coverage is limited." in duplicated.reasons
