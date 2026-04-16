from app.confidence import score_confidence
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
