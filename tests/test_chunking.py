from app.chunking import chunk_pages
from app.schemas import ParsedPage


def test_chunking_is_deterministic_and_preserves_metadata() -> None:
    pages = [
        ParsedPage(
            document_name="paper.pdf",
            page_number=1,
            text=(
                "Paragraph one explains the method in detail.\n\n"
                "Paragraph two adds further technical description and assumptions.\n\n"
                "Paragraph three concludes the section."
            ),
        )
    ]

    first = chunk_pages(pages, max_words=8, overlap_words=2)
    second = chunk_pages(pages, max_words=8, overlap_words=2)

    assert [chunk.model_dump() for chunk in first] == [chunk.model_dump() for chunk in second]
    assert first[0].document_name == "paper.pdf"
    assert first[0].page_start == 1
    assert first[0].chunk_index == 0


def test_long_paragraphs_are_split_with_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(20))
    pages = [ParsedPage(document_name="long.pdf", page_number=2, text=text)]

    chunks = chunk_pages(pages, max_words=7, overlap_words=2)

    assert len(chunks) == 4
    assert chunks[1].text.startswith("word5")
    assert chunks[-1].page_start == 2
