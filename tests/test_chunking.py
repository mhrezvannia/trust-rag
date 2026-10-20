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


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


def test_overlap_is_carried_between_paragraph_chunks() -> None:
    # Two 5-word paragraphs do not fit into one 8-word chunk, so the second chunk
    # must start with the last two words of the first one.
    text = _words("a", 5) + "\n\n" + _words("b", 5)
    pages = [ParsedPage(document_name="doc.pdf", page_number=1, text=text)]

    chunks = chunk_pages(pages, max_words=8, overlap_words=2)

    assert [chunk.text for chunk in chunks] == [
        "a0 a1 a2 a3 a4",
        "a3 a4 b0 b1 b2 b3 b4",
    ]


def test_overlap_is_dropped_when_it_would_exceed_max_words() -> None:
    text = _words("a", 5) + "\n\n" + _words("b", 7)
    pages = [ParsedPage(document_name="doc.pdf", page_number=1, text=text)]

    chunks = chunk_pages(pages, max_words=8, overlap_words=2)

    assert chunks[1].text == _words("b", 7)


def test_zero_overlap_produces_disjoint_chunks() -> None:
    text = _words("a", 5) + "\n\n" + _words("b", 5)
    pages = [ParsedPage(document_name="doc.pdf", page_number=1, text=text)]

    chunks = chunk_pages(pages, max_words=8, overlap_words=0)

    assert [chunk.text for chunk in chunks] == [_words("a", 5), _words("b", 5)]


def test_small_paragraphs_are_merged_into_one_chunk() -> None:
    text = "First short paragraph.\n\nSecond short paragraph.\r\n\r\nThird."
    pages = [ParsedPage(document_name="doc.pdf", page_number=1, text=text)]

    chunks = chunk_pages(pages, max_words=50, overlap_words=5)

    assert len(chunks) == 1
    assert chunks[0].text == "First short paragraph. Second short paragraph. Third."


def test_chunks_respect_max_words_and_have_unique_sequential_ids() -> None:
    pages = [
        ParsedPage(document_name="a.pdf", page_number=1, text=_words("x", 23)),
        ParsedPage(document_name="a.pdf", page_number=2, text="   \n\n  "),
        ParsedPage(document_name="b.pdf", page_number=1, text=_words("y", 4) + "\n\n" + _words("z", 6)),
    ]

    chunks = chunk_pages(pages, max_words=6, overlap_words=2)

    assert all(len(chunk.text.split()) <= 6 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert {chunk.document_name for chunk in chunks} == {"a.pdf", "b.pdf"}
    # Chunks never span pages and the blank page produces no chunk.
    assert all(chunk.page_start == chunk.page_end for chunk in chunks)
    assert 2 not in {chunk.page_start for chunk in chunks}
    assert chunks[-1].chunk_id == f"b.pdf-chunk-{len(chunks) - 1}"


def test_buffered_text_is_flushed_before_long_paragraph_windows() -> None:
    text = "intro words here\n\n" + _words("w", 10)
    pages = [ParsedPage(document_name="doc.pdf", page_number=3, text=text)]

    chunks = chunk_pages(pages, max_words=6, overlap_words=1)

    assert chunks[0].text == "intro words here"
    assert chunks[1].text == _words("w", 6)
    assert chunks[2].text.startswith("w5")
