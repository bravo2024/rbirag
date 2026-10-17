"""Structure-aware chunking: labels, sections, merging, oversized splitting."""
from regintel import chunk

SAMPLE = """CHAPTER I
PRELIMINARY AND SCOPE
1. Short title and commencement. These Directions may be called the Test Directions, 2026, and shall come into force with immediate effect. The provisions apply to every entity specified in paragraph 3 of these Directions.
1.1 The entities shall establish a board-approved policy for the matters covered under these Directions and review it periodically.
(a) the policy shall be reviewed at least once a year;
(b) the policy shall be placed before the board of the entity.
2. Maintenance of records. Every entity shall maintain records of all transactions for a period of at least five years from the date of the transaction, in electronic form where possible.
2.1 The records shall be preserved in a manner that they can be retrieved readily and produced to the Reserve Bank on demand.
3. Reporting requirements. The entities shall submit a quarterly return to the Reserve Bank of India giving the details specified in the Annex.
ANNEX A
The return shall contain the entity name, registration number and the data fields listed below.
"""


def _pages(text):
    return [{"page": 1, "text": text}]


def test_basic_chunks_and_labels():
    chunks = chunk.build_chunks("t1", _pages(SAMPLE))
    assert chunks, "chunker produced nothing"
    ids = [c["chunk_id"] for c in chunks]
    assert ids == [f"t1::{i:04d}" for i in range(len(chunks))]
    text_all = " ".join(c["text"] for c in chunks)
    assert "five years" in text_all
    paras = {c["para"] for c in chunks}
    assert paras & {"1", "1.1", "2", "3"}
    assert any("CHAPTER I" in c["section"] for c in chunks)
    assert all(c["n_words"] <= chunk.HARD_MAX_WORDS + 60 for c in chunks)


def test_para_end_tracks_last_merged_label():
    chunks = chunk.build_chunks("t5", _pages(SAMPLE))
    assert all("para_end" in c for c in chunks)
    spans = [c for c in chunks if c["para_end"] != c["para"]]
    assert spans, "expected at least one chunk merging several labels"
    assert all(c["para_end"] for c in spans)


def test_oversized_paragraph_splits():
    sentence = "The entity shall comply with the requirements of these Directions at all times. "
    big = "9. " + sentence * 90  # ~1100 words, no sub-clauses
    chunks = chunk.build_chunks("t2", _pages(big))
    assert len(chunks) >= 2
    assert all(c["n_words"] <= chunk.HARD_MAX_WORDS + 60 for c in chunks)


def test_empty_input():
    assert chunk.build_chunks("t3", [{"page": 1, "text": "   \n  \n"}]) == []


def test_blank_line_preserved_as_paragraph_break():
    text = "1. First paragraph of the unit with enough words to be kept.\n\nSecond paragraph of the same unit also kept together."
    chunks = chunk.build_chunks("t4", _pages(text))
    assert chunks and "\n\n" in chunks[0]["text"]
