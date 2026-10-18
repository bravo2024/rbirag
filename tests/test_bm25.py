"""BM25 tokenizer + ranking over a toy corpus."""
import numpy as np

from regintel.index.bm25 import BM25Index, tokenize

DOCS = [
    "Every entity shall maintain records of transactions for five years.",
    "Know Your Customer norms for Non-Banking Financial Companies.",
    "Auction of Government of India Treasury Bills.",
]


def _index():
    return BM25Index.build([tokenize(d) for d in DOCS])


def test_tokenizer_keeps_identifiers():
    toks = tokenize("As per RBI/2026-27/293 and paragraph 5.2.1 of DOR.STR.REC.51/21.04.048/2024-25, "
                    "cyber incidents shall be reported within six hours.")
    assert "rbi/2026-27/293" in toks
    assert "5.2.1" in toks
    assert any(t.startswith("dor.str.rec.51") for t in toks)
    assert "six" in toks and "hour" in toks


def test_ranking():
    bm = _index()
    assert bm.search("five years records")[0][0] == 0
    assert bm.search("Know Your Customer")[0][0] == 1
    assert bm.search("treasury bills auction")[0][0] == 2


def test_mask_blocks_docs():
    bm = _index()
    mask = np.array([False, True, True])
    hits = bm.search("five years records", mask=mask)
    assert all(i != 0 for i, _ in hits)


def test_roundtrip(tmp_path):
    bm = _index()
    bm.save(tmp_path)
    again = BM25Index.load(tmp_path)
    assert again.n_docs == bm.n_docs
    assert again.search("five years records")[0][0] == 0
