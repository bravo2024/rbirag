"""Dense-index alignment guard: fingerprint, safe incremental rebuild, load check."""
import hashlib
import json
import logging

import numpy as np
import pytest

from regintel import config, ingest, util
from regintel.index import fusion
from regintel.index.dense import (IndexMismatchError, check_alignment, dense_meta, fingerprint,
                                  prefix_matches)


def _chunk(i: int, text: str | None = None) -> dict:
    text = text or f"passage number {i} about capital adequacy and provisioning norms"
    return {"chunk_id": f"d{i // 3}::{i % 3:04d}", "doc_id": f"d{i // 3}", "text": text,
            "text_hash": util.sha256_text(text)[:16]}


class FakeEncoder:
    """Deterministic 8-d unit vectors from the text; records what it encoded."""
    model_name = "fake-encoder"
    encoded: list[str] = []

    def encode_passages(self, texts):
        FakeEncoder.encoded.extend(texts)
        out = []
        for t in texts:
            raw = np.frombuffer(hashlib.sha256(t.encode()).digest()[:8], dtype=np.uint8)
            v = raw.astype(np.float32) + 1.0
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    monkeypatch.setattr(config, "CHUNKS", tmp_path / "chunks.jsonl")
    monkeypatch.setattr(config, "DOCUMENTS", tmp_path / "documents.jsonl")
    monkeypatch.setattr(config, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(ingest, "Encoder", FakeEncoder)
    FakeEncoder.encoded = []

    def write(chunks):
        config.CHUNKS.write_text("\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8")

    return write


def _rebuild(**kw):
    FakeEncoder.encoded = []
    ingest.rebuild_indexes(progress=lambda *_: None, **kw)
    return list(FakeEncoder.encoded)


def test_fingerprint_matches_prefix_and_detects_same_count_rewrite():
    base = [_chunk(i) for i in range(6)]
    fp = fingerprint(base)
    assert prefix_matches(fp, base)
    assert prefix_matches(fp, base + [_chunk(6)])                    # corpus only grew
    rewritten = base[:3] + [_chunk(3, "rechunked: same id, different text")] + base[4:]
    assert [c["chunk_id"] for c in rewritten] == [c["chunk_id"] for c in base]
    assert not prefix_matches(fp, rewritten)                          # ids equal, text moved
    assert not prefix_matches(fp, base[:4])                           # shorter than the index
    assert not prefix_matches(None, base)


def test_rebuild_reuses_then_appends_only_when_aligned(corpus):
    chunks = [_chunk(i) for i in range(6)]
    corpus(chunks)
    assert len(_rebuild()) == 6                                       # first build: everything
    assert dense_meta(config.INDEX_DIR)["fingerprint"]["n"] == 6
    assert _rebuild() == []                                           # unchanged: reused
    corpus(chunks + [_chunk(6), _chunk(7)])
    assert _rebuild() == [_chunk(6)["text"], _chunk(7)["text"]]       # appended: new ones only
    assert dense_meta(config.INDEX_DIR)["fingerprint"]["n"] == 8
    assert np.load(config.INDEX_DIR / "dense_fp16.npy").shape[0] == 8


def test_rebuild_full_encodes_after_same_count_rechunk(corpus):
    chunks = [_chunk(i) for i in range(6)]
    corpus(chunks)
    _rebuild()
    rechunked = chunks[:2] + [_chunk(2, "moved boundary: new text under the same id")] + chunks[3:]
    corpus(rechunked)
    encoded = _rebuild()
    assert encoded == [c["text"] for c in rechunked]                  # all 6, not 0 reused
    vecs = np.load(config.INDEX_DIR / "dense_fp16.npy")
    expect = FakeEncoder().encode_passages([rechunked[2]["text"]])[0]
    assert np.allclose(vecs[2].astype(np.float32), expect, atol=1e-2)  # row 2 is the new text


def test_rebuild_full_encodes_legacy_index_without_fingerprint(corpus):
    corpus([_chunk(i) for i in range(4)])
    _rebuild()
    meta = dense_meta(config.INDEX_DIR)
    meta.pop("fingerprint")
    (config.INDEX_DIR / "dense_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert len(_rebuild()) == 4                                       # cannot verify -> re-encode
    assert "fingerprint" in dense_meta(config.INDEX_DIR)


def test_rebuild_full_encodes_when_encoder_changed(corpus, monkeypatch):
    chunks = [_chunk(i) for i in range(4)]
    corpus(chunks)
    _rebuild()
    monkeypatch.setattr(FakeEncoder, "model_name", "other-encoder")
    encoded = _rebuild()
    assert encoded == [c["text"] for c in chunks]                     # no cross-space reuse
    assert dense_meta(config.INDEX_DIR)["encoder"] == "other-encoder"


def test_engine_query_encoder_follows_dense_meta(corpus, monkeypatch):
    corpus([_chunk(i) for i in range(4)])
    _rebuild()
    seen = {}

    class CaptureEncoder:
        def __init__(self, model=None):
            seen["model"] = model

    monkeypatch.setattr(config, "EMBED_MODEL", "different-default")
    monkeypatch.setattr(fusion, "Encoder", CaptureEncoder)
    fusion.Engine()
    assert seen["model"] == "fake-encoder"


def test_engine_load_raises_on_mismatch_and_warns_on_legacy(corpus, caplog):
    chunks = [_chunk(i) for i in range(6)]
    corpus(chunks)
    _rebuild()
    assert fusion.Engine().ready                                      # aligned: loads fine
    corpus(chunks[:2] + [_chunk(2, "rechunked text")] + chunks[3:])   # same count, new text
    with pytest.raises(IndexMismatchError, match="build_index.py --full"):
        fusion.Engine()
    # legacy index (no fingerprint): warn only
    meta = dense_meta(config.INDEX_DIR)
    meta.pop("fingerprint")
    (config.INDEX_DIR / "dense_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="regintel.index.fusion"):
        assert fusion.Engine().ready
    assert "no alignment fingerprint" in caplog.text
    assert check_alignment(config.INDEX_DIR, chunks) == "legacy"
