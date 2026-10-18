"""Dense retrieval: sentence-transformers encoder + FAISS IndexFlatIP search
with an exact NumPy fallback.

Vectors are stored float16 (.npy) and searched in float32. The FAISS flat
index (exact inner product over L2-normalised vectors = cosine similarity) is
built in memory at load — at RBI-corpus scale the build is milliseconds, so
no persisted index file is needed and the .npy stays the canonical artifact.
The NumPy path over the same matrix is byte-identical, so environments
without faiss (and audits that force it) see the same neighbours. Set
REGINTEL_DENSE_BACKEND=numpy to force the fallback.

Masked search (institution / doc-type / date filters) searches the full flat
index and filters afterwards — still exact, and at this scale ~1 ms.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from regintel import config

log = logging.getLogger(__name__)


class IndexMismatchError(RuntimeError):
    """The dense vectors on disk do not belong to the current chunks.jsonl."""


def _chunk_key(c: dict) -> str:
    # chunk ids are positional (doc::0007), so a rechunk with unchanged counts
    # keeps every id; the text hash is what notices the content moved.
    return f"{c['chunk_id']}|{c.get('text_hash') or hashlib.sha256(c['text'].encode('utf-8')).hexdigest()[:16]}"


def _digest(chunks: list[dict]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(_chunk_key(c).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def fingerprint(chunks: list[dict]) -> dict:
    """Alignment fingerprint stored in dense_meta.json: sha256 over the ordered
    (chunk_id, text_hash) pairs plus the length it covers, so a later, longer
    corpus can still be checked against its first ``n`` chunks."""
    return {"algo": "sha256(chunk_id|text_hash)", "n": len(chunks), "sha256": _digest(chunks)}


def prefix_matches(fp: dict | None, chunks: list[dict]) -> bool:
    """True when ``chunks[:fp.n]`` is exactly the corpus the vectors were built from."""
    if not fp or int(fp.get("n", -1)) > len(chunks) or int(fp.get("n", -1)) < 0:
        return False
    return _digest(chunks[: int(fp["n"])]) == fp.get("sha256")


def check_alignment(directory: Path, chunks: list[dict]) -> str:
    """'ok' | 'legacy' (no fingerprint stored: cannot verify) — raises
    IndexMismatchError when a stored fingerprint disagrees with chunks.jsonl."""
    meta = dense_meta(directory) or {}
    fp = meta.get("fingerprint")
    if not fp:
        return "legacy"
    if int(fp.get("n", -1)) != len(chunks) or not prefix_matches(fp, chunks):
        raise IndexMismatchError(
            f"dense index in {directory} was built for {fp.get('n')} chunks "
            f"(sha256 {str(fp.get('sha256'))[:12]}…) but chunks.jsonl has {len(chunks)} that "
            "differ — the corpus was rechunked or rewritten after indexing. Rebuild it: "
            "python scripts/build_index.py --full")
    return "ok"


def _forced_backend() -> str:
    return os.environ.get("REGINTEL_DENSE_BACKEND", "").strip().lower()


class Encoder:
    """Lazy sentence-transformers wrapper. `model` is a hub id or local path."""

    def __init__(self, model: str | Path | None = None, query_prefix: str | None = None,
                 device: str | None = None):
        self.model_name = str(model or config.EMBED_MODEL)
        prefix = config.EMBED_QUERY_PREFIX_OVERRIDES.get(self.model_name, config.EMBED_QUERY_PREFIX)
        self.query_prefix = prefix if query_prefix is None else query_prefix
        self.device = device if device is not None else config.EMBED_DEVICE
        self._m = None

    @property
    def model(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer

            try:
                # cached models load without hub round-trips (saves 5-45 s per load)
                self._m = SentenceTransformer(self.model_name, device=self.device,
                                              local_files_only=True)
            except Exception:  # not cached yet, or older ST without the flag
                self._m = SentenceTransformer(self.model_name, device=self.device)
            # per-model trained envelope (e.g. hinglish-st-33m was trained at 192)
            self._m.max_seq_length = config.EMBED_MAX_SEQ_OVERRIDES.get(self.model_name, 512)
        return self._m

    def encode_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        texts = [self.query_prefix + t for t in texts]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        order = np.argsort([-len(t) for t in texts])  # long first: uniform batches
        enc = self.model.encode([texts[i] for i in order], batch_size=batch_size,
                                normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)
        out = np.empty_like(enc)
        out[order] = enc
        return out.astype(np.float32)


class DenseIndex:
    """Flat inner-product search over float16 vectors (FAISS, NumPy fallback)."""

    def __init__(self, vectors: np.ndarray, backend: str | None = None):
        self.vectors = vectors.astype(np.float16)
        self._f32: np.ndarray | None = None
        self.backend = "numpy"
        self._faiss = None
        want = (backend or _forced_backend() or "auto").lower()
        if want in ("auto", "faiss"):
            try:
                import faiss

                index = faiss.IndexFlatIP(self.matrix.shape[1])
                index.add(np.ascontiguousarray(self.matrix))
                self._faiss = index
                self.backend = "faiss"
            except Exception as exc:  # faiss missing or broken -> NumPy search
                if want == "faiss":
                    log.warning("faiss requested but unavailable (%s); using NumPy search", exc)

    @property
    def matrix(self) -> np.ndarray:
        if self._f32 is None:
            self._f32 = self.vectors.astype(np.float32)
        return self._f32

    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    def search(self, qvec: np.ndarray, k: int = 100,
               mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-9)
        k = min(k, self.n)
        if k <= 0:
            return []
        if self._faiss is not None:
            try:
                # with a mask, search everything and filter (flat = exact)
                ks = self.n if mask is not None else k
                scores, ids = self._faiss.search(q.reshape(1, -1), ks)
                pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
                if mask is not None:
                    pairs = [(i, s) for i, s in pairs if mask[i]][:k]
                return pairs
            except Exception as exc:  # noqa: BLE001
                log.warning("faiss search failed (%s); using NumPy search", exc)
        scores = self.matrix @ q
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(int(i), float(scores[i])) for i in idx if np.isfinite(scores[i])]

    def save(self, directory: Path, meta: dict) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "dense_fp16.npy", self.vectors)
        (directory / "dense_meta.json").write_text(
            json.dumps({**meta, "n": int(self.n), "dim": int(self.vectors.shape[1]),
                        "storage": "flat_fp16"}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        return cls(np.load(Path(directory) / "dense_fp16.npy"))


def build_vectors(encoder: Encoder, texts: list[str]) -> np.ndarray:
    return encoder.encode_passages(texts)


def append_vectors(existing_path: Path, encoder: Encoder, new_texts: list[str],
                   expect_n: int | None = None) -> np.ndarray:
    """Encode only `new_texts` and append to the stored matrix.

    `expect_n` lets callers assert the stored matrix is exactly the prefix of
    the corpus that was already encoded; a mismatch falls back to a full encode
    by the caller.
    """
    old = np.load(existing_path) if Path(existing_path).exists() else None
    if old is None:
        return build_vectors(encoder, new_texts)
    if expect_n is not None and old.shape[0] != expect_n:
        raise ValueError(f"index length {old.shape[0]} != expected prefix {expect_n}")
    new = build_vectors(encoder, new_texts).astype(np.float16)
    return np.vstack([old, new])


def dense_meta(directory: Path) -> dict | None:
    p = Path(directory) / "dense_meta.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None
