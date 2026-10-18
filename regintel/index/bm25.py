"""BM25 over a sparse weight matrix, with a tokenizer tuned for legal text.

Kept whole as tokens: dotted paragraph numbers (5.2.1), circular numbers
(DOR.STR.REC.51/21.04.048/2024-25), RBI/2023-24/52 references and financial-year
ranges — the identifiers practitioners actually search by.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

STOP = frozenset(
    "a an and are as at be been but by for from has have had if in into is it its of on or such "
    "than that the their them then there these they this those to was were which will with would "
    "shall may can any all per also not no under".split())

_CIRCULAR = re.compile(r"\b[A-Za-z]{2,}(?:\.[A-Za-z]{1,}){1,}\.?[\w./-]*\d[\w./-]*")
_RBI_REF = re.compile(r"\bRBI/\d{4}-\d{2,4}/\d+", re.I)
_DOTTED = re.compile(r"\b\d+(?:\.\d+)+(?:\([a-z0-9]{1,4}\))*")
_YEAR_RANGE = re.compile(r"\b\d{4}-\d{2,4}\b")
_WORD = re.compile(r"[a-z0-9]+")


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    text = text.replace("\u20b9", " rupees ").replace("\u2019", "'")
    text = re.sub(r"(?<=\d),(?=\d{2,3}\b)", "", text)
    toks: list[str] = []
    for rx in (_CIRCULAR, _RBI_REF):
        for m in rx.finditer(text):
            toks.append(m.group(0).rstrip(".,;:)").lower())
    for m in _YEAR_RANGE.finditer(text):
        toks.append(m.group(0))
    dotted = []
    for m in _DOTTED.finditer(text):
        t = m.group(0).lower()
        toks.append(t)
        base = t.split("(")[0]
        if base != t:
            toks.append(base)
        dotted.append(m.span())
    if dotted:  # "5.2.1" should not also yield 5, 2, 1
        buf, last = [], 0
        for s, e in dotted:
            buf.append(text[last:s])
            last = e
        buf.append(text[last:])
        text = " ".join(buf)
    for w in _WORD.findall(text.lower()):
        if w in STOP or (len(w) == 1 and not w.isdigit()):
            continue
        toks.append(_stem(w))
    return toks


class BM25Index:
    def __init__(self, weights: sp.csc_matrix, vocab: dict[str, int], k1: float, b: float):
        self.W = weights.tocsc()
        self.vocab = vocab
        self.k1 = k1
        self.b = b

    @classmethod
    def build(cls, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        dl = np.zeros(len(docs_tokens), dtype=np.float32)
        for i, toks in enumerate(docs_tokens):
            dl[i] = len(toks)
            counts: dict[int, int] = {}
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(i)
                cols.append(j)
                tfs.append(c)
        n = len(docs_tokens)
        rows_a = np.asarray(rows, dtype=np.int32)
        cols_a = np.asarray(cols, dtype=np.int32)
        tf = np.asarray(tfs, dtype=np.float32)
        df = np.bincount(cols_a, minlength=len(vocab)).astype(np.float32)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = max(float(dl.mean()), 1e-9)
        denom = tf + k1 * (1 - b + b * dl[rows_a] / avgdl)
        w = idf[cols_a] * tf * (k1 + 1) / denom
        mat = sp.csc_matrix((w.astype(np.float32), (rows_a, cols_a)), shape=(n, len(vocab)))
        return cls(mat, vocab, k1, b)

    @property
    def n_docs(self) -> int:
        return self.W.shape[0]

    def scores(self, query: str | list[str]) -> np.ndarray:
        toks = tokenize(query) if isinstance(query, str) else query
        out = np.zeros(self.n_docs, dtype=np.float32)
        for t in set(toks):
            j = self.vocab.get(t)
            if j is None:
                continue
            s, e = self.W.indptr[j], self.W.indptr[j + 1]
            out[self.W.indices[s:e]] += self.W.data[s:e]
        return out

    def search(self, query: str, k: int = 100, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        sc = self.scores(query)
        if mask is not None:
            sc = np.where(mask, sc, 0.0)
        k = min(k, len(sc))
        if k <= 0:
            return []
        idx = np.argpartition(-sc, k - 1)[:k]
        idx = idx[np.argsort(-sc[idx], kind="stable")]
        return [(int(i), float(sc[i])) for i in idx if sc[i] > 0]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sp.save_npz(directory / "bm25.npz", self.W, compressed=True)
        (directory / "bm25_vocab.json").write_text(
            json.dumps({"k1": self.k1, "b": self.b, "vocab": self.vocab}, separators=(",", ":")),
            encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        directory = Path(directory)
        meta = json.loads((directory / "bm25_vocab.json").read_text(encoding="utf-8"))
        weights = sp.load_npz(directory / "bm25.npz")
        return cls(weights, meta["vocab"], float(meta["k1"]), float(meta["b"]))
