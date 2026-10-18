"""FAISS and NumPy backends must return identical results (exact flat search)."""
from __future__ import annotations

import numpy as np
import pytest

from regintel.index.dense import DenseIndex


def _unit(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_faiss_matches_numpy_unmasked():
    pytest.importorskip("faiss")
    vecs = _unit(300, 64, 1)
    q = _unit(1, 64, 2)[0]
    idx_f = DenseIndex(vecs, backend="faiss")
    idx_n = DenseIndex(vecs, backend="numpy")
    assert idx_f.backend == "faiss" and idx_n.backend == "numpy"
    a = idx_f.search(q, k=20)
    b = idx_n.search(q, k=20)
    assert [i for i, _ in a] == [i for i, _ in b]
    assert np.allclose([s for _, s in a], [s for _, s in b], atol=1e-6)


def test_faiss_matches_numpy_masked():
    pytest.importorskip("faiss")
    vecs = _unit(300, 64, 3)
    q = _unit(1, 64, 4)[0]
    mask = np.zeros(300, dtype=bool)
    mask[10:60] = True
    idx_f = DenseIndex(vecs, backend="faiss")
    idx_n = DenseIndex(vecs, backend="numpy")
    a = idx_f.search(q, k=10, mask=mask)
    b = idx_n.search(q, k=10, mask=mask)
    assert [i for i, _ in a] == [i for i, _ in b]
    assert all(mask[i] for i, _ in a)


def test_env_forces_numpy(monkeypatch):
    monkeypatch.setenv("REGINTEL_DENSE_BACKEND", "numpy")
    idx = DenseIndex(_unit(50, 16, 5))
    assert idx.backend == "numpy"
