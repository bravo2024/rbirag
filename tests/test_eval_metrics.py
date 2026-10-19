"""Pure IR metric helpers shared by the eval scripts."""
import math

from regintel.eval_metrics import hit_at_k, mrr_at_k, ndcg_at_k, pearson


def test_hit_and_mrr_cutoffs():
    docs = ["a", "b", "c"]
    assert hit_at_k(docs, {"b"}, 2)
    assert not hit_at_k(docs, {"b"}, 1)
    assert not hit_at_k(docs, {"z"}, 3)
    assert mrr_at_k(docs, {"b"}, 3) == 0.5
    assert mrr_at_k(docs, {"z"}, 3) == 0.0


def test_ndcg_binary_gains():
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0                  # gold at rank 1
    assert abs(ndcg_at_k(["x", "a"], {"a"}, 2) - 1 / math.log2(3)) < 1e-9
    assert ndcg_at_k(["b", "a"], {"a", "b"}, 2) == 1.0             # both, ideal order
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) < 1.0         # one displaced
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 1) == 1.0             # k truncates ideal too
    # repeated passages of the same gold doc must not double-count
    assert ndcg_at_k(["a", "a", "a"], {"a"}, 3) == 1.0
    assert ndcg_at_k(["a", "b", "b"], {"a", "b"}, 3) == 1.0


def test_pearson():
    assert abs(pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12
    assert abs(pearson([1, 2, 3], [6, 4, 2]) + 1.0) < 1e-12
    assert pearson([1], [1]) is None
    assert pearson([1, 1], [1, 2]) is None      # zero variance


def test_judge_agreement_excludes_pairs_served_by_the_same_model():
    from regintel.eval_metrics import judge_agreement
    a, b = {"gemini/lite": 3}, {"gemini/3.7": 3}
    # record 0: different models; record 1: same model served both; record 2: missing score
    out = judge_agreement([1.0, 0.5, None, 0.2], [a, a, a, a], [0.5, 0.5, 1.0, 0.4], [b, a, b, b])
    assert out["n"] == 2 and out["n_same_model"] == 1 and out["n_pairs_scored"] == 3
    assert out["mean_abs_delta"] == 0.35
    assert "excluded" in out["note"]


def test_judge_agreement_not_computed_when_same_model_throughout():
    from regintel.eval_metrics import judge_agreement
    a = {"gemini/lite": 3}
    out = judge_agreement([1.0, 0.8], [a, a], [1.0, 0.8], [a, a])
    assert out["n"] == 0 and out["pearson"] is None and out["mean_abs_delta"] is None
    assert out["note"] == "not computed: same model served both judges (2/2 scored pairs)"
