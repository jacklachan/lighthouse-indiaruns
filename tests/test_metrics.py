"""Metric correctness against hand-computed values."""
import math

from lighthouse import metrics


def test_ndcg_perfect_order_is_one():
    rels = [5, 4, 3, 2, 1, 0]
    assert metrics.ndcg_at_k(rels, 6) == 1.0


def test_ndcg_reversed_is_low():
    perfect = metrics.ndcg_at_k([5, 4, 3, 2, 1], 5)
    reversed_ = metrics.ndcg_at_k([1, 2, 3, 4, 5], 5)
    assert perfect == 1.0
    assert reversed_ < 0.8


def test_dcg_known_value():
    # rels [3,2] -> (2^3-1)/log2(2) + (2^2-1)/log2(3) = 7/1 + 3/1.585
    expected = 7.0 + 3.0 / math.log2(3)
    assert abs(metrics.dcg_at_k([3, 2], 2) - expected) < 1e-9


def test_precision_at_k_threshold():
    rels = [5, 3, 2, 4, 0]  # tier>=3 relevant -> ranks 1,2,4
    assert metrics.precision_at_k(rels, 5) == 3 / 5
    assert metrics.precision_at_k(rels, 2) == 1.0


def test_average_precision_known():
    # relevant at ranks 1 and 3 (tier>=3); R=2
    # AP = (1/1 + 2/3) / 2
    rels = [5, 0, 4, 0]
    expected = (1.0 + 2 / 3) / 2
    assert abs(metrics.average_precision(rels) - expected) < 1e-9


def test_average_precision_no_relevant_is_zero():
    assert metrics.average_precision([0, 1, 2, 0]) == 0.0


def test_composite_in_unit_range():
    rels = [5, 4, 5, 3, 4, 2, 3, 1, 0, 3] + [0] * 40
    out = metrics.evaluate_ranking(rels)
    assert 0.0 <= out["composite"] <= 1.0
    assert set(out) >= {"NDCG@10", "NDCG@50", "MAP", "P@10", "composite"}
