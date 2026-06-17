"""normalize_semantic: bounds, clipping, and the NumPy-2 degenerate-range regression."""

import numpy as np

from lighthouse.scoring import normalize_semantic, raw_semantic_fit


def test_explicit_bounds_affine_map():
    out = normalize_semantic(np.array([0.0, 5.0, 10.0]), lo=0.0, hi=10.0)
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_out_of_bounds_values_are_clipped():
    out = normalize_semantic(np.array([-5.0, 5.0, 15.0]), lo=0.0, hi=10.0)
    assert np.allclose(out, [0.0, 0.5, 1.0])
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_empty_input_returns_empty():
    out = normalize_semantic(np.array([]))
    assert out.size == 0


def test_degenerate_constant_does_not_crash():
    # Regression guard: a near-constant batch drives hi-lo < 1e-6, which previously
    # used the ndarray.ptp() method removed in NumPy 2.0 and raised AttributeError.
    # Must not raise, and must stay within [0, 1].
    out = normalize_semantic(np.full(8, 0.5))
    assert out.shape == (8,)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_raw_semantic_fit_top_k_lifts_specialist_over_shallow_generalist():
    """A specialist (3 strong + 7 weak facets) should outscore a generalist
    that ties the specialist's max but has shallow, broad coverage. Under the
    old mean-over-all-facets aggregation the generalist won because their
    average was higher; the top-K aggregation rewards depth where it counts."""
    facet_emb = np.eye(10, dtype=np.float32)  # one orthogonal facet per axis
    specialist = np.array([[0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    generalist = np.array(
        [[0.9, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]], dtype=np.float32
    )
    s = raw_semantic_fit(specialist, facet_emb)[0]
    g = raw_semantic_fit(generalist, facet_emb)[0]
    assert s > g


def test_raw_semantic_fit_degenerates_to_mean_when_k_exceeds_facets():
    """A rubric with fewer than TOP_K_FACETS facets must not crash — the top-K
    aggregation should degrade cleanly to a plain mean."""
    facet_emb = np.eye(2, dtype=np.float32)
    cand = np.array([[0.8, 0.4]], dtype=np.float32)
    out = raw_semantic_fit(cand, facet_emb)
    expected = 0.6 * 0.8 + 0.4 * 0.6  # max=0.8, mean=0.6
    assert abs(float(out[0]) - expected) < 1e-6


def test_inbatch_fallback_is_monotonic():
    raw = np.array([0.1, 0.2, 0.3, 0.9])
    out = normalize_semantic(raw)  # no bounds -> in-batch percentile path
    assert np.all(np.diff(out) >= 0.0)  # order-preserving transform
    assert out.min() >= 0.0 and out.max() <= 1.0
