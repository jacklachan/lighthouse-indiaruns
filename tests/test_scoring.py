"""normalize_semantic: bounds, clipping, and the NumPy-2 degenerate-range regression."""

import numpy as np

from lighthouse.scoring import normalize_semantic


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


def test_inbatch_fallback_is_monotonic():
    raw = np.array([0.1, 0.2, 0.3, 0.9])
    out = normalize_semantic(raw)  # no bounds -> in-batch percentile path
    assert np.all(np.diff(out) >= 0.0)  # order-preserving transform
    assert out.min() >= 0.0 and out.max() <= 1.0
