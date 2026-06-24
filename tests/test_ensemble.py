"""Tests for src/ensemble.py stacking invariants on synthetic OOF arrays.

Calls the underlying _slsqp_oof_preds / _ridge_oof_preds / _stack_xgb directly
(rather than build_ensemble) to avoid method-selection flakiness.
"""
import numpy as np

from src.ensemble import (
    _slsqp_oof_preds,
    _ridge_oof_preds,
    _stack_xgb,
    build_ensemble,
)


def _make_synthetic_oofs(n=500, seed=42):
    """Build 4 correlated OOF arrays with a known linear signal.

    Target = 0.4*m1 + 0.3*m2 + 0.2*m3 + 0.1*m4 + small noise.
    Ensembles should be able to recover positive weights on m1..m4.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    # Each model is the true linear combo plus per-model noise.
    coefs = np.array([0.4, 0.3, 0.2, 0.1])
    noise_scale = np.array([0.05, 0.08, 0.10, 0.12])
    oofs = X * coefs + rng.normal(scale=noise_scale, size=(n, 4))
    y = X @ coefs + rng.normal(scale=0.01, size=n)
    # Test stack mirrors oof shape.
    test = rng.normal(size=(100, 4)) * coefs + rng.normal(scale=noise_scale, size=(100, 4))
    return y, oofs, test


def test_slsqp_weights_sum_to_one():
    y, oofs, test = _make_synthetic_oofs()
    _, _, _, weights = _slsqp_oof_preds(oofs, y, test)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6), (
        f"SLSQP weights sum = {weights.sum()}, expected 1.0"
    )


def test_slsqp_weights_non_negative_and_bounded():
    y, oofs, test = _make_synthetic_oofs()
    _, _, _, weights = _slsqp_oof_preds(oofs, y, test)
    assert np.all(weights >= -1e-9), f"SLSQP weights < 0: {weights}"
    assert np.all(weights <= 1.0 + 1e-9), f"SLSQP weights > 1: {weights}"


def test_ridge_coefficients_non_negative():
    y, oofs, test = _make_synthetic_oofs()
    _, _, _, coefs = _ridge_oof_preds(oofs, y, test)
    assert np.all(coefs >= -1e-9), f"Ridge coefs < 0: {coefs}"


def test_xgb_stacker_r2_within_sanity_bound():
    """XGB stacker R^2 must not be wildly worse than the best single model."""
    y, oofs, test = _make_synthetic_oofs()
    individual_r2 = [1 - np.var(y - oofs[:, i]) / np.var(y) for i in range(oofs.shape[1])]
    best_single = max(individual_r2)
    _, _, stacker_r2 = _stack_xgb(oofs, y, test)
    # Sanity bound: stacker shouldn't be 0.15 worse than best single model.
    assert stacker_r2 >= best_single - 0.15, (
        f"XGB stacker R2={stacker_r2:.4f} vs best single R2={best_single:.4f}"
    )


def test_build_ensemble_returns_valid_method_and_weights():
    y, oofs, test = _make_synthetic_oofs()
    names = ["m1", "m2", "m3", "m4"]
    _, _, _, method, coefs = build_ensemble(y, [oofs[:, i] for i in range(4)], [test[:, i] for i in range(4)], names)
    assert method in ("SLSQP", "Ridge+", "XGBStack", "EqualMean"), f"unexpected method {method}"
    # Final coefs shape matches n_models.
    assert coefs.shape == (4,)