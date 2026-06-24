"""
Ensemble methods - Ridge stacking, SLSQP weight tuning, weighted averaging,
and a 2nd-level XGBoost stacker.

All stacking methods are updated to run out-of-fold using cross-validation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import xgboost as xgb

from config import STACKER_PARAMS, SEED, N_SLSQP_RESTARTS

np.random.seed(SEED)   # reproducible SLSQP dirichlet random restarts (B1)


def _slsqp_weights(oofs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find non-negative weights summing to 1 that maximize OOF R^2."""
    n_models = oofs.shape[1]

    def neg_r2(w):
        w = np.clip(w, 0, None)
        s = w.sum()
        if s == 0:
            return 0.0
        w = w / s
        blend = oofs @ w
        return -r2_score(y, blend)

    best = None
    for _ in range(N_SLSQP_RESTARTS):
        x0 = np.random.dirichlet(np.ones(n_models))
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bounds = [(0.0, 1.0)] * n_models
        res = minimize(neg_r2, x0=x0, bounds=bounds, constraints=cons,
                       method="SLSQP", options={"maxiter": 200, "ftol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    w = np.clip(best.x, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(n_models) / n_models
    return w


def _oof_stack_predict(
    fit_fn,
    oof_stack: np.ndarray,
    y: np.ndarray,
    test_stack: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Generic out-of-fold stack predictor — shared KFold+accumulation pattern.

    Parameters
    ----------
    fit_fn : callable ``(X_tr, y_tr) -> (model, coefs_array)``.
        ``coefs_array`` is whatever per-fit coefficients the caller wants
        averaged (e.g. ``Ridge.coef_``); pass ``np.zeros(n_models)`` if N/A.
    oof_stack : ``(n_samples, n_models)`` meta-features from base models.
    y : target vector.
    test_stack : ``(n_test, n_models)`` test meta-features.
    n_splits : inner CV fold count.

    Returns
    -------
    tuple
        ``(meta_oof, meta_test, r2, mean_coefs)``.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y))
    meta_test = np.zeros(len(test_stack))
    coefs_list: list[np.ndarray] = []

    for tr, va in kf.split(oof_stack):
        model, coefs = fit_fn(oof_stack[tr], y[tr])
        meta_oof[va] = model.predict(oof_stack[va])
        meta_test += model.predict(test_stack) / n_splits
        coefs_list.append(coefs)

    return meta_oof, meta_test, r2_score(y, meta_oof), np.mean(coefs_list, axis=0)


def _slsqp_oof_preds(
    oof_stack: np.ndarray, y: np.ndarray, test_stack: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Train SLSQP weight blender out-of-fold and predict on test."""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    meta_oof = np.zeros(len(y))
    meta_test = np.zeros(len(test_stack))
    weights_list = []

    for tr, va in kf.split(oof_stack):
        w = _slsqp_weights(oof_stack[tr], y[tr])
        meta_oof[va] = oof_stack[va] @ w
        meta_test += (test_stack @ w) / 5.0
        weights_list.append(w)

    r2 = r2_score(y, meta_oof)
    mean_weights = np.mean(weights_list, axis=0)
    mean_weights = mean_weights / mean_weights.sum() if mean_weights.sum() > 0 else mean_weights
    return meta_oof, meta_test, r2, mean_weights


def _ridge_oof_preds(
    oof_stack: np.ndarray, y: np.ndarray, test_stack: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Train Ridge stacker out-of-fold and predict on test."""
    def _fit(X_tr, y_tr):
        model = Ridge(alpha=1.0, positive=True)
        model.fit(X_tr, y_tr)
        return model, model.coef_

    return _oof_stack_predict(_fit, oof_stack, y, test_stack)


def _stack_xgb(
    oof_stack: np.ndarray, y: np.ndarray, test_stack: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train an XGB stacker with internal CV on the meta-features.

    Note: ``_oof_stack_predict`` doesn't expose the inner val split, so this
    stacker passes the *training* fold as its eval_set. With
    ``STACKER_PARAMS['early_stopping_rounds']=30`` present, xgboost requires
    an eval_set; using the training fold disables early stopping (train loss
    never rises) but keeps the helper signature clean. Meta features are
    only 5 columns, so this is the right trade-off.
    """
    n_models = oof_stack.shape[1]

    def _fit(X_tr, y_tr):
        model = xgb.XGBRegressor(**STACKER_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr)], verbose=False)
        return model, np.zeros(n_models)

    meta_oof, meta_test, r2, _ = _oof_stack_predict(_fit, oof_stack, y, test_stack)
    return meta_oof, meta_test, r2


def build_ensemble(
    y: np.ndarray,
    oof_list: list[np.ndarray],
    test_list: list[np.ndarray],
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, float, str, np.ndarray]:
    """Build the best of {weighted blend, Ridge stack, XGB stack, SLSQP tune}.

    Returns
    -------
    tuple
        ``(final_oof, final_predictions, oof_r2, method_name, weights_or_coefs)``
    """
    oof_stack = np.column_stack(oof_list)
    test_stack = np.column_stack(test_list)
    n_models = oof_stack.shape[1]

    # --- 1. SLSQP-tuned weight blending (proper OOF)
    slsqp_oof, slsqp_test, slsqp_r2, w = _slsqp_oof_preds(oof_stack, y, test_stack)
    print(f"  SLSQP OOF R2    = {slsqp_r2:.6f}   weights={dict(zip(names, np.round(w, 3)))}")

    # --- 2. Ridge stacking (proper OOF)
    ridge_oof, ridge_test, ridge_r2, coefs = _ridge_oof_preds(oof_stack, y, test_stack)
    print(f"  Ridge OOF R2    = {ridge_r2:.6f}   coefs={dict(zip(names, np.round(coefs, 3)))}")

    # --- 3. 2nd-level XGB stacker (proper OOF)
    xgb_meta_oof, xgb_meta_test, xgb_meta_r2 = _stack_xgb(oof_stack, y, test_stack)
    print(f"  XGB stacker R2  = {xgb_meta_r2:.6f}")

    # --- 4. Simple average
    avg_oof = oof_stack.mean(axis=1)
    avg_test = test_stack.mean(axis=1)
    avg_r2 = r2_score(y, avg_oof)
    print(f"  Equal mean R2   = {avg_r2:.6f}")

    candidates = [
        ("SLSQP",     slsqp_r2, slsqp_oof, slsqp_test, w),
        ("Ridge+",    ridge_r2, ridge_oof, ridge_test, coefs),
        ("XGBStack",  xgb_meta_r2, xgb_meta_oof, xgb_meta_test, np.ones(n_models) / n_models),
        ("EqualMean", avg_r2, avg_oof, avg_test, np.ones(n_models) / n_models)
    ]
    candidates.sort(key=lambda c: c[1], reverse=True)
    name, r2, oof_preds, test_preds, final_w = candidates[0]
    print(f"  -> Selected: {name}  (OOF R2 {r2:.6f})")
    
    return oof_preds, test_preds, r2, name, final_w
