"""
Ensemble methods — Ridge Stacking and Weighted Averaging.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def build_ensemble(
    y: np.ndarray,
    lgb_oof: np.ndarray,
    xgb_oof: np.ndarray,
    cat_oof: np.ndarray,
    lgb_preds: np.ndarray,
    xgb_preds: np.ndarray,
    cat_preds: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    """Find the best ensemble strategy (Ridge stacking vs Weighted Average).

    Returns
    -------
    tuple
        ``(final_predictions, final_oof_r2, ensemble_method_name)``
    """
    # ── 1. Ridge Stacking ────────────────────────────────────
    oof_stack = np.column_stack([lgb_oof, xgb_oof, cat_oof])
    test_stack = np.column_stack([lgb_preds, xgb_preds, cat_preds])

    meta = Ridge(alpha=1.0)
    meta.fit(oof_stack, y)
    stacked_oof = meta.predict(oof_stack)
    stacked_preds = meta.predict(test_stack)

    stacked_r2 = r2_score(y, stacked_oof)
    print(f"  Stacked OOF R2 = {stacked_r2:.6f}")
    print(f"  Ridge weights: {meta.coef_}")

    # ── 2. Weighted Average Grid Search ──────────────────────
    best_r2 = -1
    best_w = (1 / 3, 1 / 3, 1 / 3)

    for w1 in np.arange(0.1, 0.8, 0.05):
        for w2 in np.arange(0.1, 0.8 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            blend = w1 * lgb_oof + w2 * xgb_oof + w3 * cat_oof
            r2 = r2_score(y, blend)
            if r2 > best_r2:
                best_r2 = r2
                best_w = (w1, w2, w3)

    print(f"  Best weighted avg R2 = {best_r2:.6f}")
    print(
        f"  Weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CAT={best_w[2]:.2f}"
    )

    # ── Select Best ──────────────────────────────────────────
    if stacked_r2 > best_r2:
        final_preds = stacked_preds
        final_r2 = stacked_r2
        method = "Ridge Stacking"
    else:
        final_preds = (
            best_w[0] * lgb_preds
            + best_w[1] * xgb_preds
            + best_w[2] * cat_preds
        )
        final_r2 = best_r2
        method = f"Weighted Average ({best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f})"

    return final_preds, final_r2, method
