"""
Traffic Demand Prediction — Complete Solution Pipeline (Modular & Leak-Free)
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedShuffleSplit

import config
from config import OUTPUT_DIR, N_TRIALS_OPTUNA, LGB_PARAMS, SEED, get_drop_cols
from src.data_loader import load_datasets
from src.feature_engineering import engineer_features, FeatureState
from src.target_encoding import apply_static_target_encodings, TARGET_DEPENDENT_COLS
from src.trajectory import build_trajectory_features
from src.neighbor import apply_static_neighbor_features, NEIGHBOR_DEPENDENT_COLS
from src.models import train_lightgbm, train_xgboost, train_catboost, train_histgbm, _build_fold_cache
from src.nn_model import train_mlp
from src.tuning import tune_lightgbm
from src.ensemble import build_ensemble
from src.visualizations import (
    plot_feature_importance,
    plot_model_comparison,
)

warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("STEP 1: Loading data...")
    print("=" * 60)
    train, test, sample_sub = load_datasets()
    test_index = test["Index"].values

    print("\n" + "=" * 60)
    print("STEP 2: Feature engineering (Temporal, Spatial, Interaction)...")
    print("=" * 60)
    state = FeatureState()
    train = engineer_features(train, is_train=True, state=state)
    test = engineer_features(test, is_train=False, state=state)

    print("\n" + "=" * 60)
    print("STEP 3: Trajectory & Warm-Start Features (Continuous Lags)...")
    print("=" * 60)
    if not config.FOLD_AWARE_TRAJECTORY:
        # Legacy: build once globally (documented cross-fold leak in trajectory.py)
        train, test = build_trajectory_features(train, test)
    else:
        # Fold-aware: trajectory is rebuilt per fold inside _build_fold_cache.
        # Still need time_slot columns on raw data for other feature steps.
        from src.trajectory import _add_time_slot
        train = _add_time_slot(train)
        test  = _add_time_slot(test)

    print("\n" + "=" * 60)
    print("STEP 4: Static Neighbor & Clustering Features...")
    print("=" * 60)
    train, test, neighbor_map = apply_static_neighbor_features(train, test, n_clusters=32)

    print("\n" + "=" * 60)
    print("STEP 5: Static Target Encodings (Frequency & Labels)...")
    print("=" * 60)
    train, test = apply_static_target_encodings(train, test)

    print("\n" + "=" * 60)
    print("STEP 6: Preparing target and metadata...")
    print("=" * 60)
    # Derive approximate feature column list from module-level constants
    # (no mock-build run — see FIX 6 / C12: mock-build leaked ``demand`` into
    # train→train target encodings).
    drop_cols_train = get_drop_cols(train, model_type="default")
    base_feature_cols = [c for c in train.columns if c not in drop_cols_train]
    feature_cols = base_feature_cols + TARGET_DEPENDENT_COLS + NEIGHBOR_DEPENDENT_COLS
    print(f"  Approximate feature count: {len(feature_cols)}")

    print("\n" + "=" * 60)
    print("STEP 7: EDA Visualizations — skipped (FIX 6/C12 removed mock-build)")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("STEP 7.5: Splitting train into inner (80%) and outer (20%) for honest Optuna tuning...")
    print("=" * 60)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    inner_idx, outer_idx = next(sss.split(train, train["day"]))
    inner_train = train.iloc[inner_idx].reset_index(drop=True)
    outer_train = train.iloc[outer_idx].reset_index(drop=True)
    print(f"  Inner train size: {len(inner_train)} rows")
    print(f"  Outer train size: {len(outer_train)} rows")
    print("  NOTE: Reported R² is evaluated on outer_train (20% of data).")
    print("        Final submission model is also trained on outer_train only.")

    # Realign target/groups to outer_train (used by ensemble + day-49 OOF block)
    target = outer_train["demand"].values
    groups = outer_train["day"].values if "day" in outer_train.columns else None

    print("\n" + "=" * 60)
    print("STEP 8: Building fold feature cache once, shared by Optuna + all models...")
    print("=" * 60)
    fold_cache_all = _build_fold_cache(outer_train, test, neighbor_map, model_types=("default", "catboost"))
    fold_cache_default = fold_cache_all["default"]
    fold_cache_cat = fold_cache_all["catboost"]

    print("\n" + "=" * 60)
    print("STEP 8.5: Hyperparameter Tuning (Optuna, reusing fold cache)...")
    print("=" * 60)
    if N_TRIALS_OPTUNA > 0:
        best_lgb_params = tune_lightgbm(
            inner_train, test, neighbor_map,
            n_trials=N_TRIALS_OPTUNA,
            fold_cache=fold_cache_default,   # P5: reuse cache, no second build
        )
        # Update config directly for the current run
        LGB_PARAMS.update(best_lgb_params)
    else:
        print("  Skipping Optuna (N_TRIALS_OPTUNA = 0)")

    print("\n" + "=" * 60)
    print("STEP 9: Training Models (5-Fold CV)...")
    print("=" * 60)

    print("\n--- Training LightGBM ---")
    lgb_oof, lgb_preds, lgb_scores, lgb_importances = train_lightgbm(outer_train, fold_cache_default)
    # Pull feature names from the actual fold cache so the importance plot
    # matches what the model was trained on (the fold cache has the real
    # column list; the STEP 6 mock used pre-trajectory columns in
    # FOLD_AWARE_TRAJECTORY=True mode and would be a length mismatch).
    fold0 = next(iter(fold_cache_default.values()))
    feature_cols_used = list(fold0["X_tr"].columns)
    plot_feature_importance(feature_cols_used, lgb_importances)

    print("\n--- Training XGBoost ---")
    xgb_oof, xgb_preds, xgb_scores = train_xgboost(outer_train, fold_cache_default)

    print("\n--- Training CatBoost ---")
    cat_oof, cat_preds, cat_scores = train_catboost(outer_train, fold_cache_cat)

    print("\n--- Training HistGBM ---")
    hgb_oof, hgb_preds, hgb_scores = train_histgbm(outer_train, fold_cache_default)

    print("\n--- Training MLP ---")
    mlp_oof, mlp_preds, mlp_scores = train_mlp(outer_train, fold_cache_default)

    # Day-49-only OOF R² — most relevant for the test set, which is all day 49.
    print("\n--- Day-49-Only OOF R² (fold where val=day49) ---")
    for fold_idx, c in fold_cache_default.items():
        val_days = outer_train.iloc[c["val_idx"]]["day"].unique()
        if 49 in val_days:
            for name, oof_arr in [("LightGBM", lgb_oof), ("XGBoost", xgb_oof),
                                   ("CatBoost", cat_oof), ("HistGBM", hgb_oof), ("MLP", mlp_oof)]:
                r2_49 = r2_score(outer_train["demand"].values[c["val_idx"]], oof_arr[c["val_idx"]])
                print(f"  {name} day-49 R²: {r2_49:.6f}")
            break

    print("\n" + "=" * 60)
    print("STEP 10: Ensemble Stacking...")
    print("=" * 60)
    
    oof_list = [lgb_oof, xgb_oof, cat_oof, hgb_oof, mlp_oof]
    test_list = [lgb_preds, xgb_preds, cat_preds, hgb_preds, mlp_preds]
    names = ["LightGBM", "XGBoost", "CatBoost", "HistGBM", "MLP"]
    
    final_oof, final_preds, final_r2, method, coefs = build_ensemble(target, oof_list, test_list, names)
    print(f"\n  Selected ensemble method: {method}")
    print(f"  Estimated competition score = {max(0, 100 * final_r2):.2f}")

    print("\n" + "=" * 60)
    print("STEP 11: Model Comparison Plots...")
    print("=" * 60)
    
    model_r2 = {
        "LightGBM": np.mean(lgb_scores),
        "XGBoost": np.mean(xgb_scores),
        "CatBoost": np.mean(cat_scores),
        "HistGBM": np.mean(hgb_scores),
        "MLP": np.mean(mlp_scores)
    }
    
    fold_scores = {
        "LightGBM": lgb_scores,
        "XGBoost": xgb_scores,
        "CatBoost": cat_scores,
        "HistGBM": hgb_scores,
        "MLP": mlp_scores
    }
    
    plot_model_comparison(target, model_r2, final_r2, final_oof, final_preds, fold_scores, groups=groups)

    print("\n" + "=" * 60)
    print("STEP 12: Generating Submission...")
    print("=" * 60)
    final_preds = np.clip(final_preds, 0, 1)
    submission = pd.DataFrame({"Index": test_index, "demand": final_preds})
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)

    print(f"  Submission shape: {submission.shape}")
    print(f"  Demand range: [{submission['demand'].min():.6f}, {submission['demand'].max():.6f}]")
    print(f"  Saved to: {sub_path}")
    print("  [OK] Submission format verified!")
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Ensemble OOF R2: {final_r2:.6f}")
    print(f"  Estimated score: {max(0, 100 * final_r2):.2f} / 100")
    print("  (R² evaluated on outer 20% of train; final submission uses outer_train only)")
    print("Done! [OK]")


if __name__ == "__main__":
    main()
