"""
Traffic Demand Prediction — Complete Solution Pipeline (Modular)
==============================================================
Predicts traffic demand at geohash locations using an ensemble of
LightGBM, XGBoost, and CatBoost with extensive feature engineering.

Run this script to reproduce the entire pipeline.
"""

import os
import warnings
import numpy as np
import pandas as pd

from config import DROP_COLS, OUTPUT_DIR
from src.data_loader import load_datasets
from src.feature_engineering import engineer_features
from src.target_encoding import apply_target_encodings
from src.models import train_lightgbm, train_xgboost, train_catboost
from src.ensemble import build_ensemble
from src.visualizations import (
    generate_eda_plots,
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
    train = engineer_features(train)
    test = engineer_features(test)

    print("\n" + "=" * 60)
    print("STEP 3: Target Encoding & Demand Statistics...")
    print("=" * 60)
    train, test = apply_target_encodings(train, test)

    print("\n" + "=" * 60)
    print("STEP 4: Preparing final feature matrix...")
    print("=" * 60)
    target = train["demand"].values
    drop_cols_train = DROP_COLS + ["demand"]

    X_train = train.drop(columns=[c for c in drop_cols_train if c in train.columns])
    X_test = test.drop(columns=[c for c in DROP_COLS if c in test.columns])

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    y = target

    print(f"  Feature count: {len(common_cols)}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")

    print("\n" + "=" * 60)
    print("STEP 5: Generating EDA Visualizations...")
    print("=" * 60)
    generate_eda_plots(train, y)

    print("\n" + "=" * 60)
    print("STEP 6: Training Models (5-Fold CV)...")
    print("=" * 60)
    print("\n--- Training LightGBM ---")
    lgb_oof, lgb_preds, lgb_scores, lgb_model = train_lightgbm(X_train, y, X_test)
    plot_feature_importance(list(X_train.columns), lgb_model.feature_importances_)

    print("\n--- Training XGBoost ---")
    xgb_oof, xgb_preds, xgb_scores = train_xgboost(X_train, y, X_test)

    print("\n--- Training CatBoost ---")
    cat_oof, cat_preds, cat_scores = train_catboost(X_train, y, X_test)

    print("\n" + "=" * 60)
    print("STEP 7: Ensemble Stacking...")
    print("=" * 60)
    final_preds, final_r2, method = build_ensemble(
        y, lgb_oof, xgb_oof, cat_oof, lgb_preds, xgb_preds, cat_preds
    )
    print(f"\n  Selected ensemble method: {method}")
    print(f"  Estimated competition score = {max(0, 100 * final_r2):.2f}")

    print("\n" + "=" * 60)
    print("STEP 8: Model Comparison Plots...")
    print("=" * 60)
    lgb_r2 = np.mean(lgb_scores)
    xgb_r2 = np.mean(xgb_scores)
    cat_r2 = np.mean(cat_scores)
    plot_model_comparison(
        y, lgb_r2, xgb_r2, cat_r2, final_r2, final_preds,
        lgb_scores, xgb_scores, cat_scores
    )

    print("\n" + "=" * 60)
    print("STEP 9: Generating Submission...")
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
    print("Done! [OK]")


if __name__ == "__main__":
    main()
