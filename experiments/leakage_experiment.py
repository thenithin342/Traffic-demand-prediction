"""
Leakage-magnitude experiment for src/trajectory.py.

Runs the full pipeline twice:
  (1) CV_MODE = "kfold"            — baseline (current production behaviour)
  (2) CV_MODE = "groupkfold_day"   — day-level groups, much harder split

Uses LightGBM only (the most-feature-sensitive model) and skips Optuna
/ plots / ensemble. Logs per-fold R2 and OOF R2 for each run, then
prints a delta and writes output/leakage_experiment.json.

Usage:
    python experiments/leakage_experiment.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Allow running from project root without installing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data.data_loader import load_datasets
from src.features.feature_engineering import engineer_features, FeatureState
from src.features.target_encoding import apply_static_target_encodings
from src.features.trajectory import build_trajectory_features
from src.features.neighbor import apply_static_neighbor_features
from src.models.models import _build_fold_cache, train_lightgbm

warnings.filterwarnings("ignore")


def run_one_mode(cv_mode: str) -> dict:
    """Run a single LightGBM CV pass under the given CV_MODE."""
    print(f"\n{'=' * 60}\nCV_MODE = {cv_mode}\n{'=' * 60}")
    config.CV_MODE = cv_mode
    t0 = time.time()

    train, test, _ = load_datasets()

    state = FeatureState()
    train = engineer_features(train, is_train=True, state=state)
    test = engineer_features(test, is_train=False, state=state)
    train, test = build_trajectory_features(train, test)
    train, test, neighbor_map = apply_static_neighbor_features(train, test, n_clusters=32)
    train, test = apply_static_target_encodings(train, test)

    fold_cache = _build_fold_cache(train, test, neighbor_map, model_types=("default",))["default"]

    target = train["demand"].values
    oof, preds, scores, _ = train_lightgbm(train, fold_cache)
    oof_r2 = r2_score(target, oof)
    elapsed = time.time() - t0

    print(f"\n  [{cv_mode}] OOF R2 = {oof_r2:.6f}, fold R2 = {[f'{s:.4f}' for s in scores]}")

    return {
        "cv_mode": cv_mode,
        "oof_r2": float(oof_r2),
        "fold_r2": [float(s) for s in scores],
        "elapsed_sec": float(elapsed),
    }


def main() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    results = {
        "kfold": run_one_mode("kfold"),
        "groupkfold_day": run_one_mode("groupkfold_day"),
    }

    base = results["kfold"]["oof_r2"]
    gk = results["groupkfold_day"]["oof_r2"]
    delta_abs = base - gk
    delta_pct = 100.0 * delta_abs / abs(base) if base else float("nan")

    results["delta"] = {
        "abs_r2_drop": float(delta_abs),
        "pct_r2_drop": float(delta_pct),
        "leakage_material": bool(delta_pct > 5.0),
    }

    out_path = os.path.join(config.OUTPUT_DIR, "leakage_experiment.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("LEAKAGE EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  KFold            OOF R2 = {base:.6f}")
    print(f"  GroupKFold(day)  OOF R2 = {gk:.6f}")
    print(f"  Delta R2 drop        = {delta_abs:+.6f}  ({delta_pct:+.2f}%)")
    print(f"  Material (>5%)?      = {results['delta']['leakage_material']}")
    print(f"  Results written to:  {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
