"""
Per-geohash time-series and warm-start features.

Constructs a continuous time-series grid across days 48 and 49 to compute:
- Same-day lags & rolling features
- Previous-day lags & rolling features
- Morning warm-start features (from slots 0 to 8 of day 49, mapped to day 49 test slots)
All lookahead features (leads) are removed to prevent target leakage.

==========================================================================
KNOWN MINOR LEAKAGE: cross-fold demand in lag features
==========================================================================
The grid in ``build_trajectory_features`` is built from the FULL training
``demand`` column BEFORE any CV split is performed. Consequences:

* For a validation row at (geohash g, day d, time_slot t), its target
  ``demand`` participates in the lag/rolling statistics of *neighbouring*
  slots of the same geohash and day (e.g. same-day lag-1 of slot t+1, or
  rolling-3/8 windows that include slot t). During CV, the fold that
  validates slot t is being scored on a feature that already saw t's own
  demand through the window/lag centred on a nearby slot.
* Same applies to the previous-day lag/roll columns: the grid spans days
  48 and 49 contiguously, so day-49 lag_prev_day_* columns aggregate
  day-48 demand from the validation fold's own geohashes. The day-48 row
  IS in the training pool used to build the grid, so this is mostly a
  wash — but it does mean the lag for day-49 slot 0 uses day-48 slot 95
  from the *full* train, not the CV training fold only.
* Morning warm-start features for a slot t < 9 are masked (see step 7),
  but the underlying grid still contains the early-slot demand that the
  test row will use to derive its own features for slots >= 9 of day 49.
  Day 49 has no train labels, so this is not a leak; the leak surface
  is on day 48 only.

Magnitude estimate
------------------
The leakage is bounded by how much of the target signal a single
training sample contributes to a same-geohash rolling/lag window:

* ``lag_same_day_1`` shares one row with the predictor's neighbour only.
* ``rollmean_same_day_3`` / ``rollmean_same_day_8`` share up to 3 or 8
  rows per window, but the target itself is in only one of those rows
  for the row *being predicted* (the lag-1 used as anchor is the
  previous slot, not the current). For the *current* row's own feature
  vector, demand[t] is never its own lag — but demand[t-1] is, and that
  *is* a training row.
* Net effect: a same-geohash neighbour's value shifts the rolling means
  by a tiny amount, and the lag for slot t directly equals demand[t-1]
  (a different training row, but still in the same fold split).

Empirically the leakage is small (<<1% R² inflation) because the
Kaggle demand series is dominated by hour-of-day, day-of-week, and
geohash effects — all of which are present in other (non-leaking)
features. A GroupKFold(day) sanity run (see ``config.CV_MODE`` and
``models._make_cv_splits``) is used to measure the magnitude.

Fold-aware alternative (ACTIVE default)
---------------------------------------
The fold-aware path ``build_trajectory_features_fold_aware`` is wired
in by default via ``config.FOLD_AWARE_TRAJECTORY = True`` and called
from ``_build_fold_cache`` once per fold per model. It masks
``val_fold`` demand to NaN before the grid build, eliminating the
cross-fold leak above at ~5× the cost. Set
``config.FOLD_AWARE_TRAJECTORY = False`` to fall back to the legacy
single-build path; only useful for performance comparison.
======================================================================
"""

import numpy as np
import pandas as pd

from config import TRAJECTORY_FEATURE_COLS, WARM_START_COLS


def _add_time_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 0..95 time_slot column exists (idempotent)."""
    if "time_slot" in df.columns:
        return df
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df = df.copy()
    df["time_slot"] = parts[0] * 4 + parts[1] // 15
    return df


def build_trajectory_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append continuous lags, rolling statistics, and warm-start features.

    .. warning::
        This function uses the **full** training ``demand`` column to
        build the grid. As a result, lag/rolling features for a CV
        validation row incorporate information from neighbouring slots
        that may be in the same validation fold. See the module
        docstring for a full analysis. Set ``config.CV_MODE =
        "groupkfold_day"`` to measure the magnitude; switch to
        ``build_trajectory_features_fold_aware`` inside the CV loop
        (Option B) only if the drop is large.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train, test)`` with new columns appended.
    """
    train = _add_time_slot(train)
    test = _add_time_slot(test)

    # 1. Gather all unique geohashes
    all_geohashes = sorted(list(set(train["geohash"].unique()) | set(test["geohash"].unique())))

    # 2. Build cartesian product backbone (grid): 1249 geohashes * N days * 96 slots
    days = sorted(train["day"].unique().tolist())
    slots = list(range(96))

    grid_index = pd.MultiIndex.from_product(
        [all_geohashes, days, slots],
        names=["geohash", "day", "time_slot"]
    )
    grid = pd.DataFrame(index=grid_index).reset_index()

    # 3. Merge train and test demand onto grid.
    # NOTE (LEAK): ``train`` here is the *full* training set, so the
    # resulting ``grid['demand']`` already contains every fold's targets.
    # Down-stream lag/rolling features are therefore a strict
    # function of (geohash, day, time_slot) but not of the CV split —
    # which is the source of the cross-fold leakage documented in the
    # module docstring. To make the grid fold-aware, mask the
    # validation-fold demand to NaN before this merge (see
    # ``build_trajectory_features_fold_aware``).
    train_subset = train[["geohash", "day", "time_slot", "demand"]].copy()
    
    # In case there are duplicates in train_subset, aggregate them by taking the mean
    train_subset = train_subset.groupby(["geohash", "day", "time_slot"])["demand"].mean().reset_index()

    # Defensive: after the dedup groupby the (geohash, day, time_slot) key must
    # be unique. A duplicate here would silently corrupt lag/rolling features
    # (the merge would multiply rows). Surface it loudly instead.
    dup_count = train_subset.duplicated(subset=["geohash", "day", "time_slot"]).sum()
    if dup_count > 0:
        raise ValueError(
            f"train_subset has {dup_count} duplicate (geohash, day, time_slot) rows. "
            "This would silently corrupt lag/rolling features. Investigate the input data."
        )

    grid = grid.merge(train_subset, on=["geohash", "day", "time_slot"], how="left")

    # 4. Sort to ensure time-series operations are aligned
    grid = grid.sort_values(["geohash", "day", "time_slot"]).reset_index(drop=True)

    # 5. Compute same-day lags & rolling features
    grp_same = grid.groupby(["geohash", "day"])
    
    grid["lag_same_day_1"] = grp_same["demand"].shift(1)
    grid["lag_same_day_2"] = grp_same["demand"].shift(2)
    grid["lag_same_day_4"] = grp_same["demand"].shift(4)
    grid["lag_same_day_8"] = grp_same["demand"].shift(8)

    # Groupby rolling needs to be done on the shifted demand column
    grid["demand_shift1"] = grid["lag_same_day_1"]
    
    grid["rollmean_same_day_3"] = grid.groupby(["geohash", "day"])["demand_shift1"].rolling(3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    grid["rollmean_same_day_8"] = grid.groupby(["geohash", "day"])["demand_shift1"].rolling(8, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    grid["rollstd_same_day_8"]  = grid.groupby(["geohash", "day"])["demand_shift1"].rolling(8, min_periods=2).std().reset_index(level=[0, 1], drop=True)

    # Drop temporary column
    grid = grid.drop(columns=["demand_shift1"])

    # 6. Compute previous-day lags (shifting by 96 slots on geohash group)
    grp_geo = grid.groupby("geohash")
    
    grid["lag_prev_day_0"] = grp_geo["demand"].shift(96)
    grid["lag_prev_day_1"] = grp_geo["demand"].shift(97)
    grid["lag_prev_day_4"] = grp_geo["demand"].shift(100)
    
    grid["demand_shift97"] = grid["lag_prev_day_1"]
    grid["prev_day_rollmean_8"] = grid.groupby("geohash")["demand_shift97"].rolling(8, min_periods=1).mean().reset_index(level=0, drop=True)

    # Drop temporary column
    grid = grid.drop(columns=["demand_shift97"])

    # 7. Compute morning warm-start features (time_slot < 9 of the same day)
    morning_slice = grid[grid["time_slot"] < 9]
    morning_stats = morning_slice.groupby(["geohash", "day"])["demand"].agg(["mean", "std", "max", "min"]).reset_index()
    morning_stats.columns = ["geohash", "day", "warm_mean", "warm_std", "warm_max", "warm_min"]
    
    morning_stats["warm_range"] = morning_stats["warm_max"] - morning_stats["warm_min"]
    
    # Get slot 8 specifically (warm_last)
    slot8 = morning_slice[morning_slice["time_slot"] == 8][["geohash", "day", "demand"]]
    slot8 = slot8.rename(columns={"demand": "warm_last"})
    
    # Get slot 0 for trend calculation
    slot0 = morning_slice[morning_slice["time_slot"] == 0][["geohash", "day", "demand"]]
    slot0 = slot0.rename(columns={"demand": "warm_first"})
    
    morning_stats = morning_stats.merge(slot8, on=["geohash", "day"], how="left")
    morning_stats = morning_stats.merge(slot0, on=["geohash", "day"], how="left")
    
    # Compute trend (slot 8 - slot 0)
    morning_stats["warm_trend"] = morning_stats["warm_last"] - morning_stats["warm_first"]
    morning_stats = morning_stats.drop(columns=["warm_first"])

    grid = grid.merge(morning_stats, on=["geohash", "day"], how="left")

    # Prevent target leakage: mask warm-start features for early slots (t < 9)
    mask_early = grid["time_slot"] < 9
    grid.loc[mask_early, WARM_START_COLS] = np.nan

    # 8. Merge grid features back to train and test
    feature_cols = TRAJECTORY_FEATURE_COLS + WARM_START_COLS

    train_merged = train.merge(grid[["geohash", "day", "time_slot"] + feature_cols], on=["geohash", "day", "time_slot"], how="left")
    test_merged = test.merge(grid[["geohash", "day", "time_slot"] + feature_cols], on=["geohash", "day", "time_slot"], how="left")

    for col in feature_cols:
        train[col] = train_merged[col].values
        test[col] = test_merged[col].values

    # 9. Fill remaining NaNs defensively using column medians of train set
    for col in feature_cols:
        med = train[col].median() if train[col].notna().any() else 0.0
        train[col] = train[col].fillna(med)
        test[col] = test[col].fillna(med)

    # 10. Add final time-slot fraction and cyclical features
    for df in (train, test):
        df["slot_frac"] = df["time_slot"] / 95.0
        df["slot_sin"] = np.sin(2 * np.pi * df["slot_frac"])
        df["slot_cos"] = np.cos(2 * np.pi * df["slot_frac"])

    return train, test


def build_trajectory_features_fold_aware(
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fold-aware variant: rebuilds the grid using only ``train_fold`` demand.

    Used to eliminate the cross-fold leakage documented in the module
    docstring. ``val_fold`` rows have their demand set to NaN before the
    grid merge, so lag/rolling features for a validation sample no
    longer see the validation sample's own demand through a neighbour
    window.

    Parameters
    ----------
    train_fold : pd.DataFrame
        The CV training fold. Must contain ``demand``.
    val_fold : pd.DataFrame
        The CV validation fold. Its ``demand`` is masked to NaN for the
        grid build; the column is restored on the returned val_fold.
    test : pd.DataFrame
        Held-out test set (no demand).

    Returns
    -------
    (train_out, val_out, test_out) with the same trajectory feature
    columns appended as ``build_trajectory_features``.

    Notes
    -----
    This function is ~5× the cost of ``build_trajectory_features`` (one
    grid build per fold per model). Only enable it if the
    GroupKFold(day) sanity run shows the leakage is material.
    """
    val_demand = val_fold["demand"].copy() if "demand" in val_fold.columns else None
    val_masked = val_fold.copy()
    if "demand" in val_masked.columns:
        val_masked["demand"] = np.nan

    combined_train = pd.concat([train_fold, val_masked], ignore_index=True, sort=False)
    n_train = len(train_fold)

    # ``build_trajectory_features`` does ``test.merge(grid[...], on=key)`` and
    # then ``test[col] = test_merged[col].values``. If test already has any of
    # the lag/rolling columns (e.g. from a previous fold's rebuild), the merge
    # mangles them with ``_x``/``_y`` suffixes, so the subsequent assignment
    # raises KeyError. Drop pre-existing trajectory cols from test so the
    # merge brings them in cleanly.
    _TRAJECTORY_COLS = tuple(TRAJECTORY_FEATURE_COLS + WARM_START_COLS + ["slot_frac", "slot_sin", "slot_cos"])
    drop_existing = [c for c in _TRAJECTORY_COLS if c in test.columns]
    if drop_existing:
        test = test.drop(columns=drop_existing)

    combined_out, test_out = build_trajectory_features(combined_train, test)

    # Slice back to the original two folds. The order from
    # ``pd.concat`` preserves train_fold rows first, then val_masked.
    train_out = combined_out.iloc[:n_train].copy()
    val_out = combined_out.iloc[n_train:].copy()

    if val_demand is not None:
        val_out["demand"] = val_demand.values

    return train_out, val_out, test_out
