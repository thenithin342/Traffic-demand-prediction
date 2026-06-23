"""
Target encoding, frequency encoding, and demand-statistic features.

All aggregate statistics are computed from **day 48 only** to prevent
temporal leakage into the day-49 test set.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import TARGET_SMOOTHING


def _smooth_target_encode(
    train_ref: pd.DataFrame,
    train_full: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
    target: str = "demand",
    smoothing: int = TARGET_SMOOTHING,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bayesian-smoothed target encoding for *group_col*."""
    global_mean = train_ref[target].mean()
    agg = train_ref.groupby(group_col)[target].agg(["mean", "count"])
    agg["smooth_mean"] = (
        agg["count"] * agg["mean"] + smoothing * global_mean
    ) / (agg["count"] + smoothing)

    col_name = f"{group_col}_target_enc"
    train_full[col_name] = train_full[group_col].map(agg["smooth_mean"]).fillna(global_mean)
    test_df[col_name] = test_df[group_col].map(agg["smooth_mean"]).fillna(global_mean)

    return train_full, test_df


def apply_target_encodings(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create all target-encoded and aggregate-statistic features.

    Parameters
    ----------
    train, test : pd.DataFrame
        DataFrames **after** feature engineering (must contain ``demand``,
        ``geohash``, ``geohash_prefix4``, ``geohash_prefix5``, ``time_slot``,
        ``Weather``, ``RoadType``, ``day``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train, test)`` with new columns appended.
    """
    train_day48 = train[train["day"] == 48]

    # ── Target encoding by grouping columns ──────────────────
    for col in ("geohash", "geohash_prefix4", "geohash_prefix5"):
        train, test = _smooth_target_encode(train_day48, train, test, col)

    # Geo x time-slot
    train_day48 = train_day48.copy()
    train_day48["geo_time"] = train_day48["geohash"] + "_" + train_day48["time_slot"].astype(str)
    train["geo_time"] = train["geohash"] + "_" + train["time_slot"].astype(str)
    test["geo_time"] = test["geohash"] + "_" + test["time_slot"].astype(str)
    train, test = _smooth_target_encode(train_day48, train, test, "geo_time")

    for col in ("Weather", "RoadType"):
        train, test = _smooth_target_encode(train_day48, train, test, col)

    # ── Frequency encoding ───────────────────────────────────
    geo_freq = train["geohash"].value_counts().to_dict()
    train["geohash_freq"] = train["geohash"].map(geo_freq)
    test["geohash_freq"] = test["geohash"].map(geo_freq).fillna(0)

    # ── Geohash demand statistics (from day 48) ──────────────
    geo_stats = train_day48.groupby("geohash")["demand"].agg(
        ["mean", "std", "median", "min", "max"]
    )
    geo_stats.columns = [
        "geo_demand_mean", "geo_demand_std", "geo_demand_median",
        "geo_demand_min", "geo_demand_max",
    ]
    for col in geo_stats.columns:
        train[col] = train["geohash"].map(geo_stats[col]).fillna(geo_stats[col].median())
        test[col] = test["geohash"].map(geo_stats[col]).fillna(geo_stats[col].median())

    # ── Time-slot demand statistics ──────────────────────────
    time_stats = train_day48.groupby("time_slot")["demand"].agg(["mean", "std", "median"])
    time_stats.columns = [
        "timeslot_demand_mean", "timeslot_demand_std", "timeslot_demand_median",
    ]
    for col in time_stats.columns:
        train[col] = train["time_slot"].map(time_stats[col]).fillna(time_stats[col].median())
        test[col] = test["time_slot"].map(time_stats[col]).fillna(time_stats[col].median())

    # ── Label encoding for tree models ───────────────────────
    for col in ("geohash", "geohash_prefix4", "geohash_prefix5"):
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0)
        le.fit(combined)
        train[f"{col}_label"] = le.transform(train[col])
        test[f"{col}_label"] = le.transform(test[col])

    return train, test
