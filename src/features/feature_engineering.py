"""
Feature engineering pipeline - temporal, spatial, categorical, and
interaction features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.geohash_decoder import decode_geohash


@dataclass
class FeatureState:
    """Container for fit-once-use-many feature engineering state.

    Populated during the ``is_train=True`` pass; consumed during
    ``is_train=False``. Fields with defaults can be safely read before the
    train pass, but ``temp_bins`` is ``None`` until the train pass fits the
    binner.
    """

    temp_median_global: float = 20.0
    weather_medians: dict[str, float] = field(default_factory=dict)
    temp_bins: np.ndarray | None = None
    _fitted: bool = False


def fit_feature_state(df: pd.DataFrame, state: FeatureState) -> FeatureState:
    """Fit feature engineering state (temp bins, weather medians) from training data."""
    df = df.copy()
    state.temp_median_global = (
        pd.to_numeric(df["Temperature"], errors="coerce").median()
    )
    state.weather_medians = (
        df.groupby("Weather")["Temperature"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").median())
        .to_dict()
    )
    _, state.temp_bins = pd.cut(
        pd.to_numeric(df["Temperature"], errors="coerce"), bins=10, retbins=True,
    )
    # expand bins slightly to handle test values out of range
    state.temp_bins[0] = -1000
    state.temp_bins[-1] = 1000
    state._fitted = True
    return state


def apply_feature_state(df: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    """Apply fitted *state* to *df* (train or test). All transform logic lives here."""
    df = df.copy()

    # ----------------------------- temporal features
    time_parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"] = time_parts[0]
    df["minute"] = time_parts[1]
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15  # 0-95

    df["is_morning_rush"] = ((df["hour"] >= 7) & (df["hour"] <= 10)).astype(int)
    df["is_evening_rush"] = ((df["hour"] >= 16) & (df["hour"] <= 19)).astype(int)
    df["is_rush_hour"] = (df["is_morning_rush"] | df["is_evening_rush"]).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_midday"] = ((df["hour"] >= 11) & (df["hour"] <= 14)).astype(int)
    # NOTE: removed `is_weekend_proxy = (day >= 49)` because it is a perfect
    # day indicator. In a 2-day GroupKFold it lets the model memorise the
    # per-fold demand distribution, inflating OOF R^2 without helping test.

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["time_slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["time_slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 96)

    # ----------------------------- geohash spatial features
    # Cache decoding to avoid per-row overhead
    unique_geohashes = df["geohash"].unique()
    decoded = {gh: decode_geohash(gh) for gh in unique_geohashes}

    df["latitude"] = df["geohash"].map(lambda x: decoded[x][0])
    df["longitude"] = df["geohash"].map(lambda x: decoded[x][1])

    df["lat_sq"] = df["latitude"] ** 2
    df["lon_sq"] = df["longitude"] ** 2
    df["lat_x_lon"] = df["latitude"] * df["longitude"]

    df["geohash_prefix4"] = df["geohash"].str[:4]
    df["geohash_prefix5"] = df["geohash"].str[:5]

    # ----------------------------- categorical encoding
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    road_map = {"Residential": 0, "Street": 1, "Highway": 2, "Unknown": 3}
    df["RoadType_encoded"] = df["RoadType"].map(road_map).fillna(3).astype(int)

    df["LargeVehicles"] = df["LargeVehicles"].fillna("Not Allowed")
    df["LargeVehicles_encoded"] = (df["LargeVehicles"] == "Allowed").astype(int)

    df["Landmarks"] = df["Landmarks"].fillna("No")
    df["Landmarks_encoded"] = (df["Landmarks"] == "Yes").astype(int)

    df["Weather"] = df["Weather"].fillna("Unknown")
    weather_map = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3, "Unknown": 4}
    df["Weather_encoded"] = df["Weather"].map(weather_map).fillna(4).astype(int)

    # One-hot columns
    for w in ["Sunny", "Rainy", "Foggy", "Snowy"]:
        df[f"weather_{w.lower()}"] = (df["Weather"] == w).astype(int)
    for r in ["Residential", "Street", "Highway"]:
        df[f"road_{r.lower()}"] = (df["RoadType"] == r).astype(int)

    # ----------------------------- temperature imputation
    # BUG FIX: capture the missing mask BEFORE we fillna, otherwise
    # `temp_missing` is always 0 (a useless constant column).
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df["temp_missing"] = df["Temperature"].isna().astype(int)

    df["Temperature"] = df["Temperature"].fillna(df["Weather"].map(state.weather_medians))
    df["Temperature"] = df["Temperature"].fillna(state.temp_median_global)

    df["temp_bin"] = pd.cut(df["Temperature"], bins=state.temp_bins, labels=False)
    df["temp_bin"] = df["temp_bin"].fillna(5).astype(int)

    # ----------------------------- interaction features
    df["lanes_x_rush"] = df["NumberofLanes"] * df["is_rush_hour"]
    df["lanes_x_large"] = df["NumberofLanes"] * df["LargeVehicles_encoded"]
    df["lanes_x_landmarks"] = df["NumberofLanes"] * df["Landmarks_encoded"]
    df["lat_x_hour"] = df["latitude"] * df["hour"]
    df["lon_x_hour"] = df["longitude"] * df["hour"]
    df["temp_x_rush"] = df["Temperature"] * df["is_rush_hour"]
    df["temp_x_night"] = df["Temperature"] * df["is_night"]
    df["lanes_x_road"] = df["NumberofLanes"] * df["RoadType_encoded"]
    df["lanes_x_weather"] = df["NumberofLanes"] * df["Weather_encoded"]

    # Speed limit proxy from road type
    speed_map = {"Residential": 30, "Street": 50, "Highway": 120, "Unknown": 50}
    df["speed_limit_proxy"] = df["RoadType"].map(speed_map).fillna(50).astype(int)

    # Hour x Road type interaction
    df["hour_x_road"] = df["hour"] * df["RoadType_encoded"]

    # NOTE: `prefix4_density` is computed in the target-encoding stage using
    # the combined train+test frame, so it is consistent for both splits.
    # The original per-call value_counts() on a single df would leak test
    # info into the training frame and vice versa.

    return df


def engineer_features(
    df: pd.DataFrame,
    is_train: bool = True,
    state: FeatureState | None = None,
) -> pd.DataFrame:
    """Apply all feature engineering transformations to *df*.

    Thin wrapper: fits state on train, applies fitted state to test.
    Creates 30+ new columns covering temporal signals, geohash coordinates,
    categorical encodings, missing-value flags, and interaction terms.
    """
    if state is None:
        state = FeatureState()
    if is_train:
        fit_feature_state(df, state)
        return apply_feature_state(df, state)
    if not state._fitted:
        raise ValueError(
            "FeatureState has not been fitted. "
            "Call engineer_features(df, is_train=True, state=state) on training data first."
        )
    return apply_feature_state(df, state)
