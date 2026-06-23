"""
Feature engineering pipeline — temporal, spatial, categorical, and interaction features.
"""

import numpy as np
import pandas as pd

from src.geohash_decoder import decode_geohash


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations to *df* (in-place safe).

    Creates 38+ new columns covering temporal signals, geohash coordinates,
    categorical encodings, missing-value flags, and interaction terms.

    Parameters
    ----------
    df : pd.DataFrame
        Raw train or test DataFrame with original columns.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with engineered features appended.
    """
    df = df.copy()

    # ── Temporal features ────────────────────────────────────
    time_parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"] = time_parts[0]
    df["minute"] = time_parts[1]
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15  # 0-95

    df["is_morning_rush"] = ((df["hour"] >= 7) & (df["hour"] <= 10)).astype(int)
    df["is_evening_rush"] = ((df["hour"] >= 16) & (df["hour"] <= 19)).astype(int)
    df["is_rush_hour"] = (df["is_morning_rush"] | df["is_evening_rush"]).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_midday"] = ((df["hour"] >= 11) & (df["hour"] <= 14)).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["time_slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["time_slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 96)

    # ── Geohash spatial features ─────────────────────────────
    geo_coords = df["geohash"].apply(decode_geohash)
    df["latitude"] = geo_coords.apply(lambda x: x[0])
    df["longitude"] = geo_coords.apply(lambda x: x[1])
    df["geohash_prefix4"] = df["geohash"].str[:4]
    df["geohash_prefix5"] = df["geohash"].str[:5]

    # ── Categorical encoding ─────────────────────────────────
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    road_map = {"Residential": 0, "Street": 1, "Highway": 2, "Unknown": 3}
    df["RoadType_encoded"] = df["RoadType"].map(road_map)

    df["LargeVehicles_encoded"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["Landmarks_encoded"] = (df["Landmarks"] == "Yes").astype(int)

    df["Weather"] = df["Weather"].fillna("Unknown")
    weather_map = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3, "Unknown": 4}
    df["Weather_encoded"] = df["Weather"].map(weather_map)

    # One-hot columns
    for w in ["Sunny", "Rainy", "Foggy", "Snowy"]:
        df[f"weather_{w.lower()}"] = (df["Weather"] == w).astype(int)
    for r in ["Residential", "Street", "Highway"]:
        df[f"road_{r.lower()}"] = (df["RoadType"] == r).astype(int)

    # ── Temperature imputation ───────────────────────────────
    temp_median_global = df["Temperature"].median()
    df["Temperature"] = df.groupby("Weather")["Temperature"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Temperature"] = df["Temperature"].fillna(temp_median_global)
    df["temp_missing"] = df["Temperature"].isna().astype(int)
    df["temp_bin"] = pd.cut(df["Temperature"], bins=10, labels=False)
    df["temp_bin"] = df["temp_bin"].fillna(5)

    # ── Interaction features ─────────────────────────────────
    df["lanes_x_rush"] = df["NumberofLanes"] * df["is_rush_hour"]
    df["lanes_x_large"] = df["NumberofLanes"] * df["LargeVehicles_encoded"]
    df["lanes_x_landmarks"] = df["NumberofLanes"] * df["Landmarks_encoded"]
    df["lat_x_hour"] = df["latitude"] * df["hour"]
    df["lon_x_hour"] = df["longitude"] * df["hour"]
    df["temp_x_rush"] = df["Temperature"] * df["is_rush_hour"]
    df["temp_x_night"] = df["Temperature"] * df["is_night"]
    df["lanes_x_road"] = df["NumberofLanes"] * df["RoadType_encoded"]

    # Density proxy
    prefix4_counts = df["geohash_prefix4"].value_counts()
    df["prefix4_density"] = df["geohash_prefix4"].map(prefix4_counts)

    return df
