import pandas as pd
import pytest

from src.feature_engineering import engineer_features, FeatureState


def test_engineer_features_shape_and_columns():
    df = pd.DataFrame({
        "Index": [0, 1],
        "geohash": ["qp02z1", "qp02zt"],
        "day": [48, 48],
        "timestamp": ["0:0", "7:15"],
        "RoadType": ["Residential", "Highway"],
        "NumberofLanes": [1, 3],
        "LargeVehicles": ["Not Allowed", "Allowed"],
        "Landmarks": ["No", "Yes"],
        "Temperature": [25.0, float("nan")],
        "Weather": ["Sunny", "Rainy"],
    })
    out = engineer_features(df)
    # Original columns preserved
    assert "Index" in out.columns
    # Engineered columns present
    for c in [
        "hour", "minute", "time_slot", "is_morning_rush", "is_evening_rush",
        "is_rush_hour", "is_night", "is_midday",
        "hour_sin", "hour_cos", "time_slot_sin", "time_slot_cos",
        "latitude", "longitude", "geohash_prefix4", "geohash_prefix5",
        "RoadType_encoded", "LargeVehicles_encoded", "Landmarks_encoded",
        "Weather_encoded", "Temperature", "temp_missing", "temp_bin",
        "lanes_x_rush", "lanes_x_large", "lanes_x_landmarks",
        "lat_x_hour", "lon_x_hour", "temp_x_rush", "temp_x_night",
        "lanes_x_road",
    ]:
        assert c in out.columns, f"Missing engineered column: {c}"
    # BUG REGRESSION: temp_missing must reflect the input NaN, not 0.
    # Row 1 had NaN temperature and should be flagged.
    assert int(out.loc[0, "temp_missing"]) == 0
    assert int(out.loc[1, "temp_missing"]) == 1
    # BUG REGRESSION: is_weekend_proxy was a perfect day indicator; removed.
    assert "is_weekend_proxy" not in out.columns
    # Temperature NaN imputed to per-Weather median (or global median)
    assert out["Temperature"].notna().all()
    # No NaN in any of the derived numeric features we rely on
    for c in ["hour", "time_slot", "latitude", "longitude"]:
        assert out[c].notna().all()


def test_rush_hour_flags():
    df = pd.DataFrame({
        "Index": [0, 1, 2, 3],
        "geohash": ["u", "u", "u", "u"],
        "day": [48, 48, 48, 48],
        "timestamp": ["8:0", "17:30", "3:0", "12:0"],
        "RoadType": ["Street"] * 4,
        "NumberofLanes": [2] * 4,
        "LargeVehicles": ["Not Allowed"] * 4,
        "Landmarks": ["No"] * 4,
        "Temperature": [20.0] * 4,
        "Weather": ["Sunny"] * 4,
    })
    out = engineer_features(df)
    assert int(out.loc[0, "is_morning_rush"]) == 1
    assert int(out.loc[1, "is_evening_rush"]) == 1
    assert int(out.loc[2, "is_night"]) == 1
    assert int(out.loc[3, "is_midday"]) == 1


def test_state_required_when_is_train_false():
    """Calling engineer_features(is_train=False) on a fresh FeatureState must
    raise a clear error rather than silently fall back to wrong defaults."""
    df = pd.DataFrame({
        "Index": [0, 1],
        "geohash": ["qp02z1", "qp02zt"],
        "day": [48, 48],
        "timestamp": ["0:0", "7:15"],
        "RoadType": ["Residential", "Highway"],
        "NumberofLanes": [1, 3],
        "LargeVehicles": ["Not Allowed", "Allowed"],
        "Landmarks": ["No", "Yes"],
        "Temperature": [25.0, 22.0],
        "Weather": ["Sunny", "Rainy"],
    })
    fresh = FeatureState()  # temp_bins=None, weather_medians={}, temp_median=20.0
    with pytest.raises(ValueError, match="not been fitted"):
        engineer_features(df, is_train=False, state=fresh)
