import pandas as pd
import numpy as np
from src.target_encoding import apply_static_target_encodings, compute_target_dependent_encodings


def test_static_target_encodings():
    train = pd.DataFrame({
        "Index": [0, 1, 2],
        "geohash": ["a", "b", "a"],
        "day": [48, 48, 48],
    })
    test = pd.DataFrame({
        "Index": [3, 4],
        "geohash": ["a", "c"],  # 'c' is unseen
        "day": [49, 49],
    })
    tr, te = apply_static_target_encodings(train, test)
    
    assert "geohash_freq" in tr.columns
    assert "geohash_freq" in te.columns
    assert "prefix4_density" in tr.columns
    assert "prefix4_density" in te.columns
    assert "geohash_label" in tr.columns
    assert "geohash_label" in te.columns

    # No NaNs
    assert tr["geohash_freq"].notna().all()
    assert te["geohash_freq"].notna().all()


def test_target_dependent_encodings_leak_free():
    # Create train fold
    train_fold = pd.DataFrame({
        "geohash": ["a", "a", "b"],
        "geohash_prefix4": ["a", "a", "b"],
        "geohash_prefix5": ["a", "a", "b"],
        "time_slot": [10, 10, 11],
        "Weather": ["Sunny", "Sunny", "Rainy"],
        "RoadType": ["Street", "Street", "Highway"],
        "demand": [0.2, 0.4, 0.9],
    })
    
    # Create validation fold with different targets
    val_fold = pd.DataFrame({
        "geohash": ["a", "b"],
        "geohash_prefix4": ["a", "b"],
        "geohash_prefix5": ["a", "b"],
        "time_slot": [10, 11],
        "Weather": ["Sunny", "Rainy"],
        "RoadType": ["Street", "Highway"],
        "demand": [0.95, 0.05],  # Very different values to detect leakage
    })
    
    test_df = pd.DataFrame({
        "geohash": ["a", "c"],
        "geohash_prefix4": ["a", "c"],
        "geohash_prefix5": ["a", "c"],
        "time_slot": [10, 12],
        "Weather": ["Sunny", "Sunny"],
        "RoadType": ["Street", "Street"],
    })
    
    tr, val, te = compute_target_dependent_encodings(train_fold, val_fold, test_df)
    
    # Assert columns exist
    for col in ["geohash_target_enc", "geo_time_target_enc", "geo_demand_mean", "timeslot_demand_mean", "weather_slot_mean"]:
        assert col in tr.columns
        assert col in val.columns
        assert col in te.columns
        assert tr[col].notna().all()
        assert val[col].notna().all()
        assert te[col].notna().all()

    # Leakage check:
    # Target encoding for geohash 'a' in train_fold should be based on its values in train_fold (0.2 and 0.4).
    # Since there are 2 values (0.2, 0.4), the mean is 0.3.
    # Smooth mean = (count * mean + smoothing * global_mean) / (count + smoothing)
    # Global mean of train_fold = (0.2 + 0.4 + 0.9) / 3 = 0.5.
    # For smoothing = 10.0:
    # Smooth mean = (2 * 0.3 + 10 * 0.5) / 12 = 5.6 / 12 = 0.466667.
    # Val rows for 'a' must equal 0.466667 (full-train agg, no leak from val_fold).
    # Train rows for 'a' use OOF encoding: each is computed from an inner fold that
    # EXCLUDES that row's own demand, so they are NOT exactly 0.466667.
    val_a_enc = val.loc[val["geohash"] == "a", "geohash_target_enc"].iloc[0]
    tr_a_enc = tr.loc[tr["geohash"] == "a", "geohash_target_enc"].iloc[0]

    # Val must equal full-train smoothed mean (no leak from val_fold's target 0.95).
    assert abs(val_a_enc - 0.4666666666666667) < 1e-6
    # Train uses OOF encoding — not equal to full-train smoothed mean.
    assert abs(tr_a_enc - 0.4666666666666667) > 1e-3
    # Train and val encodings differ because train is OOF, val is full-train agg.
    assert abs(tr_a_enc - val_a_enc) > 1e-3
