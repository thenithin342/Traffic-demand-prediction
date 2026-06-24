"""Tests for src/trajectory.py::build_trajectory_features.

Covers:
- same-day lag equals previous slot's demand (independent of step-9 median fill)
- warm_last equals slot-8 demand for t >= 9 (non-NaN pre-fill)
- warm_last for t < 9 is filled with column median (step-9 fill behaviour)
- prev-day lag maps (g, 49, t-1) -> demand at (g, 48, t-1)
- cyclical features appended on returned train/test
"""
import numpy as np
import pandas as pd

from src.trajectory import build_trajectory_features


def _make_frames():
    """Build a 1-geohash, 2-day (48+49), 96-slot deterministic demand ramp.

    Demand series: 0.00, 0.01, 0.02, ..., 0.95 per slot, replicated on
    each day so cross-day lags are computable.
    """
    geo = "qp02z1"
    rows = []
    for day in (48, 49):
        for slot in range(96):
            rows.append({
                "Index": len(rows),
                "geohash": geo,
                "day": day,
                "timestamp": f"{slot // 4}:{(slot % 4) * 15}",
                "demand": slot / 100.0,
            })
    train = pd.DataFrame(rows)

    # Minimal test frame: 1 row on day 49, slot 50 to exercise merge.
    test = pd.DataFrame([{
        "Index": 99999,
        "geohash": geo,
        "day": 49,
        "timestamp": "12:30",
        "RoadType": "Residential",
        "NumberofLanes": 1,
        "LargeVehicles": "Not Allowed",
        "Landmarks": "No",
        "Temperature": 25.0,
        "Weather": "Sunny",
    }])
    return train, test


def test_lag_same_day_1_matches_previous_slot_demand():
    train, test = _make_frames()
    train_out, _ = build_trajectory_features(train, test)

    # Build lookup of demand by (day, slot)
    demand_lookup = {(int(r.day), int(r.time_slot)): float(r.demand)
                     for r in train_out.itertuples(index=False)}

    # For slot t >= 1, lag_same_day_1 should equal demand at t-1.
    mismatches = []
    for r in train_out.itertuples(index=False):
        if r.time_slot >= 1:
            expected = demand_lookup[(int(r.day), int(r.time_slot) - 1)]
            if not np.isclose(r.lag_same_day_1, expected, atol=1e-9):
                mismatches.append((r.day, r.time_slot, expected, r.lag_same_day_1))
    assert not mismatches, f"lag_same_day_1 mismatches: {mismatches[:5]}"


def test_lag_prev_day_1_maps_to_previous_day_previous_slot():
    train, test = _make_frames()
    train_out, _ = build_trajectory_features(train, test)

    demand_lookup = {(int(r.day), int(r.time_slot)): float(r.demand)
                     for r in train_out.itertuples(index=False)}

    # For day 49, slot t >= 1: lag_prev_day_1 should equal demand at (48, t-1).
    for t in (1, 10, 50, 95):
        row = train_out[(train_out["day"] == 49) & (train_out["time_slot"] == t)].iloc[0]
        expected = demand_lookup[(48, t - 1)]
        assert np.isclose(row.lag_prev_day_1, expected, atol=1e-9), (
            f"day=49 slot={t}: expected {expected}, got {row.lag_prev_day_1}"
        )


def test_warm_last_at_t_ge_9_equals_slot_8_demand():
    train, test = _make_frames()
    train_out, _ = build_trajectory_features(train, test)

    # Demand at slot 8 of each day.
    for day in (48, 49):
        slot8_demand = train_out[(train_out["day"] == day) & (train_out["time_slot"] == 8)].iloc[0].demand
        rows_t9 = train_out[(train_out["day"] == day) & (train_out["time_slot"] == 9)]
        assert np.isclose(rows_t9.iloc[0].warm_last, slot8_demand, atol=1e-9), (
            f"day={day} t=9: warm_last={rows_t9.iloc[0].warm_last} expected={slot8_demand}"
        )
        # Also check a later slot (e.g. t=50) still tracks slot-8 demand.
        rows_t50 = train_out[(train_out["day"] == day) & (train_out["time_slot"] == 50)]
        assert np.isclose(rows_t50.iloc[0].warm_last, slot8_demand, atol=1e-9)


def test_warm_last_at_t_lt_9_filled_with_column_median():
    """Step 7 masks warm cols to NaN on the grid; step 9 fills with column median.

    For t < 9, the final returned train has warm_last == column median, not slot-8 demand.
    """
    train, test = _make_frames()
    train_out, _ = build_trajectory_features(train, test)

    expected_median = float(train_out["warm_last"].median())
    early = train_out[train_out["time_slot"] < 9]
    for r in early.itertuples(index=False):
        # All early-slot rows are filled to the same value (column median).
        assert np.isclose(r.warm_last, expected_median, atol=1e-9), (
            f"day={r.day} t={r.time_slot}: warm_last={r.warm_last} expected_median={expected_median}"
        )


def test_cyclical_features_appended():
    train, test = _make_frames()
    train_out, test_out = build_trajectory_features(train, test)
    for c in ("slot_frac", "slot_sin", "slot_cos"):
        assert c in train_out.columns, f"missing cyclical col {c} in train"
        assert c in test_out.columns, f"missing cyclical col {c} in test"
    # slot_sin^2 + slot_cos^2 == 1 (within fp tolerance)
    sumsq = train_out["slot_sin"] ** 2 + train_out["slot_cos"] ** 2
    assert np.allclose(sumsq, 1.0, atol=1e-9)