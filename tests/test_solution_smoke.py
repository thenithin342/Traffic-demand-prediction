"""End-to-end smoke test of submission construction.

Replicates solution.py lines 158-170 (the submission block) on a 100-row
sample of train + 50-row sample of test, using fake uniform predictions.
Writes the submission to a tmp_path to avoid polluting output/submission.csv.
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_datasets


@pytest.fixture(scope="module")
def loaded():
    """Load real data once per module; truncate to a small smoke sample."""
    train, test, _ = load_datasets()
    train_sample = train.head(100).reset_index(drop=True)
    test_sample = test.head(50).reset_index(drop=True)
    return train_sample, test_sample


def test_submission_csv_format(loaded, tmp_path):
    train_sample, test_sample = loaded
    test_index = test_sample["Index"].values

    # Fake predictions: uniform in [0, 1]. This test isn't checking model quality,
    # only the submission format contract.
    final_preds = np.random.default_rng(0).uniform(0.0, 1.0, size=len(test_sample))
    final_preds = np.clip(final_preds, 0, 1)

    submission = pd.DataFrame({"Index": test_index, "demand": final_preds})
    sub_path = os.path.join(str(tmp_path), "submission.csv")
    submission.to_csv(sub_path, index=False)

    # File exists and is parseable.
    assert os.path.exists(sub_path)
    roundtrip = pd.read_csv(sub_path)
    assert list(roundtrip.columns) == ["Index", "demand"]
    assert len(roundtrip) == len(test_sample)

    # No NaN, dtype is float, predictions clipped to [0, 1].
    assert roundtrip["demand"].notna().all()
    assert roundtrip["demand"].dtype.kind == "f"
    assert roundtrip["demand"].between(0.0, 1.0).all()

    # Index column matches test order exactly.
    assert (roundtrip["Index"].values == test_index).all()