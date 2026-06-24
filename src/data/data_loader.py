"""
Data loading and initial preprocessing utilities.
"""

import os
import pandas as pd
from config import DATA_DIR


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample-submission CSVs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, test, sample_submission)`` DataFrames.
    """
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    print(f"  Train shape : {train.shape}")
    print(f"  Test shape  : {test.shape}")
    print(f"  Sample sub  : {sample_sub.shape}")

    return train, test, sample_sub
