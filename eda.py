"""Standalone Exploratory Data Analysis script.

Loads train/test using the same config the rest of the pipeline uses, so this
works regardless of where the script is run from.
"""

from src.data_loader import load_datasets

train, test, _ = load_datasets()

print("=== TRAIN SHAPE ===")
print(train.shape)
print("\n=== TEST SHAPE ===")
print(test.shape)
print("\n=== DTYPES ===")
print(train.dtypes)
print("\n=== MISSING VALUES (TRAIN) ===")
print(train.isnull().sum())
print("\n=== MISSING VALUES (TEST) ===")
print(test.isnull().sum())
print("\n=== DESCRIBE (TRAIN) ===")
print(train.describe())
print("\n=== TARGET STATS ===")
print(f"demand min   : {train['demand'].min()}")
print(f"demand max   : {train['demand'].max()}")
print(f"demand mean  : {train['demand'].mean()}")
print(f"demand median: {train['demand'].median()}")
print(f"demand std   : {train['demand'].std()}")

print("\n=== CATEGORICAL UNIQUE VALUES ===")
for col in ["geohash", "day", "timestamp", "RoadType", "LargeVehicles",
            "Landmarks", "Weather"]:
    n = train[col].nunique()
    print(f"\n{col}: {n} unique values")
    print(f"  Top 5: {train[col].value_counts().head(5).to_dict()}")
    if col in test.columns:
        train_vals = set(train[col].dropna().unique())
        test_vals = set(test[col].dropna().unique())
        new_in_test = test_vals - train_vals
        if new_in_test:
            print(f"  NEW in test (not in train): {sorted(new_in_test)[:10]}")

print("\n=== DAY RANGE ===")
print(f"Train days: {sorted(train['day'].unique())}")
print(f"Test  days: {sorted(test['day'].unique())}")

print("\n=== TIMESTAMP UNIQUE ===")
print(f"Train timestamps: {sorted(train['timestamp'].unique())}")

print("\n=== GEOHASH OVERLAP ===")
train_geo = set(train["geohash"].unique())
test_geo = set(test["geohash"].unique())
print(f"Train geohashes: {len(train_geo)}")
print(f"Test  geohashes: {len(test_geo)}")
print(f"Common        : {len(train_geo & test_geo)}")
print(f"Only in test  : {len(test_geo - train_geo)}")

print("\n=== NumberofLanes VALUES ===")
print(train["NumberofLanes"].value_counts())

print("\n=== Temperature RANGE ===")
print(f"Train: min={train['Temperature'].min()}, max={train['Temperature'].max()}")
print(f"Test : min={test['Temperature'].min()}, max={test['Temperature'].max()}")
