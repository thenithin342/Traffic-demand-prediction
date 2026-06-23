import pandas as pd
import numpy as np

train = pd.read_csv(r'C:\Users\theni\OneDrive\Desktop\Traffic demand prediction\data\dataset\train.csv')
test = pd.read_csv(r'C:\Users\theni\OneDrive\Desktop\Traffic demand prediction\data\dataset\test.csv')

print('=== TRAIN SHAPE ===')
print(train.shape)
print('\n=== TEST SHAPE ===')
print(test.shape)
print('\n=== DTYPES ===')
print(train.dtypes)
print('\n=== MISSING VALUES (TRAIN) ===')
print(train.isnull().sum())
print('\n=== MISSING VALUES (TEST) ===')
print(test.isnull().sum())
print('\n=== DESCRIBE (TRAIN) ===')
print(train.describe())
print('\n=== TARGET STATS ===')
print(f"demand min: {train['demand'].min()}")
print(f"demand max: {train['demand'].max()}")
print(f"demand mean: {train['demand'].mean()}")
print(f"demand median: {train['demand'].median()}")
print(f"demand std: {train['demand'].std()}")

print('\n=== CATEGORICAL UNIQUE VALUES ===')
for col in ['geohash', 'day', 'timestamp', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']:
    print(f"\n{col}: {train[col].nunique()} unique values")
    print(f"  Top 5: {train[col].value_counts().head(5).to_dict()}")
    if col in test.columns:
        # Check if test has values not in train
        train_vals = set(train[col].dropna().unique())
        test_vals = set(test[col].dropna().unique())
        new_in_test = test_vals - train_vals
        if new_in_test:
            print(f"  NEW in test (not in train): {new_in_test}")

print('\n=== DAY RANGE ===')
print(f"Train days: {sorted(train['day'].unique())}")
print(f"Test days: {sorted(test['day'].unique())}")

print('\n=== TIMESTAMP UNIQUE ===')
print(f"Train timestamps: {sorted(train['timestamp'].unique())}")

print('\n=== GEOHASH OVERLAP ===')
train_geo = set(train['geohash'].unique())
test_geo = set(test['geohash'].unique())
print(f"Train geohashes: {len(train_geo)}")
print(f"Test geohashes: {len(test_geo)}")
print(f"Common: {len(train_geo & test_geo)}")
print(f"Only in test: {len(test_geo - train_geo)}")

print('\n=== NumberofLanes VALUES ===')
print(train['NumberofLanes'].value_counts())

print('\n=== Temperature RANGE ===')
print(f"Train: min={train['Temperature'].min()}, max={train['Temperature'].max()}")
print(f"Test: min={test['Temperature'].min()}, max={test['Temperature'].max()}")
