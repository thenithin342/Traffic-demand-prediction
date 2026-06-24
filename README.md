# Traffic Demand Prediction

AI-powered traffic demand forecasting using an **ensemble of LightGBM, XGBoost, CatBoost, HistGradientBoosting, and MLP** with extensive feature engineering.

## Problem

Predict the `demand` (continuous, in [0, 1]) at geohash locations for the day-49 test set, given temporal, road, weather, and spatial features. Metric:

```
score = max(0, 100 * R2(actual, predicted))
```

## Dataset

| File | Shape |
|---|---|
| `data/dataset/train.csv` | 77,299 x 11 |
| `data/dataset/test.csv`  | 41,778 x 10 |
| `data/dataset/sample_submission.csv` | 5 x 2 |

Train covers days 48 (69,427) and 49 (7,872). Test is day 49 (41,778). 1,249 geohash cells, 1,180 of which appear in test.

## Quick start

```bash
pip install -r requirements.txt
python solution.py
```

This will:
1. Load and preprocess the dataset
2. Engineer 80+ features (temporal, spatial, trajectory, neighbor, target-enc)
3. Train LGB, XGB, CatBoost, HistGradientBoosting, MLP with 2-fold GroupKFold (day-level; honest temporal split)
4. Tune ensemble weights with SLSQP and a 2nd-level XGB stacker
5. Pick the better of (weighted blend, stacked)
6. Write `output/submission.csv` and 14 plots in `output/plots/`

## Validation strategy

- **Primary**: 2-fold `GroupKFold` on `day` (grouped on the `day` column). Each
  fold's validation is a day distinct from training; this gives an honest
  temporal OOF R² because the model must extrapolate across days. Surfaces
  any cross-fold leakage in trajectory lag / rolling features.
- **Secondary**: classic KFold used inside the ensemble meta-stacker (Ridge,
  SLSQP, XGB) over OOF predictions. Stacker-internal splits are not
  group-aware — they're for blending only, not for reporting.
- The "OOF R2" reported in the log is the **day-49-only** OOF when `day=49`
  appears in the validation fold. This mirrors the actual test distribution
  (the test set is day 49).
- See `config.py::CV_MODE` (toggle `"kfold"` / `"groupkfold_day"`) and
  `FOLD_AWARE_TRAJECTORY` (lag/rolling grid rebuilt per fold vs. once
  globally; True is the leak-safe default).

## Project structure

```
.
+- config.py            # all settings (paths, hyperparams, ensemble)
+- solution.py          # end-to-end pipeline
+- eda.py               # standalone EDA script
+- tests/               # pytest unit tests
+- src/
|  +- data_loader.py
|  +- geohash_decoder.py
|  +- feature_engineering.py
|  +- target_encoding.py
|  +- trajectory.py     # per-geohash time-series features (NEW)
|  +- neighbor.py       # spatial / cluster / neighbor features (NEW)
|  +- models.py         # LGB / XGB / CatBoost / HistGradient CV trainers
|  +- ensemble.py       # Ridge, SLSQP weights, XGB stacker
|  +- visualizations.py
+- data/dataset/        # train / test / sample_submission
+- output/              # submission.csv + plots/
+- requirements.txt
```

## Tests

```bash
pytest -q
```
