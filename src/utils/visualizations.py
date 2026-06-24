"""
Visualization scripts for EDA and model performance evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

from config import PLOTS_DIR

sns.set_theme(style="whitegrid", palette="viridis")


def _ensure_dir() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)


def generate_eda_plots(train: pd.DataFrame, y: np.ndarray) -> None:
    """Generate all Exploratory Data Analysis plots."""
    _ensure_dir()
    print("  Generating EDA plots...")

    # 01 Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(y, bins=80, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0].set_title("Demand Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Demand")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(np.log1p(y), bins=80, color="#FF5722", edgecolor="white", alpha=0.85)
    axes[1].set_title("Log(1 + Demand) Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("log(1 + demand)")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_demand_distribution.png"), dpi=150)
    plt.close()

    # 02 Demand by hour
    fig, ax = plt.subplots(figsize=(12, 5))
    hourly_demand = train.groupby("hour")["demand"].mean()
    ax.bar(hourly_demand.index, hourly_demand.values, color="#4CAF50", edgecolor="white")
    ax.set_title("Average Demand by Hour of Day", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Demand")
    ax.set_xticks(range(24))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_demand_by_hour.png"), dpi=150)
    plt.close()

    # 03 Demand by road type
    fig, ax = plt.subplots(figsize=(8, 5))
    road_demand = train.groupby("RoadType_encoded")["demand"].mean()
    road_labels = ["Residential", "Street", "Highway", "Unknown"]
    colors = ["#2196F3", "#FF9800", "#F44336", "#9E9E9E"]
    ax.bar(
        range(len(road_demand)),
        road_demand.values,
        color=colors[: len(road_demand)],
        edgecolor="white",
    )
    ax.set_xticks(range(len(road_demand)))
    ax.set_xticklabels(road_labels[: len(road_demand)])
    ax.set_title("Average Demand by Road Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Demand")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_demand_by_roadtype.png"), dpi=150)
    plt.close()

    # 04 Demand by weather
    fig, ax = plt.subplots(figsize=(8, 5))
    weather_demand = train.groupby("Weather_encoded")["demand"].mean()
    weather_labels = ["Sunny", "Rainy", "Foggy", "Snowy", "Unknown"]
    weather_colors = ["#FFC107", "#2196F3", "#9E9E9E", "#00BCD4", "#795548"]
    ax.bar(
        range(len(weather_demand)),
        weather_demand.values,
        color=weather_colors[: len(weather_demand)],
        edgecolor="white",
    )
    ax.set_xticks(range(len(weather_demand)))
    ax.set_xticklabels(weather_labels[: len(weather_demand)])
    ax.set_title("Average Demand by Weather", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Demand")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_demand_by_weather.png"), dpi=150)
    plt.close()

    # 05 Temperature vs Demand
    fig, ax = plt.subplots(figsize=(10, 5))
    sample_idx = np.random.choice(len(train), min(5000, len(train)), replace=False)
    ax.scatter(
        train["Temperature"].iloc[sample_idx],
        train["demand"].iloc[sample_idx],
        alpha=0.3, s=8, c="#673AB7",
    )
    ax.set_title("Temperature vs Demand (sampled)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Demand")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_temperature_vs_demand.png"), dpi=150)
    plt.close()

    # 06 Heatmap
    if "NumberofLanes" in train.columns:
        fig, ax = plt.subplots(figsize=(14, 5))
        pivot = train.pivot_table(
            values="demand", index="NumberofLanes", columns="hour", aggfunc="mean"
        )
        sns.heatmap(pivot, cmap="YlOrRd", annot=False, ax=ax, linewidths=0.3)
        ax.set_title("Demand Heatmap: NumberofLanes x Hour", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "06_heatmap_lanes_hour.png"), dpi=150)
        plt.close()

    # 07 Spatial Demand
    if "latitude" in train.columns and "longitude" in train.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        geo_mean = train.groupby("geohash").agg(
            {"latitude": "first", "longitude": "first", "demand": "mean"}
        )
        sc = ax.scatter(
            geo_mean["longitude"], geo_mean["latitude"],
            c=geo_mean["demand"], cmap="hot_r", s=12, alpha=0.8,
        )
        plt.colorbar(sc, label="Mean Demand")
        ax.set_title("Spatial Distribution of Mean Demand", fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "07_spatial_demand.png"), dpi=150)
        plt.close()

    # 08 Correlation Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    top_feat_cols = [
        "hour", "time_slot", "NumberofLanes", "Temperature", "latitude", "longitude",
        "LargeVehicles_encoded", "Landmarks_encoded", "RoadType_encoded", "Weather_encoded",
        "is_rush_hour", "is_night", "geohash_target_enc", "geo_time_target_enc",
        "geo_demand_mean", "geo_demand_std", "timeslot_demand_mean",
        "lag_same_day_1", "lag_same_day_4", "rollmean_same_day_8",
        "lag_prev_day_0", "prev_day_rollmean_8",
        "warm_mean", "geo_cluster",
    ]
    top_feat_cols = [c for c in top_feat_cols if c in train.columns]
    corr_data = train[top_feat_cols + ["demand"]].corr()
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(
        corr_data, mask=mask, cmap="coolwarm", annot=True, fmt=".2f",
        ax=ax, linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "08_correlation_heatmap.png"), dpi=150)
    plt.close()


def plot_feature_importance(features: list, importances: np.ndarray, top_n: int = 30) -> None:
    """Plot LightGBM top feature importances (mean across folds)."""
    _ensure_dir()
    fi = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    fig, ax = plt.subplots(figsize=(10, 12))
    n = min(top_n, len(fi))
    ax.barh(range(n), fi["importance"].values[:n], color="#4CAF50")
    ax.set_yticks(range(n))
    ax.set_yticklabels(fi["feature"].values[:n])
    ax.invert_yaxis()
    ax.set_title(f"LightGBM Feature Importance (Top {n}, mean across folds)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "09_lgb_feature_importance.png"), dpi=150)
    plt.close()
    print("  Saved: 09_lgb_feature_importance.png")


def plot_model_comparison(
    y: np.ndarray,
    model_r2: dict,
    final_r2: float,
    final_oof: np.ndarray,
    final_preds: np.ndarray,
    fold_scores: dict,
    groups: np.ndarray | None = None,
) -> None:
    """Generate model comparison plots, residual plots, and Day 49 OOF score."""
    _ensure_dir()
    print("  Generating model evaluation plots...")

    names = list(model_r2.keys()) + ["Ensemble"]
    scores = list(model_r2.values()) + [final_r2]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63", "#795548"]

    # 10 Model R2 comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, scores, color=colors[:len(names)], edgecolor="white", width=0.55)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{score:.4f}", ha="center", va="bottom", fontweight="bold",
        )
    ax.set_title("Model R2 Score Comparison (OOF)", fontsize=14, fontweight="bold")
    ax.set_ylabel("R2 Score")
    ax.set_ylim(min(scores) - 0.05, max(scores) + 0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "10_model_comparison.png"), dpi=150)
    plt.close()

    # 11 Actual vs Predicted
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y, bins=60, alpha=0.5, label="Actual (Train)", color="#2196F3", density=True)
    ax.hist(final_preds, bins=60, alpha=0.5, label="Predicted (Test)", color="#FF5722", density=True)
    ax.set_title("Actual vs Predicted Demand Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Demand")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "11_actual_vs_predicted_dist.png"), dpi=150)
    plt.close()

    # 12 Foldwise R2
    fig, ax = plt.subplots(figsize=(10, 5))
    n_folds = max(len(v) for v in fold_scores.values())
    x_pos = np.arange(n_folds)
    width = 0.8 / max(len(fold_scores), 1)
    palette = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    for i, (name, sc_list) in enumerate(fold_scores.items()):
        ax.bar(x_pos + (i - len(fold_scores) / 2) * width + width / 2,
               sc_list, width, label=name, color=palette[i % len(palette)], alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
    ax.set_title("Fold-wise R2 Scores", fontsize=14, fontweight="bold")
    ax.set_ylabel("R2 Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "12_foldwise_r2.png"), dpi=150)
    plt.close()

    # 13 Day-49-only R2 (the realistic test mirror)
    if groups is not None:
        day49_mask = (groups == 49)
        if day49_mask.any():
            day49_r2 = r2_score(y[day49_mask], final_oof[day49_mask])
            print(f"  Day-49-only OOF R2 = {day49_r2:.6f}")
            
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(["Overall OOF R2", "Day-49 OOF R2"], [final_r2, day49_r2], color=["#2196F3", "#E91E63"], width=0.4)
            ax.set_title("Overall vs Day-49 OOF R2 Score", fontsize=14, fontweight="bold")
            ax.set_ylabel("R2 Score")
            for i, val in enumerate([final_r2, day49_r2]):
                ax.text(i, val + 0.005, f"{val:.5f}", ha="center", va="bottom", fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "13_day49_r2.png"), dpi=150)
            plt.close()

    # 14 Residuals vs Predicted
    residuals = y - final_oof
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(final_oof), min(5000, len(final_oof)), replace=False)
    ax.scatter(final_oof[sample_idx], residuals[sample_idx], alpha=0.3, s=8, c="#E91E63")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Residuals vs Predicted Demand (OOF)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Demand")
    ax.set_ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "14_residuals.png"), dpi=150)
    plt.close()
