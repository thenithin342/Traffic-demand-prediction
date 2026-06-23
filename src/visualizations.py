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

from config import PLOTS_DIR, N_FOLDS

sns.set_theme(style="whitegrid", palette="viridis")


def generate_eda_plots(train: pd.DataFrame, y: np.ndarray):
    """Generate all Exploratory Data Analysis plots."""
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
        alpha=0.3,
        s=8,
        c="#673AB7",
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
        ax.set_title("Demand Heatmap: NumberofLanes × Hour", fontsize=14, fontweight="bold")
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
            geo_mean["longitude"],
            geo_mean["latitude"],
            c=geo_mean["demand"],
            cmap="hot_r",
            s=12,
            alpha=0.8,
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
        "geo_demand_mean", "geo_demand_std", "timeslot_demand_mean"
    ]
    top_feat_cols = [c for c in top_feat_cols if c in train.columns]
    corr_data = train[top_feat_cols + ["demand"]].corr()
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(
        corr_data, mask=mask, cmap="coolwarm", annot=True, fmt=".2f",
        ax=ax, linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 7}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "08_correlation_heatmap.png"), dpi=150)
    plt.close()


def plot_feature_importance(features: list, importances: np.ndarray):
    """Plot LightGBM top feature importances."""
    fi = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    fig, ax = plt.subplots(figsize=(10, 12))
    top_n = min(30, len(fi))
    ax.barh(range(top_n), fi["importance"].values[:top_n], color="#4CAF50")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(fi["feature"].values[:top_n])
    ax.invert_yaxis()
    ax.set_title("LightGBM Feature Importance (Top 30)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "09_lgb_feature_importance.png"), dpi=150)
    plt.close()
    print("  Saved: 09_lgb_feature_importance.png")


def plot_model_comparison(
    y: np.ndarray,
    lgb_r2: float,
    xgb_r2: float,
    cat_r2: float,
    final_r2: float,
    final_preds: np.ndarray,
    lgb_scores: list,
    xgb_scores: list,
    cat_scores: list,
):
    """Generate model comparison and residual plots."""
    print("  Generating model evaluation plots...")
    
    # 10 Model R2 comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["LightGBM", "XGBoost", "CatBoost", "Ensemble"]
    r2_scores = [lgb_r2, xgb_r2, cat_r2, final_r2]
    colors_bar = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
    bars = ax.bar(models, r2_scores, color=colors_bar, edgecolor="white", width=0.5)
    for bar, score in zip(bars, r2_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Model R² Score Comparison (OOF)", fontsize=14, fontweight="bold")
    ax.set_ylabel("R² Score")
    ax.set_ylim(min(r2_scores) - 0.05, max(r2_scores) + 0.03)
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
    x_pos = np.arange(N_FOLDS)
    width = 0.25
    ax.bar(x_pos - width, lgb_scores, width, label="LightGBM", color="#4CAF50", alpha=0.85)
    ax.bar(x_pos, xgb_scores, width, label="XGBoost", color="#2196F3", alpha=0.85)
    ax.bar(x_pos + width, cat_scores, width, label="CatBoost", color="#FF9800", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(N_FOLDS)])
    ax.set_title("Fold-wise R² Scores", fontsize=14, fontweight="bold")
    ax.set_ylabel("R² Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "12_foldwise_r2.png"), dpi=150)
    plt.close()
