"""
Spatial / neighbor / cluster features.

Modified to separate static spatial/clustering features (computed once globally)
from target-dependent neighbor/cluster statistics (computed dynamically inside each CV fold).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Module-level column lists — produced by ``compute_target_dependent_neighbor_features``.
NEIGHBOR_DEPENDENT_COLS: list[str] = [
    "cluster_demand_mean", "cluster_demand_std", "cluster_demand_max", "cluster_demand_min",
    "triple_demand_mean", "triple_demand_std", "geo_weather_demand_mean",
    "neighbor_demand_mean", "neighbor_demand_max", "neighbor_demand_std",
    "neighbor_weighted_mean",
]


def apply_static_neighbor_features(
    train: pd.DataFrame, test: pd.DataFrame, n_clusters: int = 32
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Append target-independent spatial / cluster features and precompute neighbor mappings.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict]
        ``(train, test, neighbor_map)``
    """
    train = train.copy()
    test = test.copy()

    # 1. Clustering based on coordinates. Fit on the union of train+test
    # geohashes so test-only locations get real cluster assignments
    # (0..n_clusters-1) rather than the -1 sentinel.
    geo_centers = (
        pd.concat(
            [
                train[["geohash", "latitude", "longitude"]],
                test[["geohash", "latitude", "longitude"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=["geohash"])
        .reset_index(drop=True)
    )
    if len(geo_centers) >= n_clusters:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km.fit(geo_centers[["latitude", "longitude"]])
        geo_centers["geo_cluster"] = km.labels_
    else:
        geo_centers["geo_cluster"] = 0

    train = train.merge(geo_centers[["geohash", "geo_cluster"]], on="geohash", how="left")
    test = test.merge(geo_centers[["geohash", "geo_cluster"]], on="geohash", how="left")
    train["geo_cluster"] = train["geo_cluster"].fillna(-1).astype(int)
    test["geo_cluster"] = test["geo_cluster"].fillna(-1).astype(int)
    
    cluster_sizes = geo_centers["geo_cluster"].value_counts().to_dict()
    train["cluster_size"] = train["geo_cluster"].map(cluster_sizes).fillna(0)
    test["cluster_size"] = test["geo_cluster"].map(cluster_sizes).fillna(0)

    # Distance from each row to its own cluster centroid
    cluster_centroids = geo_centers.groupby("geo_cluster")[["latitude", "longitude"]].mean().reset_index()
    cluster_centroids.columns = ["geo_cluster", "centroid_lat", "centroid_lon"]
    
    train = train.merge(cluster_centroids, on="geo_cluster", how="left")
    test = test.merge(cluster_centroids, on="geo_cluster", how="left")
    
    for df in (train, test):
        df["cluster_dist"] = np.hypot(
            df["latitude"] - df["centroid_lat"].fillna(0),
            df["longitude"] - df["centroid_lon"].fillna(0)
        )
        df.drop(columns=["centroid_lat", "centroid_lon"], inplace=True)

    # 2. Precompute neighbor map (nearest 6 geohashes by distance)
    coords = geo_centers.set_index("geohash")[["latitude", "longitude"]]
    geohash_arr = coords.index.to_numpy()
    lat_arr = coords["latitude"].to_numpy()
    lon_arr = coords["longitude"].to_numpy()

    diff_lat = lat_arr[:, None] - lat_arr[None, :]
    diff_lon = lon_arr[:, None] - lon_arr[None, :]
    dist = np.hypot(diff_lat, diff_lon)
    k = min(6, len(geohash_arr) - 1)  # 6 true neighbors, no self
    nbr_idx = np.argsort(dist, axis=1)[:, 1:k+1]   # skip index 0 = self

    neighbor_map = {}
    for i, gh in enumerate(geohash_arr):
        neighbor_map[gh] = [(geohash_arr[j], dist[i, j]) for j in nbr_idx[i]]

    return train, test, neighbor_map


def compute_target_dependent_neighbor_features(
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    test_df: pd.DataFrame,
    neighbor_map: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute and map target-dependent neighbor and cluster features inside CV loop."""
    train_fold = train_fold.copy()
    val_fold = val_fold.copy()
    test_df = test_df.copy()

    # ------- Cluster-level demand stats
    cluster_stats = train_fold.groupby("geo_cluster")["demand"].agg(["mean", "std", "max", "min"])
    cluster_stats.columns = [
        "cluster_demand_mean", "cluster_demand_std",
        "cluster_demand_max", "cluster_demand_min",
    ]
    global_median_cluster = cluster_stats.median().to_dict()
    for col in cluster_stats.columns:
        fallback = global_median_cluster.get(col, 0.0)
        train_fold[col] = train_fold["geo_cluster"].map(cluster_stats[col]).fillna(fallback)
        val_fold[col] = val_fold["geo_cluster"].map(cluster_stats[col]).fillna(fallback)
        test_df[col] = test_df["geo_cluster"].map(cluster_stats[col]).fillna(fallback)

    # ------- (Weather, RoadType, time_slot) mean — vectorised via merge (P1)
    triple_stats = (
        train_fold.groupby(["Weather", "RoadType", "time_slot"])["demand"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "triple_demand_mean", "std": "triple_demand_std"})
    )

    global_triple_mean = triple_stats["triple_demand_mean"].mean() if not triple_stats.empty else 0.0
    global_triple_std = triple_stats["triple_demand_std"].mean() if not triple_stats.empty else 0.0

    triple_merge_cols = ["Weather", "RoadType", "time_slot",
                         "triple_demand_mean", "triple_demand_std"]
    train_fold = train_fold.merge(triple_stats[triple_merge_cols],
                                  on=["Weather", "RoadType", "time_slot"], how="left")
    val_fold = val_fold.merge(triple_stats[triple_merge_cols],
                              on=["Weather", "RoadType", "time_slot"], how="left")
    test_df = test_df.merge(triple_stats[triple_merge_cols],
                            on=["Weather", "RoadType", "time_slot"], how="left")
    train_fold["triple_demand_mean"] = train_fold["triple_demand_mean"].fillna(global_triple_mean)
    train_fold["triple_demand_std"] = train_fold["triple_demand_std"].fillna(global_triple_std)
    val_fold["triple_demand_mean"] = val_fold["triple_demand_mean"].fillna(global_triple_mean)
    val_fold["triple_demand_std"] = val_fold["triple_demand_std"].fillna(global_triple_std)
    test_df["triple_demand_mean"] = test_df["triple_demand_mean"].fillna(global_triple_mean)
    test_df["triple_demand_std"] = test_df["triple_demand_std"].fillna(global_triple_std)

    # ------- (geohash, Weather) mean — vectorised via merge (P2)
    gw_stats = (
        train_fold.groupby(["geohash", "Weather"])["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "geo_weather_demand_mean"})
    )
    global_gw_mean = train_fold["demand"].mean()

    gw_merge_cols = ["geohash", "Weather", "geo_weather_demand_mean"]
    train_fold = train_fold.merge(gw_stats[gw_merge_cols],
                                  on=["geohash", "Weather"], how="left")
    val_fold = val_fold.merge(gw_stats[gw_merge_cols],
                              on=["geohash", "Weather"], how="left")
    test_df = test_df.merge(gw_stats[gw_merge_cols],
                            on=["geohash", "Weather"], how="left")
    train_fold["geo_weather_demand_mean"] = train_fold["geo_weather_demand_mean"].fillna(global_gw_mean)
    val_fold["geo_weather_demand_mean"] = val_fold["geo_weather_demand_mean"].fillna(global_gw_mean)
    test_df["geo_weather_demand_mean"] = test_df["geo_weather_demand_mean"].fillna(global_gw_mean)

    # ------- Neighbor mean/max demand (vectorised via numpy — P3)
    geo_means_s = train_fold.groupby("geohash")["demand"].mean()
    geo_means = geo_means_s.to_dict()
    global_geo_mean = train_fold["demand"].mean()

    gh_list = list(neighbor_map.keys())
    nbr_matrix = np.array(
        [[geo_means.get(nb, global_geo_mean) for nb, _ in neighbor_map[gh]]
         for gh in gh_list],
        dtype=np.float64,
    )
    dist_matrix = np.array(
        [[d for _, d in neighbor_map[gh]] for gh in gh_list],
        dtype=np.float64,
    )
    w_matrix = 1.0 / (dist_matrix + 1e-5)
    w_sum = w_matrix.sum(axis=1, keepdims=True)

    nbr_mean_s = pd.Series(nbr_matrix.mean(axis=1), index=gh_list)
    nbr_max_s = pd.Series(nbr_matrix.max(axis=1), index=gh_list)
    nbr_std_s = pd.Series(nbr_matrix.std(axis=1), index=gh_list)
    nbr_wt_mean_s = pd.Series(
        (nbr_matrix * w_matrix).sum(axis=1) / w_sum.flatten(),
        index=gh_list,
    )

    for df in (train_fold, val_fold, test_df):
        df["neighbor_demand_mean"] = df["geohash"].map(nbr_mean_s).fillna(global_geo_mean)
        df["neighbor_demand_max"] = df["geohash"].map(nbr_max_s).fillna(global_geo_mean)
        df["neighbor_demand_std"] = df["geohash"].map(nbr_std_s).fillna(0.0)
        df["neighbor_weighted_mean"] = df["geohash"].map(nbr_wt_mean_s).fillna(global_geo_mean)

    # ------- Fill all NaNs defensively
    new_cols = [
        "cluster_demand_mean", "cluster_demand_std",
        "cluster_demand_max", "cluster_demand_min",
        "triple_demand_mean", "triple_demand_std",
        "geo_weather_demand_mean",
        "neighbor_demand_mean", "neighbor_demand_max",
        "neighbor_demand_std", "neighbor_weighted_mean"
    ]
    for c in new_cols:
        med = train_fold[c].median() if train_fold[c].notna().any() else 0.0
        train_fold[c] = train_fold[c].fillna(med)
        val_fold[c] = val_fold[c].fillna(med)
        test_df[c] = test_df[c].fillna(med)

    return train_fold, val_fold, test_df
