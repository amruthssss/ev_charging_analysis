"""Phase 5 Model 3: Cluster stations by behavioral features."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLEANED_DATA_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.logger import get_logger

logger = get_logger(__name__)
SEGMENT_FEATURES: List[str] = [
    "session_count",
    "unique_users",
    "avg_energy_kwh",
    "avg_duration_min",
    "avg_cost_usd",
    "avg_distance_km",
    "avg_temperature_c",
    "dc_fast_ratio",
    "commuter_ratio",
]
CLUSTER_OUTPUT_PATH: Path = PROCESSED_DATA_DIR / "station_clusters.csv"
MODEL_CLUSTER_OUTPUT_PATH: Path = MODELS_DIR / "station_clusters.csv"
CLUSTER_PROFILE_PATH: Path = PROCESSED_DATA_DIR / "station_cluster_profile.csv"
CLUSTER_REPORT_PATH: Path = REPORTS_DIR / "cluster_analysis.md"


def _utc_timestamp() -> str:
    """Generate a timezone-aware UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_station_source() -> pd.DataFrame:
    """Return the raw session data that feeds segmentation."""
    df = pd.read_csv(CLEANED_DATA_PATH)
    logger.info("Loaded cleaned sessions for clustering with shape %s", df.shape)
    return df


def aggregate_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-station features used by K-Means."""
    grouped = df.groupby("charging_station_id")  # Work one station at a time regardless of chronological order.
    features = grouped.agg(
        session_count=("user_id", "size"),
        unique_users=("user_id", "nunique"),
        avg_energy_kwh=("energy_consumed_kwh", "mean"),
        avg_duration_min=("charging_duration_minutes", "mean"),
        avg_cost_usd=("charging_cost_usd", "mean"),
        avg_distance_km=("distance_driven_since_last_charge_km", "mean"),
        avg_temperature_c=("temperature_degc", "mean"),
    )
    dc_fast_ratio = grouped["charger_type"].apply(lambda s: (s == "DC Fast Charger").mean())  # Share of sessions that used DC fast hardware.
    commuter_ratio = grouped["user_type"].apply(lambda s: (s == "Commuter").mean())  # Helpful behavioral proxy for usage intent.
    features = features.join(dc_fast_ratio.rename("dc_fast_ratio"))
    features = features.join(commuter_ratio.rename("commuter_ratio"))
    features = features.reset_index()
    features = features.fillna(0.0)  # Drop-in replacements keep scaler/KMeans from seeing NaNs.
    logger.info("Built station feature table with shape %s", features.shape)
    return features


def run_station_segmentation(n_clusters: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster stations and return labeled dataframe plus cluster profile."""
    df = load_station_source()
    features = aggregate_station_features(df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[SEGMENT_FEATURES])  # Center/scale so KMeans treats each metric equally.
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(scaled)
    features["cluster"] = labels  # Attach assignments for CSV/reporting.
    cluster_summary = (
        features.groupby("cluster")[SEGMENT_FEATURES]
        .mean()
        .round(2)
        .reset_index()
    )  # Produces per-cluster centroids expressed in business units for stakeholders.
    cluster_counts = features["cluster"].value_counts().sort_index()
    logger.info("Cluster counts:\n%s", cluster_counts)
    return features, cluster_summary


def persist_segmentation_results(labels: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Save clustering outputs to CSV for downstream consumption."""
    labels.to_csv(CLUSTER_OUTPUT_PATH, index=False)
    labels.to_csv(MODEL_CLUSTER_OUTPUT_PATH, index=False)  # Mirror labels into models/ for downstream consumers.
    summary.to_csv(CLUSTER_PROFILE_PATH, index=False)
    logger.info(
        "Saved station clusters to %s and %s, profile to %s",
        CLUSTER_OUTPUT_PATH,
        MODEL_CLUSTER_OUTPUT_PATH,
        CLUSTER_PROFILE_PATH,
    )


def write_cluster_report(labels: pd.DataFrame, summary: pd.DataFrame) -> Path:
    """Record segmentation highlights for quick review."""
    lines: List[str] = ["# Phase 5 Model 3 - Station Segmentation", ""]
    lines.append(f"- Run timestamp (UTC): {_utc_timestamp()}")
    lines.append(f"- Stations segmented: {len(labels)}")
    lines.append(f"- Clusters discovered: {summary['cluster'].nunique()}")
    lines.append(f"- Cluster labels export: {MODEL_CLUSTER_OUTPUT_PATH}")
    lines.append(f"- Cluster profile export: {CLUSTER_PROFILE_PATH}")
    lines.append("")

    counts = labels["cluster"].value_counts().sort_index()
    lines.append("## Cluster Counts")  # Quick health check to ensure clusters are balanced enough for actionability.
    lines.append("| Cluster | Stations |")
    lines.append("| --- | ---: |")
    for cluster_id, count in counts.items():
        lines.append(f"| {cluster_id} | {int(count)} |")

    lines.append("")
    lines.append("## Feature Averages")  # Helps operations teams interpret behavioral signatures per cluster.
    lines.append("| " + " | ".join(["Cluster"] + SEGMENT_FEATURES) + " |")
    header_alignment = "| --- | " + " | ".join(["---:"] * len(SEGMENT_FEATURES)) + " |"
    lines.append(header_alignment)
    for _, row in summary.sort_values("cluster").iterrows():
        values = [f"{row[feature]:.2f}" for feature in SEGMENT_FEATURES]
        lines.append("| " + " | ".join([str(int(row["cluster"]))] + values) + " |")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTER_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote cluster analysis report to %s", CLUSTER_REPORT_PATH)
    return CLUSTER_REPORT_PATH


if __name__ == "__main__":
    labeled, profile = run_station_segmentation()
    persist_segmentation_results(labeled, profile)
    write_cluster_report(labeled, profile)
    logger.info("Cluster profile:\n%s", profile)
