# src/eda.py
"""Quick EDA helpers for the EV charging dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_FILENAME
from src.data_loader import load_raw_data
from src.logger import get_logger


logger = get_logger(__name__)


EDA_OUTPUT_DIR = PROCESSED_DATA_DIR / "eda"
EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLUMNS = [
    "battery_capacity_kwh",
    "energy_consumed_kwh",
    "charging_duration_hours",
    "charging_rate_kw",
    "charging_cost_usd",
    "state_of_charge_start_pct",
    "state_of_charge_end_pct",
    "distance_driven_since_last_charge_km",
    "temperature_degc",
    "vehicle_age_years",
]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply snake_case naming so downstream code can rely on stable labels."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "pct", regex=False)
        .str.replace("°", "deg", regex=False)
        .str.replace("/", "_per_", regex=False)
    )
    return df


def load_dataset(filename: str = "ev_charging_data.csv") -> pd.DataFrame:
    """Load raw CSV, normalize column names, and parse timestamps."""
    logger.info("Loading dataset: %s", filename)
    df = load_raw_data(filename)
    df = _standardize_columns(df)

    for col in ["charging_start_time", "charging_end_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force known numeric fields to float, tolerating bad strings."""
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive fields used repeatedly across analytics steps."""
    df = df.copy()
    if "charging_start_time" in df.columns:
        df["start_hour"] = df["charging_start_time"].dt.hour
        df["day_of_week"] = df["charging_start_time"].dt.day_name()
    if "charging_duration_hours" in df.columns:
        df["charging_duration_minutes"] = df["charging_duration_hours"] * 60
    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline used before analytics and modeling."""
    df = _coerce_numeric_columns(df)
    df = _add_time_features(df)
    required_cols = ["energy_consumed_kwh", "charging_duration_hours"]
    present_required = [col for col in required_cols if col in df.columns]
    if present_required:
        df = df.dropna(subset=present_required)
    return df


def report_missing_values(df: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    """Return a table with missing counts and percentages, optionally saving."""
    counts = df.isna().sum()
    pct = df.isna().mean() * 100
    report = pd.DataFrame({
        "missing_count": counts,
        "missing_pct": pct
    }).sort_values("missing_pct", ascending=False)
    if output_path is not None:
        report.to_csv(output_path)
    return report


def describe_numeric(df: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    """Return summary statistics for numeric columns, optionally saving."""
    num_cols = df.select_dtypes(include=["number"]).columns
    summary = df[num_cols].describe()
    if output_path is not None:
        summary.to_csv(output_path)
    return summary


def describe_categorical(df: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    """Return summary statistics for categorical columns, optionally saving."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        return pd.DataFrame()  # nothing categorical
    summary = df[cat_cols].describe()
    if output_path is not None:
        summary.to_csv(output_path)
    return summary


def plot_peak_hours(df: pd.DataFrame):
    """Bar chart of number of sessions per charging start hour."""
    if "charging_start_time" not in df.columns:
        logger.warning("No charging_start_time column found; skipping peak-hour plot.")
        return

    df = df.copy()
    df["start_hour"] = df["charging_start_time"].dt.hour
    sessions_by_hour = df["start_hour"].value_counts().sort_index()

    plt.figure()
    sessions_by_hour.plot(kind="bar")
    plt.xlabel("Start hour")
    plt.ylabel("Number of sessions")
    plt.title("Sessions per start hour")
    plt.tight_layout()
    plt.show()


def plot_energy_by_station(df: pd.DataFrame, top_n: int = 10):
    """Bar chart of total energy (kWh) by station (top N)."""
    required_cols = {"charging_station_id", "energy_consumed_kwh"}
    if not required_cols.issubset(df.columns):
        logger.warning("Missing columns %s for energy-by-station plot.", required_cols)
        return

    energy_by_station = (
        df.groupby("charging_station_id")["energy_consumed_kwh"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    plt.figure()
    energy_by_station.plot(kind="bar")
    plt.xlabel("Station ID")
    plt.ylabel("Total energy (kWh)")
    plt.title(f"Top {top_n} stations by total energy")
    plt.tight_layout()
    plt.show()


def plot_station_usage(df: pd.DataFrame, top_n: int = 10):
    """Bar chart of number of sessions per station (top N)."""
    if "charging_station_id" not in df.columns:
        logger.warning("Need charging_station_id column for station-usage plot.")
        return

    sessions_per_station = df["charging_station_id"].value_counts().head(top_n)

    plt.figure()
    sessions_per_station.plot(kind="bar")
    plt.xlabel("Station ID")
    plt.ylabel("Number of sessions")
    plt.title(f"Top {top_n} stations by session count")
    plt.tight_layout()
    plt.show()


def save_clean_dataset(df: pd.DataFrame, filename: str = "cleaned_ev_sessions.csv") -> Path:
    """Persist the cleaned dataset to data/processed for downstream phases."""
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    df = load_dataset(RAW_DATA_FILENAME)
    df = prepare_dataset(df)

    logger.info("Generating missing-value report...")
    logger.info("\n%s", report_missing_values(df, EDA_OUTPUT_DIR / "missing_values.csv"))

    logger.info("Generating numeric summary...")
    logger.info("\n%s", describe_numeric(df, EDA_OUTPUT_DIR / "numeric_summary.csv"))

    logger.info("Generating categorical summary...")
    cat_summary = describe_categorical(df, EDA_OUTPUT_DIR / "categorical_summary.csv")
    logger.info("\n%s", cat_summary)

    logger.info("Rendering peak-hour plot...")
    plot_peak_hours(df)

    logger.info("Rendering energy-by-station plot...")
    plot_energy_by_station(df)

    logger.info("Rendering station-usage plot...")
    plot_station_usage(df)

    cleaned_path = save_clean_dataset(df)
    logger.info("Cleaned dataset saved to %s", cleaned_path)
