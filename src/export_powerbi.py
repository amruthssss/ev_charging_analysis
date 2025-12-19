"""Generate Power BI-friendly CSV extracts from the cleaned dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CLEANED_DATA_PATH, POWERBI_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def _load_cleaned_df() -> pd.DataFrame:
    """Load the cleaned dataset with parsed timestamps."""
    date_cols = ["charging_start_time", "charging_end_time"]
    df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=date_cols)
    logger.info("Loaded cleaned dataset for Power BI export: %s", CLEANED_DATA_PATH)
    return df


def export_sessions_cleaned(df: pd.DataFrame) -> Path:
    """Export the full cleaned dataset copy for BI tooling."""
    output_path = POWERBI_DIR / "sessions_cleaned.csv"
    df.to_csv(output_path, index=False)
    logger.info("Wrote sessions_cleaned.csv to %s", output_path)
    return output_path


def export_station_master(df: pd.DataFrame) -> Path:
    """Create a distinct station master file with metadata."""
    columns = [
        "charging_station_id",
        "charging_station_location",
        "charger_type",
        "charging_rate_kw",
    ]
    available_cols = [col for col in columns if col in df.columns]
    station_master = (
        df[available_cols]
        .drop_duplicates()
        .sort_values(by=available_cols)
    )
    output_path = POWERBI_DIR / "station_master.csv"
    station_master.to_csv(output_path, index=False)
    logger.info("Wrote station_master.csv (%s rows) to %s", len(station_master), output_path)
    return output_path


def export_daily_agg(df: pd.DataFrame) -> Path:
    """Aggregate sessions per station per calendar day."""
    if "charging_start_time" not in df.columns:
        raise ValueError("charging_start_time column missing; cannot create daily agg")

    daily = (
        df.assign(session_date=df["charging_start_time"].dt.date)
        .groupby(["charging_station_id", "session_date"], as_index=False)
        .agg(
            total_sessions=("user_id", "count"),
            total_energy_kwh=("energy_consumed_kwh", "sum"),
            avg_duration_minutes=("charging_duration_minutes", "mean"),
        )
    )
    output_path = POWERBI_DIR / "daily_agg.csv"
    daily.to_csv(output_path, index=False)
    logger.info("Wrote daily_agg.csv (%s rows) to %s", len(daily), output_path)
    return output_path


def export_hourly_agg(df: pd.DataFrame) -> Path:
    """Aggregate sessions per station and hour of day."""
    if "start_hour" not in df.columns:
        raise ValueError("start_hour column missing; cannot create hourly agg")

    hourly = (
        df.groupby(["charging_station_id", "start_hour"], as_index=False)
        .agg(
            total_sessions=("user_id", "count"),
            total_energy_kwh=("energy_consumed_kwh", "sum"),
            avg_duration_minutes=("charging_duration_minutes", "mean"),
        )
    )
    output_path = POWERBI_DIR / "hourly_agg.csv"
    hourly.to_csv(output_path, index=False)
    logger.info("Wrote hourly_agg.csv (%s rows) to %s", len(hourly), output_path)
    return output_path


def export_all() -> None:
    """Run all exports sequentially."""
    df = _load_cleaned_df()
    export_sessions_cleaned(df)
    export_station_master(df)
    export_daily_agg(df)
    export_hourly_agg(df)
    logger.info("Finished generating Power BI extracts in %s", POWERBI_DIR)


if __name__ == "__main__":
    export_all()
