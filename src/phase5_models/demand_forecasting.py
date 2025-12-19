"""Utilities for Phase 5 Model 1 (demand forecasting)."""  # Describe the module purpose.

from __future__ import annotations  # Enable postponed evaluation of type hints for forward references.

from datetime import datetime, timezone  # Stamp artifacts with the run timestamp.
from pathlib import Path  # Provide filesystem path utilities that work across OSes.
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd  # Use pandas for tabular data handling needed by Prophet later on.
from joblib import dump
from matplotlib import pyplot as plt  # Provide simple plotting for forecast visualization.
from prophet import Prophet  # Bring in Prophet for univariate daily forecasting.

from src.config import MODELS_DIR, POWERBI_DIR, REPORTS_DIR  # Shared project paths.
from src.logger import get_logger  # Import the shared logging utility for consistent logs.

logger = get_logger(__name__)  # Instantiate a module-level logger tied to this file's namespace.
DAILY_AGG_FILENAME = "daily_agg.csv"  # Constant for the specific aggregate file we keep reusing.
FORECAST_MODEL_PATH = MODELS_DIR / "forecast_model.pkl"
FORECAST_REPORT_PATH = REPORTS_DIR / "forecast_validation.md"


def _utc_timestamp() -> str:
    """Return a timezone-aware ISO timestamp suitable for metadata."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_daily_agg_path() -> Path:
    """Return the absolute path to data/processed/powerbi/daily_agg.csv."""  # Clarify what the helper yields.
    return POWERBI_DIR / DAILY_AGG_FILENAME  # Combine base directory with filename to build full path.


def load_daily_sessions(parse_dates: bool = True) -> pd.DataFrame:
    """Load the aggregated daily sessions used for demand forecasting."""  # Uses powerBI daily aggregates produced in earlier phases.
    csv_path = get_daily_agg_path()  # Resolve the CSV location using the helper for consistency.
    if not csv_path.exists():  # Guard against missing data before attempting to read.
        logger.error("Missing daily aggregate file at %s", csv_path)  # Emit a clear error for debugging.
        raise FileNotFoundError(f"Missing daily aggregate file: {csv_path}")  # Fail fast when data is absent.

    # Allow callers (mainly tests) to disable date parsing when they want to control dtypes precisely.
    read_kwargs = {"parse_dates": ["session_date"]} if parse_dates else {}  # Optionally parse date column when requested.
    df = pd.read_csv(csv_path, **read_kwargs)  # Load the CSV into memory with the configured parsing options.
    logger.info("Loaded %s with shape %s", csv_path.name, df.shape)  # Log shape to confirm record counts.
    return df  # Return the DataFrame for downstream processing.


def inspect_daily_sessions(df: pd.DataFrame) -> Dict[str, object]:
    """Log and return the columns, dtypes, and key counts for sanity checks."""  # Explain the inspection scope.
    # Most downstream helpers assume the dataset contains particular columns and temporal coverage,
    # so we capture a structural snapshot (names, dtypes, date span, station counts) up front.
    summary: Dict[str, object] = {  # Build a dictionary capturing structural metadata for quick review.
        "columns": df.columns.tolist(),  # Store ordered list of column names to mirror the schema.
        "dtypes": df.dtypes.astype(str).to_dict(),  # Convert pandas dtypes to plain strings for JSON friendliness.
        "station_count": int(df["charging_station_id"].nunique()) if "charging_station_id" in df.columns else 0,  # Count unique stations when column exists.
        "date_range": _get_date_range(df),  # Capture min/max dates to verify temporal coverage.
    }

    logger.info("Columns: %s", ", ".join(summary["columns"]))  # Log the column sequence for visibility.
    logger.info("Data types:\n%s", df.dtypes)  # Emit native dtype info for each column.
    start_date, end_date = summary["date_range"]  # Unpack the tuple for logging convenience.
    logger.info(  # Aggregate structured logging for counts and coverage.
        "Stations: %s | Date range: %s -> %s",
        summary["station_count"],
        start_date,
        end_date,
    )
    return summary  # Provide the summary dict to callers that need programmatic access.


def _get_date_range(df: pd.DataFrame) -> Tuple[object, object]:
    if "session_date" not in df.columns:  # Return empty sentinels when the time column is absent.
        return (None, None)  # Explicitly state that no range could be computed.
    return (df["session_date"].min(), df["session_date"].max())  # Use pandas to extract earliest and latest dates.


def prepare_station_timeseries(df: pd.DataFrame, station_id: Union[str, int]) -> pd.DataFrame:
    """Subset one station and rename columns to Prophet's ds/y convention."""  # Document the goal of this helper.
    station_df = df[df["charging_station_id"] == station_id].copy()  # Filter rows for the requested station.
    if station_df.empty:  # Guard against typos or stations without history.
        raise ValueError(f"No records found for station_id={station_id}")  # Provide actionable failure info.
    station_df = station_df.sort_values("session_date")  # Ensure chronological order for forecasting models.
    station_df = station_df.rename(columns={"session_date": "ds", "total_sessions": "y"})  # Align column names with Prophet API.
    tidy_df = station_df[["ds", "y"]].reset_index(drop=True)  # Drop unused metrics for a lean training frame.
    if len(tidy_df) < 2:
        raise ValueError(f"Need at least 2 rows for station_id={station_id}")
    logger.info("Prepared %s rows for station %s", len(tidy_df), station_id)  # Log record count for transparency.
    return tidy_df  # Return the ds/y pair ready for Prophet.


def train_prophet_on_station(ts_df: pd.DataFrame) -> Prophet:
    """Fit a basic daily Prophet model on a single-station time series."""  # Summarize the modeling step.
    # Keep Prophet configuration intentionally conservative: sparse, noisy datasets can overfit when
    # changepoint or seasonal flexibility is too high. These settings mirror the exploratory analysis findings.
    model = Prophet(
        growth="linear",  # Assume linear trend because we lack long history for saturation effects.
        daily_seasonality=False,  # Disable automatic daily seasonality due to daily aggregates already.
        weekly_seasonality=True,  # Keep weekly seasonality to capture weekday vs weekend demand.
        yearly_seasonality=False,  # Yearly seasonality is unnecessary for two months of data.
        changepoint_prior_scale=0.2,  # Keep modest flexibility to avoid overfitting sparse points.
    )
    model.fit(ts_df)  # Train the model; Prophet expects columns named ds and y.
    logger.info("Prophet training completed with %s observations", len(ts_df))  # Confirm completion in logs.
    return model  # Return the fitted estimator for forecasting in Step 3.


def forecast_next_30_days(model: Prophet) -> pd.DataFrame:
    """Generate a 30-day daily forecast using a fitted Prophet model."""  # Explain the forecasting window.
    future = model.make_future_dataframe(periods=30, freq="D")  # Extend the training calendar by 30 days beyond observed history.
    forecast = model.predict(future)  # Let Prophet produce yhat, trend, and interval columns.
    logger.info("Generated forecast covering %s rows", len(forecast))  # Capture total points for traceability.
    return forecast  # Return the full Prophet forecast table for downstream plotting.


def plot_forecast(ts_df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    """Plot history plus 30-day forecast with confidence intervals."""
    plt.figure(figsize=(8, 4))  # Create a compact figure suitable for notebooks or reports.
    plt.plot(ts_df["ds"], ts_df["y"], label="History", marker="o")  # Markers highlight sparse historical points.
    plt.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast", color="tab:orange")  # Forecast line extends history.
    plt.fill_between(
        forecast_df["ds"],
        forecast_df["yhat_lower"],
        forecast_df["yhat_upper"],
        color="tab:orange",
        alpha=0.2,
        label="Confidence",
    )
    plt.axvline(ts_df["ds"].max(), color="gray", linestyle="--", linewidth=1)  # Visual separator between history and future.
    plt.title("Station demand forecast (30 days)")
    plt.xlabel("Date")
    plt.ylabel("Sessions per day")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_station_model(station_id: str, ts_df: pd.DataFrame, model: Prophet) -> Dict[str, float]:
    """Compute quick in-sample metrics to track forecast quality per station."""
    history_forecast = model.predict(ts_df[["ds"]])  # In-sample pass to quantify historical fit.
    actual = ts_df["y"].to_numpy(dtype=float)
    predicted = history_forecast["yhat"].to_numpy(dtype=float)
    errors = actual - predicted  # Residuals capture per-day over/under predictions.
    mae = float(np.mean(np.abs(errors)))  # Average absolute deviation for intuitive error magnitude.
    rmse = float(np.sqrt(np.mean(errors ** 2)))  # Penalizes larger spikes more strongly than MAE.
    denominator = np.where(actual == 0, np.nan, np.abs(actual))  # Avoid division-by-zero when computing MAPE.
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.nanmean(np.abs(errors) / denominator))
    metrics = {
        "station_id": station_id,
        "train_rows": len(ts_df),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
    logger.info(
        "Station %s -> rows=%d | MAE=%.2f | RMSE=%.2f | MAPE=%s",
        station_id,
        len(ts_df),
        mae,
        rmse,
        "nan" if np.isnan(mape) else f"{mape:.3f}",
    )
    return metrics


def train_models_for_all_stations(df: pd.DataFrame) -> Tuple[Dict[str, Prophet], List[Dict[str, float]]]:
    """Train Prophet for every station and collect evaluation metrics."""
    models: Dict[str, Prophet] = {}  # Holds every station's trained Prophet model.
    metrics: List[Dict[str, float]] = []  # Mirrors `models` with numeric health checks per station.
    station_ids = df["charging_station_id"].dropna().unique().tolist()  # Preserve native dtype (string/object) from CSV.
    # Iterate through the discovered IDs, training a self-contained model per location so deployments can cherry pick stations.
    for station_value in station_ids:
        station_key = str(station_value)
        try:
            ts_df = prepare_station_timeseries(df, station_value)
        except ValueError as exc:  # Skip stations without records.
            logger.warning("Skipping station %s (%s)", station_key, exc)
            continue
        model = train_prophet_on_station(ts_df)
        models[station_key] = model  # Retain the fitted estimator in-memory for serialization.
        metrics.append(evaluate_station_model(station_key, ts_df, model))  # Maintain per-station diagnostics for reporting.
    logger.info("Trained Prophet models for %d stations", len(models))
    return models, metrics


def persist_forecast_models(model_registry: Dict[str, Prophet]) -> Path:
    """Persist trained Prophet models to the models directory."""
    if not model_registry:
        raise ValueError("No models available to persist.")
    # Package metadata with the raw estimators so the artifact is self-describing when loaded weeks later.
    payload = {
        "saved_at_utc": _utc_timestamp(),
        "model_type": "prophet",
        "stations": sorted(model_registry.keys()),
        "models": model_registry,  # joblib handles serializing the fitted Prophet objects.
    }
    dump(payload, FORECAST_MODEL_PATH)
    logger.info("Saved Prophet registry to %s", FORECAST_MODEL_PATH)
    return FORECAST_MODEL_PATH


def write_forecast_report(metrics: List[Dict[str, float]]) -> Path:
    """Create a markdown validation summary for Phase 5 Model 1."""
    lines: List[str] = ["# Phase 5 Model 1 - Demand Forecasting", ""]
    lines.append(f"- Run timestamp (UTC): {_utc_timestamp()}")
    lines.append(f"- Stations trained: {len(metrics)}")
    lines.append("")

    if metrics:
        mae_values = [m["mae"] for m in metrics if not np.isnan(m["mae"])]  # Filter out nan placeholders.
        rmse_values = [m["rmse"] for m in metrics if not np.isnan(m["rmse"])]
        median_mae = float(np.median(mae_values)) if mae_values else float("nan")
        median_rmse = float(np.median(rmse_values)) if rmse_values else float("nan")
        lines.append(f"- Median MAE: {median_mae:.2f}" if not np.isnan(median_mae) else "- Median MAE: n/a")
        lines.append(f"- Median RMSE: {median_rmse:.2f}" if not np.isnan(median_rmse) else "- Median RMSE: n/a")
        lines.append("")
        lines.extend([
            "| Station | Training Rows | MAE | RMSE | MAPE |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        sorted_metrics = sorted(metrics, key=lambda m: m["mae"])  # Bubble up best-performing stations to the top of the table.
        for metric in sorted_metrics:
            mae = f"{metric['mae']:.2f}" if not np.isnan(metric["mae"]) else "n/a"
            rmse = f"{metric['rmse']:.2f}" if not np.isnan(metric["rmse"]) else "n/a"
            mape_val = metric["mape"]
            mape = f"{mape_val:.3f}" if not np.isnan(mape_val) else "n/a"
            lines.append(
                f"| {metric['station_id']} | {metric['train_rows']} | {mae} | {rmse} | {mape} |"
            )
    else:
        lines.append("No stations were trained. Ensure daily aggregates exist before rerunning.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote forecast validation report to %s", FORECAST_REPORT_PATH)
    return FORECAST_REPORT_PATH


def run_full_forecast_pipeline(plot_sample: bool = False) -> Dict[str, object]:
    """Execute the Phase 5 Model 1 workflow end-to-end."""
    # The orchestrator mirrors the sequential notebook steps (load -> inspect -> train -> persist -> report)
    # so stakeholders can re-run the entire forecasting deliverable with a single importable function.
    dataset = load_daily_sessions()
    inspect_daily_sessions(dataset)
    model_registry, metrics = train_models_for_all_stations(dataset)  # Train-and-evaluate every eligible station.
    if not model_registry:
        raise RuntimeError("No Prophet models were trained. Check input data.")
    persist_forecast_models(model_registry)
    write_forecast_report(metrics)

    sample_station_id = next(iter(model_registry.keys()))
    plot_payload = None
    if plot_sample:
        station_series = prepare_station_timeseries(dataset, sample_station_id)  # Recreate ds/y pairs for the sample
        forecast_df = forecast_next_30_days(model_registry[sample_station_id])
        plot_forecast(station_series, forecast_df)  # Quick visual to sanity-check the selected model.
        plot_payload = {
            "station_id": sample_station_id,
            "history_rows": len(station_series),
        }

    return {
        "stations_trained": len(model_registry),
        "metrics": metrics,
        "sample_plot": plot_payload,  # Useful when this function is imported and invoked programmatically.
    }


if __name__ == "__main__":
    run_full_forecast_pipeline()
