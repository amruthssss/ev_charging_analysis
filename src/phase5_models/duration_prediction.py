"""Phase 5 Model 2: tuned duration prediction pipeline."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    CLEANED_DATA_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.logger import get_logger

logger = get_logger(__name__)
TARGET_COLUMN = "charging_duration_minutes"
RESIDUALS_OUTPUT_PATH = PROCESSED_DATA_DIR / "station_duration_residuals.csv"
STATION_CLUSTER_PATH = PROCESSED_DATA_DIR / "station_clusters.csv"
DURATION_MODEL_PATH = MODELS_DIR / "duration_model.pkl"
DURATION_REPORT_PATH = REPORTS_DIR / "duration_model_metrics.md"


def _utc_timestamp() -> str:
    """Return a timezone-aware ISO 8601 timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

BASE_NUMERIC_FEATURES: List[str] = [
    "battery_capacity_kwh",
    "energy_consumed_kwh",
    "charging_rate_kw",
    "charging_cost_usd",
    "state_of_charge_start_pct",
    "state_of_charge_end_pct",
    "distance_driven_since_last_charge_km",
    "temperature_degc",
    "vehicle_age_years",
    "start_hour",
]
DERIVED_NUMERIC_FEATURES: List[str] = [
    "soc_delta_pct",
    "is_weekend",
    "session_month",
    "minutes_since_prev_session",
    "prev_session_duration",
    "station_session_load_7d",
    "station_avg_duration_7d",
    "station_utilization_ratio_7d",
    "energy_to_capacity_ratio",
    "temperature_station_anomaly",
    "weather_stress_index",
    "humidity_proxy",
]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES

BASE_CATEGORICAL_FEATURES: List[str] = [
    "vehicle_model",
    "charging_station_location",
    "charging_station_id",
    "time_of_day",
    "day_of_week",
    "charger_type",
    "user_type",
]
DERIVED_CATEGORICAL_FEATURES: List[str] = [
    "temperature_band",
    "start_hour_bin",
    "station_cluster_label",
]
CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + DERIVED_CATEGORICAL_FEATURES

WEEKEND_DAYS = {"Saturday", "Sunday"}
TEMP_BINS = [-float("inf"), 0, 10, 20, 30, float("inf")]
TEMP_LABELS = ["freezing", "cold", "mild", "warm", "hot"]
HOUR_BINS = [-1, 6, 12, 18, 24]
HOUR_LABELS = ["overnight", "morning", "afternoon", "evening"]

MODEL_CONFIGS = {
    "random_forest": {
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "param_distributions": {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "n_iter": 12,
    },
    "hist_gradient_boosting": {
        "estimator": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "param_distributions": {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [None, 6, 10],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [10, 20, 30],
            "l2_regularization": [0.0, 0.1, 0.5],
        },
        "n_iter": 12,
    },
}


def _add_station_context_features(engineered: pd.DataFrame) -> pd.DataFrame:
    """Attach rolling station-level signals that capture recent demand and utilization."""
    station_frames: List[pd.DataFrame] = []
    # Treat each station independently so we can compute leakage-free rolling metrics.
    for _, station_df in engineered.groupby("charging_station_id", sort=False):
        station_df = station_df.sort_values("charging_start_time").copy()  # Critical for time-based rolling windows.
        station_df["minutes_since_prev_session"] = (
            station_df["charging_start_time"].diff().dt.total_seconds() / 60.0
        )  # Minutes since the previous plug-in at this station.
        station_df["prev_session_duration"] = station_df[TARGET_COLUMN].shift(1)  # Quick lag feature capturing short-term persistence.

        indexed = station_df.set_index("charging_start_time")
        ones = pd.Series(1.0, index=indexed.index)
        session_counts = ones.rolling("7D", closed="left").sum()  # Counts past-week sessions without leaking current row.
        duration_sum = indexed[TARGET_COLUMN].rolling("7D", closed="left").sum()  # Total charging minutes served in that same window.
        station_df["station_session_load_7d"] = session_counts.to_numpy()
        station_df["_station_duration_minutes_7d"] = duration_sum.to_numpy()

        duration_array = station_df["_station_duration_minutes_7d"].to_numpy(dtype=float)
        count_array = station_df["station_session_load_7d"].to_numpy(dtype=float)
        avg_duration = np.divide(
            duration_array,
            count_array,
            out=np.zeros_like(duration_array),
            where=count_array != 0,
        )  # Safe division that defaults to 0 when no historical sessions exist.
        station_df["station_avg_duration_7d"] = avg_duration
        station_df["station_utilization_ratio_7d"] = duration_array / (7 * 24 * 60)  # % of total minutes in a week spent charging.

        temp_mean = indexed["temperature_degc"].rolling("30D", closed="left").mean()
        station_df["temperature_station_anomaly"] = station_df["temperature_degc"] - temp_mean.to_numpy()  # Highlights locations running warmer/colder than their own baseline.

        station_df["station_session_load_7d"] = station_df["station_session_load_7d"].fillna(0.0)
        station_df["station_utilization_ratio_7d"] = station_df["station_utilization_ratio_7d"].fillna(0.0)
        station_df["station_avg_duration_7d"] = station_df["station_avg_duration_7d"].replace(0, np.nan)
        rolling_mean_fallback = station_df[TARGET_COLUMN].expanding().mean().shift(1)
        station_df["station_avg_duration_7d"] = station_df["station_avg_duration_7d"].fillna(rolling_mean_fallback)  # Use historical mean when no 7-day history exists.
        station_df["temperature_station_anomaly"] = station_df["temperature_station_anomaly"].fillna(0.0)

        station_frames.append(
            station_df.drop(columns=["_station_duration_minutes_7d"])  # helper column no longer needed
        )

    combined = pd.concat(station_frames).sort_values("_row_id")  # Reassemble stations back into original order.
    return combined


def _attach_station_cluster_labels(engineered: pd.DataFrame) -> pd.DataFrame:
    """Merge precomputed station cluster assignments if available."""
    if not STATION_CLUSTER_PATH.exists():
        logger.warning(
            "Station cluster file missing at %s; defaulting labels to 'unknown'",
            STATION_CLUSTER_PATH,
        )
        engineered["station_cluster_label"] = "unknown"  # Keep downstream categorical pipeline happy even without clusters.
        return engineered

    clusters = pd.read_csv(STATION_CLUSTER_PATH, usecols=["charging_station_id", "cluster"])
    clusters = clusters.rename(columns={"cluster": "station_cluster_label"})
    merged = engineered.merge(clusters, on="charging_station_id", how="left")
    merged["station_cluster_label"] = merged["station_cluster_label"].fillna("unknown").astype(str)
    return merged


def load_duration_dataset() -> pd.DataFrame:
    """Read the cleaned session-level dataset for regression."""
    df = pd.read_csv(CLEANED_DATA_PATH)
    logger.info("Loaded cleaned sessions with shape %s", df.shape)
    return df


def engineer_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add helpful derived features for the regression task."""
    engineered = df.copy()  # Avoid mutating the caller's DataFrame.
    engineered["_row_id"] = np.arange(len(engineered))
    engineered["charging_start_time"] = pd.to_datetime(engineered["charging_start_time"])  # Normalize timestamp precision before deriving temporal features.
    # Capture short descriptive stats covering behavioral, temporal, and environmental signals.
    engineered["soc_delta_pct"] = engineered["state_of_charge_end_pct"] - engineered["state_of_charge_start_pct"]
    engineered["is_weekend"] = engineered["day_of_week"].isin(WEEKEND_DAYS).astype(int)
    engineered["session_month"] = engineered["charging_start_time"].dt.month  # Enables capturing seasonality-lite for the two-month dataset.
    engineered["temperature_band"] = pd.cut(
        engineered["temperature_degc"],
        bins=TEMP_BINS,
        labels=TEMP_LABELS,
        include_lowest=True,
    ).astype(str)
    engineered["start_hour_bin"] = pd.cut(
        engineered["start_hour"],
        bins=HOUR_BINS,
        labels=HOUR_LABELS,
        include_lowest=True,
    ).astype(str)
    engineered = engineered.fillna({"temperature_band": "unknown", "start_hour_bin": "unknown"})  # Keep categorical pipeline from seeing NaN tokens.

    engineered = _add_station_context_features(engineered)
    engineered = _attach_station_cluster_labels(engineered)

    # Calculate ratios in numpy space for speed while protecting against divide-by-zero from incomplete telematics.
    consumption = engineered["energy_consumed_kwh"].to_numpy(dtype=float)
    capacity = engineered["battery_capacity_kwh"].replace(0, np.nan).to_numpy(dtype=float)
    ratio = np.divide(
        consumption,
        capacity,
        out=np.zeros_like(consumption),
        where=~np.isnan(capacity),  # Only divide when battery capacity is present.
    )
    engineered["energy_to_capacity_ratio"] = np.clip(ratio, 0, 5)
    engineered["weather_stress_index"] = (engineered["temperature_degc"] - 22).abs()  # Higher numbers mean harsher operating temperatures.
    engineered["humidity_proxy"] = np.clip(
        100 - (engineered["temperature_degc"] - 20).abs() * 3,
        0,
        100,
    )  # Rudimentary humidity surrogate for lack of true sensor data.

    engineered = engineered.sort_values("_row_id").drop(columns=["_row_id"])  # Restore original ordering to keep train/test split reproducible.
    return engineered


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target vector y."""
    missing = [col for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN] if col not in df]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()  # Keep matrix ordered to match ColumnTransformer expectations.
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Create the shared preprocessing pipeline for numeric/categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )  # Ensures consistent preprocessing regardless of the downstream estimator.
    return preprocessor


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute MAE, RMSE, and R^2 metrics."""
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }  # Package metrics as primitives for JSON/report friendliness.
    logger.info(
        "Duration metrics -> MAE: %.2f | RMSE: %.2f | R2: %.3f",
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
    )
    return metrics


def summarize_residuals_by_station(
    model_name: str,
    X_subset: pd.DataFrame,
    y_true: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Aggregate residual quality metrics per station and persist them."""
    residual_df = pd.DataFrame(
        {
            "charging_station_id": X_subset["charging_station_id"].values,
            "actual_duration": y_true.values,
            "predicted_duration": predictions,
        }
    )
    residual_df["residual"] = residual_df["actual_duration"] - residual_df["predicted_duration"]  # Positive == under-prediction.
    residual_df["abs_residual"] = residual_df["residual"].abs()
    residual_df["squared_residual"] = residual_df["residual"] ** 2

    summary = (
        residual_df.groupby("charging_station_id")
        .agg(
            sessions=("actual_duration", "size"),
            mae=("abs_residual", "mean"),
            bias=("residual", "mean"),
            rmse=("squared_residual", lambda s: math.sqrt(float(s.mean()))),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )  # Higher MAE rows highlight stations needing calibration or additional features.

    summary.to_csv(RESIDUALS_OUTPUT_PATH, index=False)
    top_skewed = summary.head(5)
    logger.info(
        "Saved %s residual summary (%d stations) to %s",
        model_name,
        len(summary),
        RESIDUALS_OUTPUT_PATH,
    )
    logger.info("Stations with highest MAE: %s", top_skewed.to_dict(orient="records"))
    return summary


def persist_duration_model(pipeline: Pipeline, best_result: Dict[str, object]) -> Path:
    """Save the best-performing pipeline to disk with metadata."""
    payload = {
        "saved_at_utc": _utc_timestamp(),
        "best_model": best_result,
        "pipeline": pipeline,  # Serialized scikit-learn Pipeline with preprocessing + estimator.
    }  # The payload mirrors the forecasting artifact schema for consistency.
    dump(payload, DURATION_MODEL_PATH)
    logger.info("Saved duration pipeline to %s", DURATION_MODEL_PATH)
    return DURATION_MODEL_PATH


def write_duration_report(
    results: List[Dict[str, object]],
    best_result: Dict[str, object],
    residual_summary: Optional[pd.DataFrame],
) -> Path:
    """Create markdown summary of tuning outcomes for auditability."""
    lines: List[str] = ["# Phase 5 Model 2 - Duration Prediction", ""]
    lines.append(f"- Run timestamp (UTC): {_utc_timestamp()}")
    lines.append(f"- Models compared: {len(results)}")
    if best_result:
        metrics = best_result["metrics"]
        lines.append(
            "- Best model: {name} (MAE {mae:.2f}, RMSE {rmse:.2f}, R2 {r2:.3f})".format(
                name=best_result["model"],
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r2=metrics["r2"],
            )
        )
    if residual_summary is not None:
        lines.append(f"- Residual summary: {RESIDUALS_OUTPUT_PATH}")
    lines.append("")

    if results:
        lines.extend([
            "| Model | MAE | RMSE | R2 |",
            "| --- | ---: | ---: | ---: |",
        ])
        for result in results:
            metrics = result["metrics"]
            lines.append(
                f"| {result['model']} | {metrics['mae']:.2f} | {metrics['rmse']:.2f} | {metrics['r2']:.3f} |"
            )

    if best_result:
        lines.append("")
        lines.append("## Best Model Hyperparameters")
        for param, value in sorted(best_result["best_params"].items()):  # Preserve deterministic order for readability.
            lines.append(f"- {param}: {value}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DURATION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote duration model report to %s", DURATION_REPORT_PATH)
    return DURATION_REPORT_PATH


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, object], Pipeline, pd.DataFrame, pd.Series, np.ndarray]:
    """Run RandomizedSearchCV for a given estimator and report metrics."""
    config = MODEL_CONFIGS[model_name]
    estimator = config["estimator"]
    param_distributions = {f"model__{k}": v for k, v in config["param_distributions"].items()}  # Prefix keys so RandomizedSearch can reach into the pipeline.
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=config.get("n_iter", 10),
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    # RandomizedSearchCV is preferred over GridSearch here to stay fast while scanning the most relevant regions.
    search.fit(X_train, y_train)
    logger.info("%s best params: %s", model_name, search.best_params_)
    best_pipeline = search.best_estimator_
    preds = best_pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test, preds)
    result = {
        "model": model_name,
        "metrics": metrics,
        "best_params": search.best_params_,
    }
    return result, best_pipeline, X_test.copy(), y_test.copy(), preds


def run_duration_experiments(test_size: float = 0.2) -> Dict[str, object]:
    """Engineer features, tune multiple models, and report their metrics."""
    df = engineer_duration_features(load_duration_dataset())
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )  # Maintain repeatable baseline results via shared random state.
    results: List[Dict[str, object]] = []
    best_model_snapshot = None
    for model_name in MODEL_CONFIGS:
        logger.info("Tuning %s ...", model_name)
        result, pipeline, X_eval, y_eval, preds = tune_model(model_name, X_train, y_train, X_test, y_test)
        results.append(result)
        if (
            best_model_snapshot is None
            or result["metrics"]["mae"] < best_model_snapshot["result"]["metrics"]["mae"]
        ):
            best_model_snapshot = {
                "result": result,
                "pipeline": pipeline,
                "X_eval": X_eval,
                "y_eval": y_eval,
                "preds": preds,  # Store evaluation artifacts for residual export.
            }

    if best_model_snapshot:
        residual_summary = summarize_residuals_by_station(
            best_model_snapshot["result"]["model"],
            best_model_snapshot["X_eval"],
            best_model_snapshot["y_eval"],
            best_model_snapshot["preds"],
        )
        persist_duration_model(best_model_snapshot["pipeline"], best_model_snapshot["result"])
        write_duration_report(results, best_model_snapshot["result"], residual_summary)
    else:
        residual_summary = None

    return {
        "results": results,
        "best_result": best_model_snapshot["result"] if best_model_snapshot else None,
        "residual_summary": residual_summary,
    }


if __name__ == "__main__":
    summary = run_duration_experiments()
    for item in summary["results"]:
        logger.info("%s metrics: %s", item["model"], item["metrics"])
