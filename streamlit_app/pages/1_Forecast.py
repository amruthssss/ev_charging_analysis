from __future__ import annotations

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import smtplib
from email.message import EmailMessage

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Demand Forecasting")
st.caption("Prophet-powered forecasts built from your real charging history")
st.markdown("---")

# Futuristic styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', 'Inter', sans-serif;
            background: radial-gradient(circle at 10% 20%, rgba(99,102,241,0.08), transparent 25%),
                        radial-gradient(circle at 80% 0%, rgba(236,72,153,0.10), transparent 28%),
                        #0b1222;
            color: #e5e7eb;
        }
        .stMetric label, .stMetric div { color: #e5e7eb !important; }
        h1, h2, h3, h4 { color: #e0e7ff; }
        .glass-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 15px 45px rgba(0,0,0,0.35);
            backdrop-filter: blur(12px);
        }
        .primary-accent { color: #a5b4fc; }
        .subtle { color: #9ca3af; }
        .pill {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: linear-gradient(120deg, #6366f1, #8b5cf6, #ec4899);
            color: #0b1222;
            font-weight: 700;
            letter-spacing: 0.3px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_sessions():
    """Load cleaned sessions and ensure session_date exists."""
    data_file = project_root / "data" / "processed" / "cleaned_ev_sessions.csv"
    if not data_file.exists():
        return None, "Data file not found at data/processed/cleaned_ev_sessions.csv"
    try:
        df = pd.read_csv(data_file)
        if "charging_start_time" in df.columns:
            df["charging_start_time"] = pd.to_datetime(df["charging_start_time"])
            df["session_date"] = df["charging_start_time"].dt.date
        return df, None
    except Exception as exc:  # pragma: no cover - defensive
        return None, str(exc)


@st.cache_data
def load_daily_agg():
    """Prefer the pre-aggregated daily file; fall back to on-the-fly aggregation."""
    agg_path = project_root / "data" / "processed" / "powerbi" / "daily_agg.csv"
    if agg_path.exists():
        df = pd.read_csv(agg_path, parse_dates=["session_date"])
        return df, None
    # Fallback: aggregate from sessions
    sessions, err = load_sessions()
    if sessions is None:
        return None, err or "Unable to load sessions"
    if "session_date" not in sessions.columns:
        return None, "session_date missing in sessions"
    agg = (
        sessions.groupby(["charging_station_id", "session_date"])
        .size()
        .reset_index(name="total_sessions")
    )
    return agg, None


def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sessions per day for forecasting."""
    if "session_date" not in df.columns:
        return pd.DataFrame(columns=["ds", "y"])
    daily = df.groupby("session_date").size().reset_index(name="y")
    daily = daily.rename(columns={"session_date": "ds"})
    return daily.sort_values("ds")


def build_series_from_daily_agg(df: pd.DataFrame, station_id: str | None) -> pd.DataFrame:
    """Create ds/y frame from powerbi daily aggregates (per station if provided)."""
    working = df.copy()
    if station_id:
        working = working[working["charging_station_id"] == station_id]
    if working.empty:
        return pd.DataFrame(columns=["ds", "y"])
    if "session_date" in working.columns:
        working["session_date"] = pd.to_datetime(working["session_date"])
    daily = (
        working.groupby("session_date")["total_sessions" if "total_sessions" in working.columns else "y"]
        .sum()
        .reset_index(name="y")
    )
    daily = daily.rename(columns={"session_date": "ds"})
    return daily.sort_values("ds")


def run_prophet_forecast(series: pd.DataFrame, horizon: int, ci: int) -> tuple[pd.DataFrame, dict]:
    """Train Prophet on the provided series and return forecast + quick metrics."""
    interval_width = ci / 100
    model = Prophet(
        growth="linear",
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=interval_width,
        changepoint_prior_scale=0.2,
    )
    model.fit(series)
    future = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future)

    # Hold-out the last 7 days for a lightweight fit check when possible
    metrics = {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    if len(series) >= 14:
        holdout = min(7, len(series) // 3)
        train, test = series.iloc[:-holdout], series.iloc[-holdout:]
        holdout_model = Prophet(
            growth="linear",
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=interval_width,
            changepoint_prior_scale=0.2,
        )
        holdout_model.fit(train)
        holdout_forecast = holdout_model.predict(test[["ds"]])
        actual = test["y"].to_numpy(dtype=float)
        preds = holdout_forecast["yhat"].to_numpy(dtype=float)
        errors = actual - preds
        metrics["mae"] = float(np.mean(np.abs(errors)))
        metrics["rmse"] = float(np.sqrt(np.mean(errors ** 2)))
        denom = np.where(actual == 0, np.nan, np.abs(actual))
        with np.errstate(divide="ignore", invalid="ignore"):
            metrics["r2"] = float(1 - np.nansum((errors) ** 2) / np.nansum((actual - np.nanmean(actual)) ** 2))
    return forecast, metrics


def naive_forecast(series: pd.Series, horizon: int, ci: float) -> pd.DataFrame:
    """Simple baseline forecast using rolling mean and std for intervals."""
    mean = series.rolling(14, min_periods=7).mean().iloc[-1]
    std = series.rolling(14, min_periods=7).std().fillna(series.std()).iloc[-1]
    dates = pd.date_range(start=datetime.now().date() + timedelta(days=1), periods=horizon, freq="D")
    forecast = pd.DataFrame({
        "ds": dates,
        "yhat": np.full(horizon, round(mean, 1)),
    })
    z = 1.96 if ci >= 95 else 1.64
    forecast["yhat_lower"] = (forecast["yhat"] - z * std).clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat"] + z * std
    return forecast


# Data
sessions_df, load_err = load_sessions()
daily_agg_df, agg_err = load_daily_agg()

if load_err:
    st.error(f"Data load issue: {load_err}")
elif agg_err:
    st.warning(f"Using on-the-fly aggregation: {agg_err}")

# Sidebar controls
st.sidebar.header("Forecast Configuration")

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=60,
    value=21,
    help="Number of days to forecast"
)

station_filter = st.sidebar.selectbox(
    "Station Scope",
    ["All Stations", "Top 50 by volume", "Custom station ID"],
    help="Select which stations to include in aggregation"
)

custom_station = None
if station_filter == "Custom station ID" and sessions_df is not None:
    station_choices = sorted(sessions_df["charging_station_id"].unique().tolist())
    custom_station = st.sidebar.selectbox("Choose station", station_choices)

confidence_level = st.sidebar.slider(
    "Confidence Interval (%)",
    min_value=80,
    max_value=99,
    value=95,
    help="Prediction interval for forecast bands"
)

model_label = "Prophet" if PROPHET_AVAILABLE else "Rolling Mean Baseline"
st.sidebar.markdown(f"**Model:** {model_label}")
st.sidebar.caption("Trains instantly on selected scope; caches results per run.")

st.markdown("---")

# Main content
tab1, tab2 = st.tabs(["Forecast Results", "Model Performance"])


def get_filtered_series():
    if sessions_df is None and daily_agg_df is None:
        return pd.DataFrame(columns=["ds", "y"])

    if daily_agg_df is not None:
        series = build_series_from_daily_agg(daily_agg_df, custom_station if station_filter == "Custom station ID" else None)
        if station_filter == "Top 50 by volume" and sessions_df is not None:
            top_ids = (
                sessions_df.groupby("charging_station_id")
                .size()
                .sort_values(ascending=False)
                .head(50)
                .index
            )
            series = build_series_from_daily_agg(daily_agg_df[daily_agg_df["charging_station_id"].isin(top_ids)], None)
    else:
        working_df = sessions_df.copy()
        if station_filter == "Top 50 by volume":
            top_ids = (
                working_df.groupby("charging_station_id")
                .size()
                .sort_values(ascending=False)
                .head(50)
                .index
            )
            working_df = working_df[working_df["charging_station_id"].isin(top_ids)]
        elif station_filter == "Custom station ID" and custom_station:
            working_df = working_df[working_df["charging_station_id"] == custom_station]
        series = build_daily_series(working_df)
    return series


def build_forecast_figure(series: pd.DataFrame, forecast_df: pd.DataFrame, confidence_level: int) -> go.Figure:
    """Create the forecast chart used in the UI and email export."""
    fig = go.Figure()

    if series is not None and not series.empty:
        fig.add_trace(go.Scatter(
            x=series["ds"],
            y=series["y"],
            mode="lines",
            name="History",
            line=dict(color="rgba(165,180,252,0.7)", width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(79,70,229,0.08)",
            hovertemplate="%{x}<br>Sessions: %{y}<extra></extra>"
        ))

    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Predicted"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ec4899", width=3, shape="spline"),
            marker=dict(color="#f472b6", size=7, line=dict(color="#0b1222", width=1.2)),
        ))

        if "Upper" in forecast_df.columns and "Lower" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Upper"],
                mode="lines",
                name=f"{confidence_level}% Upper",
                line=dict(color="rgba(236,72,153,0.2)", width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Lower"],
                mode="lines",
                name=f"{confidence_level}% Lower",
                line=dict(color="rgba(236,72,153,0.2)", width=0),
                fill="tonexty",
                fillcolor="rgba(236,72,153,0.12)",
                showlegend=True
            ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sessions",
        hovermode="x unified",
        plot_bgcolor="rgba(255,255,255,0.02)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#e5e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


def send_forecast_email(recipient: str, notes: str, payload: dict, series: pd.DataFrame, confidence_level: int) -> None:
    """Send the latest forecast payload with CSV + chart attachments via SMTP."""
    if not recipient:
        raise ValueError("Recipient email is required.")
    if payload is None or payload.get("forecast") is None or payload.get("forecast").empty:
        raise ValueError("No forecast payload available to send.")
    if series is None or series.empty:
        raise ValueError("Historical series is empty; generate a forecast first.")

    secrets = st.secrets
    required_keys = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "SMTP_FROM"]
    missing = [key for key in required_keys if key not in secrets]
    if missing:
        raise ValueError(f"Missing SMTP secret(s): {', '.join(missing)}")

    forecast_df = payload["forecast"]
    model_name = payload.get("model", "Forecast")

    fig = build_forecast_figure(series, forecast_df, confidence_level)
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn").encode("utf-8")
    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")

    subject = f"EV Forecast Report - {datetime.now():%Y-%m-%d}"
    summary_lines = [
        "Hi team,",
        "",  # spacer
        f"Attached: latest forecast ({model_name}) from the dashboard.",
        f"Horizon: {len(forecast_df)} days | Confidence: {confidence_level}%",
    ]
    if notes:
        summary_lines.extend(["", f"Notes: {notes}"])
    summary_lines.extend(["", "Thanks!", "EV Analytics Platform"])

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = secrets["SMTP_FROM"]
    msg["To"] = recipient
    msg.set_content("\n".join(summary_lines))

    msg.add_attachment(
        csv_bytes,
        maintype="text",
        subtype="csv",
        filename=f"forecast_{datetime.now():%Y%m%d}.csv",
    )

    msg.add_attachment(
        chart_html,
        maintype="text",
        subtype="html",
        filename="forecast_chart.html",
    )

    host = secrets["SMTP_HOST"]
    port = int(str(secrets.get("SMTP_PORT", 587)))
    user = secrets["SMTP_USER"]
    password = secrets["SMTP_PASS"]

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)

with tab1:
    st.subheader("Demand Forecast")
    col1, col2 = st.columns([3, 1])

    series = get_filtered_series()
    if series.empty:
        st.info("No data available for the selected scope. Add charging_start_time/session_date to data.")
    else:
        coverage = (series["ds"].min(), series["ds"].max())
        st.caption(f"History coverage: {coverage[0]} → {coverage[1]} | {len(series)} days")

        # Ensure session_state stores last forecast
        if "forecast_payload" not in st.session_state:
            st.session_state["forecast_payload"] = None

        def generate_forecast():
            if len(series) < 10 and PROPHET_AVAILABLE:
                st.warning("Need at least 10 days of history for Prophet; using rolling baseline instead.")
            if PROPHET_AVAILABLE and len(series) >= 10:
                forecast, metrics = run_prophet_forecast(series, forecast_horizon, confidence_level)
            else:
                forecast = naive_forecast(series["y"], forecast_horizon, confidence_level)
                metrics = {"mae": np.nan, "rmse": np.nan, "r2": np.nan}

            forecast_tail = forecast.tail(forecast_horizon).copy()
            forecast_tail = forecast_tail.rename(columns={
                "ds": "Date",
                "yhat": "Predicted",
                "yhat_lower": "Lower",
                "yhat_upper": "Upper",
            })
            st.session_state["forecast_payload"] = {
                "forecast": forecast_tail,
                "metrics": metrics,
                "model": "Prophet" if PROPHET_AVAILABLE and len(series) >= 10 else "Rolling baseline",
            }

        if st.button("🔄 Generate Forecast", type="primary", use_container_width=True):
            with st.spinner("Training and forecasting..."):
                generate_forecast()
            st.success("Forecast ready. Scroll for details or export.")

        payload = st.session_state.get("forecast_payload")
        if payload is None:
            st.info("Click **Generate Forecast** to produce results.")
        else:
            forecast_df = payload["forecast"]
            model_name = payload["model"]

            with col1:
                fig = build_forecast_figure(series, forecast_df, confidence_level)
                st.plotly_chart(fig, use_container_width=True, theme=None)

            with col2:
                card = st.container(border=True)
                pred_series = forecast_df["Predicted"].dropna()
                if forecast_df.empty or pred_series.empty:
                    card.metric("Peak Day", "--")
                    card.metric("Avg Daily", "--")
                    card.metric("Total Period", "--")
                    card.metric("Model", model_name)
                else:
                    peak_idx = pred_series.idxmax()
                    peak_day = forecast_df.loc[peak_idx, "Date"]
                    avg_demand = pred_series.mean()
                    total_forecast = pred_series.sum()
                    card.metric("Peak Day", peak_day.strftime("%b %d"))
                    card.metric("Avg Daily", f"{avg_demand:.1f} sessions")
                    card.metric("Total Period", f"{total_forecast:,.0f} sessions")
                    card.metric("Model", model_name)


with tab2:
    st.subheader("Model Performance")
    payload = st.session_state.get("forecast_payload")
    metrics = payload.get("metrics") if payload else {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAE (sessions)", f"{metrics['mae']:.2f}" if not np.isnan(metrics.get("mae", np.nan)) else "--")
    with col2:
        st.metric("RMSE (sessions)", f"{metrics['rmse']:.2f}" if not np.isnan(metrics.get("rmse", np.nan)) else "--")
    with col3:
        history_days = 0 if sessions_df is None else len(build_daily_series(sessions_df))
        st.metric("History Days", f"{history_days}")
    with col4:
        st.metric("Algorithm", model_label)
    st.info("Each run trains on-the-fly using your filtered history. Prophet is used when available and enough data exists; otherwise a rolling baseline is applied.")

# Action buttons
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💾 Export CSV", use_container_width=True):
        payload = st.session_state.get("forecast_payload")
        if not payload:
            st.warning("Generate a forecast first.")
        else:
            csv_bytes = payload["forecast"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Forecast Data",
                data=csv_bytes,
                file_name=f"forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

with col2:
    with st.popover("📧 Send Report", use_container_width=True):
        st.caption("Send the current forecast with CSV + chart attached.")
        recipient = st.text_input("Recipient email", placeholder="ops@company.com")
        notes = st.text_area("Optional message")
        payload = st.session_state.get("forecast_payload")

        if st.button("Send now", use_container_width=True):
            if not payload:
                st.warning("Generate a forecast first.")
            else:
                series_for_email = get_filtered_series()
                if series_for_email.empty:
                    st.warning("No history for this scope; generate a forecast first.")
                elif not recipient:
                    st.warning("Enter a recipient email.")
                else:
                    with st.spinner("Sending forecast via email..."):
                        try:
                            send_forecast_email(recipient, notes, payload, series_for_email, confidence_level)
                            st.success(f"Report sent to {recipient} with CSV + chart attached.")
                        except Exception as exc:  # pragma: no cover - defensive
                            st.error(f"Could not send email: {exc}")

with col3:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("forecast_payload", None)
        st.rerun()

with col4:
    st.caption("Built for stakeholder-ready forecasting. Save and share forecasts with your team.")

st.markdown("---")
st.caption("Demand Forecasting Module | EV Charging Analytics Platform")
