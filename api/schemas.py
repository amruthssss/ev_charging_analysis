"""
Pydantic schemas for FastAPI request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import date


# ============================================
# 1. FORECAST ENDPOINT SCHEMAS
# ============================================

class ForecastRequest(BaseModel):
    """Request schema for demand forecasting."""
    station_id: str = Field(..., example="ST_101")
    days: int = Field(..., ge=1, le=365, example=30, description="Number of days to forecast")


class ForecastDataPoint(BaseModel):
    """Single forecast data point."""
    date: str = Field(..., example="2025-02-01")
    sessions: float = Field(..., example=42.0)


class ForecastResponse(BaseModel):
    """Response schema for demand forecasting."""
    station_id: str = Field(..., example="ST_101")
    forecast: List[ForecastDataPoint]


# ============================================
# 2. DURATION PREDICTION ENDPOINT SCHEMAS
# ============================================

class DurationRequest(BaseModel):
    """Request schema for duration prediction."""
    energy_kwh: float = Field(..., gt=0, example=18.5)
    start_hour: int = Field(..., ge=0, le=23, example=17)
    weekday: int = Field(..., ge=0, le=6, example=1, description="Monday=0, Sunday=6")
    station_id: str = Field(..., example="ST_101")


class DurationResponse(BaseModel):
    """Response schema for duration prediction."""
    predicted_duration_minutes: float = Field(..., example=52.3)


# ============================================
# 3. CLUSTER ENDPOINT SCHEMAS
# ============================================

class ClusterRequest(BaseModel):
    """Request schema for station clustering."""
    station_id: str = Field(..., example="ST_101")


class ClusterResponse(BaseModel):
    """Response schema for station clustering."""
    station_id: str
    cluster: int
    description: str = Field(..., example="High demand – peak hour dominant station")


# ============================================
# 4. ANALYTICS ENDPOINT SCHEMA
# ============================================

class AnalyticsResponse(BaseModel):
    """Response schema for analytics/KPIs."""
    total_sessions: int
    total_energy_kwh: float
    avg_session_duration_minutes: float
    peak_hour: int
    total_stations: int
    date_range: Dict[str, str]  # {"start": "2024-01-01", "end": "2024-12-31"}
