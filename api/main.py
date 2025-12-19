"""
EV Analytics API - FastAPI service for ML model predictions and analytics.

Run with: uvicorn api.main:app --reload
Access docs: http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path

from api.schemas import (
    ForecastRequest, ForecastResponse,
    DurationRequest, DurationResponse,
    ClusterRequest, ClusterResponse,
    AnalyticsResponse
)
from api.services.forecast_service import generate_forecast
from api.services.duration_service import predict_duration
from api.services.cluster_service import get_station_cluster, get_all_clusters
from api.utils.loaders import load_all_models


# ============================================
# FASTAPI APP INITIALIZATION
# ============================================

app = FastAPI(
    title="EV Analytics API",
    description="ML-powered API for EV charging demand forecasting, duration prediction, and station analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# STARTUP EVENT - Load Models
# ============================================

@app.on_event("startup")
async def startup_event():
    """Load all ML models when API starts."""
    try:
        load_all_models()
        print("🎉 API is ready!")
    except Exception as e:
        print(f"❌ Failed to load models on startup: {e}")
        # Optionally raise to prevent API from starting with missing models
        # raise


# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API health check."""
    return {
        "message": "EV Analytics API is running!",
        "version": "1.0.0",
        "endpoints": {
            "forecast": "/forecast - Demand forecasting for stations",
            "predict_duration": "/predict_duration - Session duration prediction",
            "cluster_station": "/cluster_station - Station cluster assignment",
            "analytics": "/analytics - Overall KPI analytics"
        },
        "docs": "/docs",
        "status": "healthy"
    }


# ============================================
# 1. FORECAST ENDPOINT
# ============================================

@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def forecast(request: ForecastRequest):
    """
    Predict future charging demand for a station.
    
    **Example Request:**
    ```json
    {
        "station_id": "ST_101",
        "days": 30
    }
    ```
    
    **Returns:** Daily forecasted sessions for the next N days.
    """
    return generate_forecast(request)


# ============================================
# 2. PREDICT DURATION ENDPOINT
# ============================================

@app.post("/predict_duration", response_model=DurationResponse, tags=["Prediction"])
async def duration(request: DurationRequest):
    """
    Predict charging session duration based on session characteristics.
    
    **Example Request:**
    ```json
    {
        "energy_kwh": 18.5,
        "start_hour": 17,
        "weekday": 1,
        "station_id": "ST_101"
    }
    ```
    
    **Returns:** Predicted duration in minutes.
    """
    return predict_duration(request)


# ============================================
# 3. CLUSTER STATION ENDPOINT
# ============================================

@app.post("/cluster_station", response_model=ClusterResponse, tags=["Clustering"])
async def cluster(request: ClusterRequest):
    """
    Get cluster assignment and description for a charging station.
    
    **Example Request:**
    ```json
    {
        "station_id": "ST_101"
    }
    ```
    
    **Returns:** Cluster number and descriptive label.
    """
    return get_station_cluster(request)


# ============================================
# 4. ANALYTICS ENDPOINT
# ============================================

@app.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
async def analytics():
    """
    Get overall KPIs and analytics across all charging sessions.
    
    **Returns:** Key performance metrics including total sessions, energy, peak hours, etc.
    """
    try:
        # Load cleaned data for analytics
        data_path = Path(__file__).parent.parent / "data" / "processed" / "cleaned_ev_sessions.csv"
        
        if not data_path.exists():
            raise HTTPException(status_code=500, detail="Analytics data not found")
        
        df = pd.read_csv(data_path)
        
        # Compute analytics
        total_sessions = len(df)
        total_energy = float(df['energy_consumed_kwh'].sum())
        avg_duration = float(df['charging_duration_minutes'].mean())
        
        # Peak hour (mode of start_hour)
        peak_hour = int(df['start_hour'].mode()[0])
        
        # Total unique stations
        total_stations = df['charging_station_id'].nunique()
        
        # Date range
        df['charging_start_time'] = pd.to_datetime(df['charging_start_time'])
        date_range = {
            "start": df['charging_start_time'].min().strftime('%Y-%m-%d'),
            "end": df['charging_start_time'].max().strftime('%Y-%m-%d')
        }
        
        return AnalyticsResponse(
            total_sessions=total_sessions,
            total_energy_kwh=round(total_energy, 2),
            avg_session_duration_minutes=round(avg_duration, 2),
            peak_hour=peak_hour,
            total_stations=total_stations,
            date_range=date_range
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics computation failed: {str(e)}")


# ============================================
# BONUS: CLUSTER SUMMARY ENDPOINT
# ============================================

@app.get("/clusters/summary", tags=["Clustering"])
async def clusters_summary():
    """
    Get summary of all station clusters.
    
    **Returns:** Overview of all clusters with station counts.
    """
    return get_all_clusters()


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "ev-analytics-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
