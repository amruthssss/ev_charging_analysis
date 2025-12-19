"""
Forecast Service - Demand forecasting using Prophet models.
"""
import pandas as pd
from datetime import datetime, timedelta
from fastapi import HTTPException
from api.utils.loaders import get_forecast_model
from api.schemas import ForecastRequest, ForecastResponse, ForecastDataPoint


def generate_forecast(request: ForecastRequest) -> ForecastResponse:
    """
    Generate demand forecast for a station.
    
    Args:
        request: ForecastRequest with station_id and days
        
    Returns:
        ForecastResponse with forecasted sessions
    """
    try:
        # Load the forecast model (dict of station_id -> Prophet model)
        models_dict = get_forecast_model()
        
        # Check if station exists in models
        if request.station_id not in models_dict:
            available_stations = list(models_dict.keys())[:5]  # Show first 5
            raise HTTPException(
                status_code=404,
                detail=f"Station '{request.station_id}' not found. Available stations (sample): {available_stations}"
            )
        
        # Get the Prophet model for this station
        model = models_dict[request.station_id]
        
        # Create future dataframe
        future_dates = pd.DataFrame({
            'ds': pd.date_range(
                start=datetime.now().date(),
                periods=request.days,
                freq='D'
            )
        })
        
        # Generate forecast
        forecast = model.predict(future_dates)
        
        # Extract predictions and format response
        forecast_data = []
        for _, row in forecast.iterrows():
            forecast_data.append(ForecastDataPoint(
                date=row['ds'].strftime('%Y-%m-%d'),
                sessions=max(0, round(row['yhat'], 1))  # Ensure non-negative
            ))
        
        return ForecastResponse(
            station_id=request.station_id,
            forecast=forecast_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")
