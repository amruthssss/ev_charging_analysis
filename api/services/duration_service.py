"""
Duration Service - Predict charging session duration.
"""
import pandas as pd
import numpy as np
from fastapi import HTTPException
from api.utils.loaders import get_duration_model, get_station_clusters
from api.schemas import DurationRequest, DurationResponse


def predict_duration(request: DurationRequest) -> DurationResponse:
    """
    Predict session duration based on input features.
    
    Args:
        request: DurationRequest with energy_kwh, start_hour, weekday, station_id
        
    Returns:
        DurationResponse with predicted duration in minutes
    """
    try:
        # Load the duration model
        model = get_duration_model()
        
        # Load station clusters to get cluster assignment
        clusters_df = get_station_clusters()
        
        # Get cluster for the station (default to most common cluster if not found)
        station_row = clusters_df[clusters_df['station_id'] == request.station_id]
        
        if len(station_row) > 0:
            cluster = station_row.iloc[0]['cluster']
        else:
            # Default to most common cluster
            cluster = clusters_df['cluster'].mode()[0]
            print(f"⚠️  Station {request.station_id} not found in clusters, using default cluster {cluster}")
        
        # Create feature dataframe matching training format
        # Common features: energy_kwh, start_hour, weekday, cluster
        # You may need to adjust based on your actual model features
        features = pd.DataFrame({
            'energy_kwh': [request.energy_kwh],
            'start_hour': [request.start_hour],
            'weekday': [request.weekday],
            'cluster': [cluster]
        })
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Ensure prediction is reasonable (positive, typically between 5-240 minutes)
        predicted_minutes = max(5.0, min(240.0, float(prediction)))
        
        return DurationResponse(
            predicted_duration_minutes=round(predicted_minutes, 1)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Duration prediction failed: {str(e)}"
        )
