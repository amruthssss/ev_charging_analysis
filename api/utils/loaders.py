"""
Model loaders - Load ML models and data at API startup.
"""
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any


# Global variables to store loaded models
forecast_model = None
duration_model = None
station_clusters_df = None


def get_models_path() -> Path:
    """Get the path to the models directory."""
    return Path(__file__).parent.parent.parent / "data" / "models"


def load_forecast_model():
    """Load the Prophet forecast model."""
    global forecast_model
    models_path = get_models_path()
    model_path = models_path / "forecast_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Forecast model not found at {model_path}")
    
    forecast_model = joblib.load(model_path)
    print(f"✅ Loaded forecast model from {model_path}")
    return forecast_model


def load_duration_model():
    """Load the duration prediction model."""
    global duration_model
    models_path = get_models_path()
    model_path = models_path / "duration_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Duration model not found at {model_path}")
    
    duration_model = joblib.load(model_path)
    print(f"✅ Loaded duration model from {model_path}")
    return duration_model


def load_station_clusters():
    """Load the station clusters CSV."""
    global station_clusters_df
    models_path = get_models_path()
    clusters_path = models_path / "station_clusters.csv"
    
    if not clusters_path.exists():
        raise FileNotFoundError(f"Station clusters not found at {clusters_path}")
    
    station_clusters_df = pd.read_csv(clusters_path)
    print(f"✅ Loaded station clusters from {clusters_path}")
    print(f"   Found {len(station_clusters_df)} stations across {station_clusters_df['cluster'].nunique()} clusters")
    return station_clusters_df


def load_all_models():
    """Load all models and data at startup."""
    print("🚀 Loading ML models and data...")
    try:
        load_forecast_model()
        load_duration_model()
        load_station_clusters()
        print("✅ All models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


def get_forecast_model():
    """Get the loaded forecast model."""
    if forecast_model is None:
        load_forecast_model()
    return forecast_model


def get_duration_model():
    """Get the loaded duration model."""
    if duration_model is None:
        load_duration_model()
    return duration_model


def get_station_clusters():
    """Get the loaded station clusters dataframe."""
    if station_clusters_df is None:
        load_station_clusters()
    return station_clusters_df
