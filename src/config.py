"""
Project-wide configuration for the EV Charging platform.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_ev_sessions.csv"
POWERBI_DIR = PROCESSED_DATA_DIR / "powerbi"
DB_DIR = DATA_DIR / "db"
SQLITE_DB_PATH = DB_DIR / "ev_analytics.db"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATA_FILENAME = "ev_charging_data.csv"

DATABASE_NAME = "ev_charging.db"
DATABASE_PATH = DATA_DIR / DATABASE_NAME
RANDOM_STATE = 42
API_HOST = "0.0.0.0"
API_PORT = 8000
FEATURE_COLUMNS = []  # fill later when you know the inputs

for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, POWERBI_DIR, MODELS_DIR, DB_DIR, LOGS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)