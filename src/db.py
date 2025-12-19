"""Utility helpers to create and populate the SQLite analytics database."""

from contextlib import closing
import sqlite3
from typing import Final

import pandas as pd

from src.config import CLEANED_DATA_PATH, SQLITE_DB_PATH
from src.logger import get_logger


DEFAULT_TABLE: Final[str] = "ev_sessions"
logger = get_logger(__name__)


def get_connection() -> sqlite3.Connection:
    """Open a SQLite connection pointing to the analytics database."""
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(SQLITE_DB_PATH)


def load_cleaned_data(table_name: str = DEFAULT_TABLE) -> None:
    """Load the cleaned CSV into SQLite, replacing any existing table."""
    df = pd.read_csv(CLEANED_DATA_PATH)
    logger.info("Loaded cleaned data from %s with shape %s", CLEANED_DATA_PATH, df.shape)

    with closing(get_connection()) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    logger.info("Loaded %s rows into '%s' at %s", len(df), table_name, SQLITE_DB_PATH)


def test_connection(table_name: str = DEFAULT_TABLE) -> None:
    """Run a simple COUNT(*) to verify data was inserted."""
    with closing(get_connection()) as conn:
        count = pd.read_sql(f"SELECT COUNT(*) AS row_count FROM {table_name};", conn)
        logger.info("Row count check:\n%s", count)


if __name__ == "__main__":
    load_cleaned_data()
    test_connection()
