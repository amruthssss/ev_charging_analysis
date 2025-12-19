"""Utilities for loading EV charging datasets from data/raw."""

import pandas as pd

from src.config import RAW_DATA_DIR, RAW_DATA_FILENAME
from src.logger import get_logger

logger = get_logger(__name__)
def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load a CSV from data/raw and report basic info.

    Args:
        filename: Name of the CSV file inside data/raw/.

    Returns:
        Raw EV charging dataset as a pandas DataFrame.
    """
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        logger.error("Missing file at %s", file_path)
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)
    logger.info("Loaded %s with shape %s", file_path.name, df.shape)
    logger.debug("Columns: %s", list(df.columns))
    logger.debug("Data types:\n%s", df.dtypes)
    return df

if __name__ == "__main__":
    data = load_raw_data(RAW_DATA_FILENAME)
    logger.info("Preview:\n%s", data.head())