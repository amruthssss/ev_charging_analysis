"""Business analytics queries for the EV charging dataset."""

from contextlib import closing

import pandas as pd

from src.db import get_connection

DEFAULT_TABLE = "ev_sessions"
HOURLY_DEMAND_QUERY = """
    SELECT start_hour, COUNT(*) AS total_sessions
    FROM {table}
    GROUP BY start_hour
    ORDER BY start_hour;
"""
STATION_DEMAND_QUERY = """
    SELECT charging_station_id, COUNT(*) AS sessions
    FROM {table}
    GROUP BY charging_station_id
    ORDER BY sessions DESC;
"""
AVG_DURATION_QUERY = """
    SELECT charging_station_id,
           AVG(charging_duration_minutes) AS avg_duration_minutes
    FROM {table}
    GROUP BY charging_station_id
    ORDER BY avg_duration_minutes DESC;
"""
MONTHLY_TRENDS_QUERY = """
    SELECT strftime('%Y', charging_start_time) AS year,
           strftime('%m', charging_start_time) AS month,
           COUNT(*) AS sessions
    FROM {table}
    GROUP BY year, month
    ORDER BY year, month;
"""


def _run_query(query_template: str, table: str) -> pd.DataFrame:
    sql = query_template.format(table=table)
    with closing(get_connection()) as conn:
        return pd.read_sql(sql, conn)


def hourly_demand(table: str = DEFAULT_TABLE) -> pd.DataFrame:
    """Return session counts grouped by hour extracted during preprocessing."""
    return _run_query(HOURLY_DEMAND_QUERY, table)


def station_demand(table: str = DEFAULT_TABLE) -> pd.DataFrame:
    """Return session counts per charging station."""
    return _run_query(STATION_DEMAND_QUERY, table)


def avg_charging_duration(table: str = DEFAULT_TABLE) -> pd.DataFrame:
    """Return average charging duration (minutes) per station."""
    return _run_query(AVG_DURATION_QUERY, table)


def monthly_trends(table: str = DEFAULT_TABLE) -> pd.DataFrame:
    """Return session counts grouped by year/month from charging_start_time."""
    return _run_query(MONTHLY_TRENDS_QUERY, table)


if __name__ == "__main__":
    print("Hourly demand:")
    print(hourly_demand().head())

    print("\nStation demand:")
    print(station_demand().head())

    print("\nAvg charging duration:")
    print(avg_charging_duration().head())

    print("\nMonthly trends:")
    print(monthly_trends().head())
