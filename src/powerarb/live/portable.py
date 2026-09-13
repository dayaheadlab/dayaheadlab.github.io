"""Portable state: move the pieces the daily job needs between DuckDB and plain CSV files.

The cloud runner has no persistent disk, so the repository holds the state as CSV and each run
rebuilds a throwaway DuckDB from it. Only what the daily job actually needs is carried:

    state/price.csv          the day-ahead price history (the only series the leak-free
                             feature set requires, ~2 MB and growing ~96 rows a day)
    state/forecast_log.csv   every forecast ever issued, with its issue time and on-time flag
    state/score_log.csv      the score of every forecast whose delivery day has cleared

The full research database (load, wind, solar, actuals) stays local; it is not needed to
produce or score a forecast.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..data.store import LONG_COLUMNS, TimeSeriesStore

log = logging.getLogger(__name__)

PRICE_SERIES = "price.day_ahead"
FILES = {"price": "price.csv", "forecast_log": "forecast_log.csv", "score_log": "score_log.csv"}


def export_state(store: TimeSeriesStore, state_dir: Path | str, zone: str) -> dict[str, int]:
    """Write the daily job's state to CSV. Timestamps are ISO-8601 UTC without offset."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    price = store.read_long(zone, PRICE_SERIES)
    price = price.assign(ts_utc=price["ts_utc"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S"))
    price[LONG_COLUMNS].to_csv(state_dir / FILES["price"], index=False)
    counts["price"] = len(price)

    for table in ("forecast_log", "score_log"):
        df = store.conn.execute(f"SELECT * FROM {table} WHERE zone = ? ORDER BY 1, 2, 3",
                                [zone]).df()
        df.to_csv(state_dir / FILES[table], index=False)
        counts[table] = len(df)

    log.info("exported state to %s: %s", state_dir, counts)
    return counts


def import_state(store: TimeSeriesStore, state_dir: Path | str) -> dict[str, int]:
    """Load CSV state into an (empty) store. Missing files are treated as empty."""
    state_dir = Path(state_dir)
    counts: dict[str, int] = {}

    price_path = state_dir / FILES["price"]
    if price_path.exists():
        df = pd.read_csv(price_path)
        counts["price"] = store.upsert(df) if not df.empty else 0
    else:
        counts["price"] = 0

    for table in ("forecast_log", "score_log"):
        path = state_dir / FILES[table]
        if not path.exists():
            counts[table] = 0
            continue
        df = pd.read_csv(path)
        if df.empty:
            counts[table] = 0
            continue
        for col in ("target_day",):
            if col in df:
                df[col] = pd.to_datetime(df[col]).dt.date
        for col in ("issued_at", "ts_utc", "scored_at"):
            if col in df:
                df[col] = pd.to_datetime(df[col])
        store.conn.register("_incoming", df)
        cols = ",".join(df.columns)
        store.conn.execute(f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM _incoming")
        store.conn.unregister("_incoming")
        counts[table] = len(df)

    log.info("imported state from %s: %s", state_dir, counts)
    return counts
