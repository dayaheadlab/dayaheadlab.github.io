"""Local time-series store on DuckDB.

One long table holds everything: (zone, series, ts_utc) -> value. Series names are dotted,
e.g. ``price.day_ahead``, ``forecast.wind_onshore.day_ahead``, ``actual.load``.
Timestamps are stored as naive UTC and returned tz-aware UTC.

Swapping in PostgreSQL/TimescaleDB later only requires re-implementing this class; nothing
else in the project touches SQL.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

LONG_COLUMNS = ["ts_utc", "zone", "series", "value", "source"]

_DDL = """
CREATE TABLE IF NOT EXISTS timeseries (
    zone        VARCHAR   NOT NULL,
    series      VARCHAR   NOT NULL,
    ts_utc      TIMESTAMP NOT NULL,
    value       DOUBLE,
    source      VARCHAR,
    ingested_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (zone, series, ts_utc)
);
CREATE TABLE IF NOT EXISTS forecast_log (
    zone        VARCHAR   NOT NULL,
    target_day  DATE      NOT NULL,
    issued_at   TIMESTAMP NOT NULL,
    ts_utc      TIMESTAMP NOT NULL,
    pred        DOUBLE,
    model       VARCHAR,
    on_time     BOOLEAN,
    PRIMARY KEY (zone, target_day, issued_at, ts_utc)
);
CREATE TABLE IF NOT EXISTS score_log (
    zone         VARCHAR   NOT NULL,
    target_day   DATE      NOT NULL,
    issued_at    TIMESTAMP NOT NULL,
    on_time      BOOLEAN,
    n            INTEGER,
    mae          DOUBLE,
    rmse         DOUBLE,
    rank_corr    DOUBLE,
    rev_forecast DOUBLE,
    rev_perfect  DOUBLE,
    capture      DOUBLE,
    scored_at    TIMESTAMP,
    model        VARCHAR,
    PRIMARY KEY (zone, target_day, issued_at)
);
"""

# Columns added after the first release; applied to stores created by an older version.
_MIGRATIONS = [
    "ALTER TABLE score_log ADD COLUMN IF NOT EXISTS model VARCHAR",
    # Error on the within-day shape. Under the shape target the level is reconstructed from the
    # previous day's mean, so level MAE mostly measures how far the day's price level moved
    # (MAE tracked the day-over-day mean jump almost exactly), not model skill.
    "ALTER TABLE score_log ADD COLUMN IF NOT EXISTS shape_mae DOUBLE",
]


def _to_naive_utc(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, utc=True)
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)


def empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=LONG_COLUMNS)


class TimeSeriesStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.path)
        self.conn.execute(_DDL)
        for sql in _MIGRATIONS:
            try:
                self.conn.execute(sql)
            except duckdb.Error:  # already applied, or an older engine without IF NOT EXISTS
                pass

    # ---- write -------------------------------------------------------------------------
    def upsert(self, df: pd.DataFrame) -> int:
        """Insert or replace rows of a long-format frame with LONG_COLUMNS."""
        if df is None or df.empty:
            return 0
        missing = set(LONG_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"long frame missing columns: {sorted(missing)}")
        out = df[LONG_COLUMNS].copy()
        out["ts_utc"] = _to_naive_utc(out["ts_utc"])
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["value"]).drop_duplicates(subset=["zone", "series", "ts_utc"], keep="last")
        self.conn.register("_incoming", out)
        self.conn.execute(
            "INSERT OR REPLACE INTO timeseries (zone, series, ts_utc, value, source) "
            "SELECT zone, series, ts_utc, value, source FROM _incoming"
        )
        self.conn.unregister("_incoming")
        return len(out)

    # ---- read --------------------------------------------------------------------------
    def read_long(
        self,
        zone: str,
        series: list[str] | str | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> pd.DataFrame:
        sql = "SELECT ts_utc, zone, series, value, source FROM timeseries WHERE zone = ?"
        params: list = [zone]
        if series:
            if isinstance(series, str):
                series = [series]
            placeholders = ",".join(["?"] * len(series))
            sql += f" AND series IN ({placeholders})"
            params += list(series)
        if start is not None:
            sql += " AND ts_utc >= ?"
            params.append(_naive(start))
        if end is not None:
            sql += " AND ts_utc < ?"
            params.append(_naive(end))
        sql += " ORDER BY series, ts_utc"
        df = self.conn.execute(sql, params).df()
        df["ts_utc"] = pd.to_datetime(df["ts_utc"]).dt.tz_localize("UTC")
        return df

    def read_wide(
        self,
        zone: str,
        series: list[str] | str | None = None,
        start=None,
        end=None,
        freq: str | None = None,
    ) -> pd.DataFrame:
        """Wide frame indexed by tz-aware UTC timestamp, one column per series.

        ``freq`` (e.g. ``"1h"``) resamples by mean. This matters because DE-LU day-ahead
        prices switch from hourly to 15-minute resolution on 2025-10-01.
        """
        long = self.read_long(zone, series, start, end)
        if long.empty:
            return pd.DataFrame()
        wide = long.pivot_table(index="ts_utc", columns="series", values="value", aggfunc="last")
        wide.columns.name = None
        if freq:
            wide = wide.resample(freq).mean()
        return wide.sort_index()

    def coverage(self) -> pd.DataFrame:
        return self.conn.execute(
            "SELECT zone, series, min(ts_utc) AS start, max(ts_utc) AS end, count(*) AS n "
            "FROM timeseries GROUP BY zone, series ORDER BY zone, series"
        ).df()

    def last_timestamp(self, zone: str, series: str) -> pd.Timestamp | None:
        row = self.conn.execute(
            "SELECT max(ts_utc) FROM timeseries WHERE zone = ? AND series = ?", [zone, series]
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return pd.Timestamp(row[0]).tz_localize("UTC")

    # ---- live forecast record ------------------------------------------------------------
    def log_forecast(self, zone: str, target_day, issued_at, preds: pd.Series, model: str,
                     on_time: bool) -> int:
        df = pd.DataFrame({
            "zone": zone,
            "target_day": pd.Timestamp(target_day).date(),
            "issued_at": _naive(issued_at),
            "ts_utc": _to_naive_utc(pd.Series(preds.index)).values,
            "pred": preds.values.astype(float),
            "model": model,
            "on_time": bool(on_time),
        })
        self.conn.register("_fc", df)
        self.conn.execute(
            "INSERT OR REPLACE INTO forecast_log SELECT zone, target_day, issued_at, ts_utc, "
            "pred, model, on_time FROM _fc")
        self.conn.unregister("_fc")
        return len(df)

    def read_forecast(self, zone: str, target_day, issued_at=None) -> pd.DataFrame:
        """Forecast rows for one target day. Default: the latest on-time issue, else latest."""
        if issued_at is None:
            row = self.conn.execute(
                "SELECT issued_at FROM forecast_log WHERE zone = ? AND target_day = ? "
                "ORDER BY on_time DESC, issued_at DESC LIMIT 1",
                [zone, pd.Timestamp(target_day).date()]).fetchone()
            if row is None:
                return pd.DataFrame(columns=["ts_utc", "pred", "issued_at", "on_time", "model"])
            issued_at = row[0]
        df = self.conn.execute(
            "SELECT ts_utc, pred, issued_at, on_time, model FROM forecast_log "
            "WHERE zone = ? AND target_day = ? AND issued_at = ? ORDER BY ts_utc",
            [zone, pd.Timestamp(target_day).date(), _naive(issued_at)]).df()
        df["ts_utc"] = pd.to_datetime(df["ts_utc"]).dt.tz_localize("UTC")
        df["issued_at"] = pd.to_datetime(df["issued_at"]).dt.tz_localize("UTC")
        return df

    def forecast_days(self, zone: str) -> list:
        rows = self.conn.execute(
            "SELECT DISTINCT target_day FROM forecast_log WHERE zone = ? ORDER BY target_day",
            [zone]).fetchall()
        return [r[0] for r in rows]

    def log_score(self, row: dict) -> None:
        cols = ["zone", "target_day", "issued_at", "on_time", "n", "mae", "rmse", "rank_corr",
                "rev_forecast", "rev_perfect", "capture", "scored_at", "model", "shape_mae"]
        vals = [row.get(c) for c in cols]
        vals[cols.index("target_day")] = pd.Timestamp(vals[cols.index("target_day")]).date()
        vals[cols.index("issued_at")] = _naive(vals[cols.index("issued_at")])
        vals[cols.index("scored_at")] = _naive(vals[cols.index("scored_at")])
        self.conn.execute(
            f"INSERT OR REPLACE INTO score_log ({','.join(cols)}) VALUES ({','.join(['?'] * len(cols))})",
            vals)

    def read_scores(self, zone: str) -> pd.DataFrame:
        df = self.conn.execute(
            "SELECT * FROM score_log WHERE zone = ? ORDER BY target_day", [zone]).df()
        return df

    def close(self) -> None:
        self.conn.close()


def _naive(ts) -> datetime:
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.tz_localize(None).to_pydatetime()
