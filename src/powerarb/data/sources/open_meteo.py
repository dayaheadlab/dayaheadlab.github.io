"""Open-Meteo historical weather (ERA5 reanalysis archive). No key required.

Series: ``weather.<var>.<point>`` for each named point in settings.weather_points[zone].
Variables: temperature_2m (C), wind_speed_100m (m/s), shortwave_radiation (W/m2).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests

from ..store import empty_long
from .base import DataSource, to_long

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["temperature_2m", "wind_speed_100m", "shortwave_radiation"]


class OpenMeteoSource(DataSource):
    name = "open_meteo"

    def __init__(self, points: dict[str, dict[str, list[float]]], timeout: int = 60):
        self.points = points
        self.timeout = timeout

    def available_series(self, zone: str) -> list[str]:
        return [f"weather.{v}.{p}" for p in self.points.get(zone, {}) for v in VARS]

    def fetch(self, zone: str, series: str, start: datetime, end: datetime) -> pd.DataFrame:
        _, var, point = series.split(".")
        lat, lon = self.points[zone][point]
        params = {
            "latitude": lat, "longitude": lon, "hourly": var, "timezone": "UTC",
            "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "end_date": (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        }
        r = requests.get(ARCHIVE_URL, params=params, timeout=self.timeout)
        r.raise_for_status()
        h = r.json()["hourly"]
        ts = pd.to_datetime(h["time"], utc=True)
        return to_long(ts, h[var], zone, series, self.name).dropna(subset=["value"])

    def fetch_all(self, zone: str, start: datetime, end: datetime) -> pd.DataFrame:
        frames = [self.fetch(zone, s, start, end) for s in self.available_series(zone)]
        return pd.concat(frames, ignore_index=True) if frames else empty_long()
