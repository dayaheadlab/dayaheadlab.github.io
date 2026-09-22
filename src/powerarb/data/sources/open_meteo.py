"""Open-Meteo weather. No key required.

Two sources, and the difference between them is the whole point:

``OpenMeteoSource`` reads the ERA5 **reanalysis** archive, i.e. the weather that actually
happened. Useful for explaining the past. Using it as a model feature would be a leak, since
nobody knows tomorrow's realised weather at gate closure.

``OpenMeteoForecastSource`` reads **forecasts**: the live forecast API for the delivery day,
and the historical-forecast archive (with ``_previous_day1``, the value as forecast one day
ahead) for training. Both therefore carry roughly the same forecast error a trader would have
faced, which is what makes them legal features. Series: ``wxfc.<var>.<point>``.
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


ARCHIVED_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_VARS = ["wind_speed_100m", "shortwave_radiation", "temperature_2m"]


class OpenMeteoForecastSource(DataSource):
    """Weather as it was *forecast*, which is the only version legal at gate closure.

    History comes from the archived-forecast endpoint using the ``_previous_day1`` variables:
    the value for hour t as predicted one day earlier. Live comes from the ordinary forecast
    endpoint. The training lead time (24-48 h) is therefore slightly longer than the live one
    (roughly 16-40 h when the job runs at 08:00 Berlin on D-1), so the model is trained on
    marginally worse weather information than it gets in production. That is the safe
    direction: it understates rather than overstates what the model can do.

    One request per point covers several years, so nothing has to be cached in the repository.
    """

    name = "open_meteo_forecast"

    def __init__(self, points: dict[str, dict[str, list[float]]], timeout: int = 180,
                 retries: int = 4):
        self.points = points
        self.timeout = timeout
        self.retries = retries

    def _get(self, url: str, params: dict) -> dict:
        """GET with exponential backoff. The archived-forecast endpoint drops connections now
        and then (2 of the first 8 live days, 2026-09-18/19); without retries each drop cost
        the day its weather features."""
        import time as _time

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code} from {url}")
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                last = e
                if attempt < self.retries:
                    _time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError(f"open-meteo failed after {self.retries + 1} attempts: {last}")

    def available_series(self, zone: str) -> list[str]:
        return [f"wxfc.{v}.{p}" for p in self.points.get(zone, {}) for v in FORECAST_VARS]

    def _parse(self, payload: dict, zone: str, point: str, suffix: str) -> pd.DataFrame:
        h = payload["hourly"]
        ts = pd.to_datetime(h["time"], utc=True)
        frames = []
        for var in FORECAST_VARS:
            key = var + suffix
            if key not in h:
                continue
            frames.append(to_long(ts, h[key], zone, f"wxfc.{var}.{point}", self.name))
        out = pd.concat(frames, ignore_index=True) if frames else empty_long()
        return out.dropna(subset=["value"])

    def fetch(self, zone: str, series: str, start: datetime, end: datetime) -> pd.DataFrame:
        point = series.split(".")[2]
        return self.fetch_history(zone, start, end, points=[point])

    def fetch_history(self, zone: str, start: datetime, end: datetime,
                      points: list[str] | None = None) -> pd.DataFrame:
        """Archived day-ahead-lead forecasts for the zone's reference points."""
        frames = []
        for point in (points or list(self.points.get(zone, {}))):
            lat, lon = self.points[zone][point]
            params = {
                "latitude": lat, "longitude": lon, "timezone": "UTC",
                "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(end).strftime("%Y-%m-%d"),
                "hourly": ",".join(v + "_previous_day1" for v in FORECAST_VARS),
            }
            frames.append(self._parse(self._get(ARCHIVED_FORECAST_URL, params), zone, point,
                                      "_previous_day1"))
        return pd.concat(frames, ignore_index=True) if frames else empty_long()

    def fetch_live(self, zone: str, forecast_days: int = 3) -> pd.DataFrame:
        """The forecast available right now, for the delivery day the job is about to bid."""
        frames = []
        for point, (lat, lon) in self.points.get(zone, {}).items():
            params = {"latitude": lat, "longitude": lon, "timezone": "UTC",
                      "forecast_days": forecast_days, "past_days": 2,
                      "hourly": ",".join(FORECAST_VARS)}
            frames.append(self._parse(self._get(LIVE_FORECAST_URL, params), zone, point, ""))
        return pd.concat(frames, ignore_index=True) if frames else empty_long()
