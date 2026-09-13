"""Energy-Charts (Fraunhofer ISE) public API. No key required. Data license CC BY 4.0.

Docs: https://api.energy-charts.info/  v2 endpoints return
``{"data": [{"timestamp": ..., "values": {series_id: value}}], "resolution": "PT15M", ...}``.

Series exposed here (timestamps converted to UTC):
    price.day_ahead                EUR/MWh; hourly until 2025-09-30, 15-minute from 2025-10-01
    forecast.<type>.day_ahead      MW; type in solar | wind_onshore | wind_offshore | load
    forecast.<type>.intraday       MW; same types, intraday forecast update
    actual.<type>                  MW; any series id of /v2/public_power (load, solar, ...)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from ..store import empty_long
from .base import DataSource, to_long

log = logging.getLogger(__name__)

BASE_URL = "https://api.energy-charts.info"

# Bidding zone -> country code used by the country-level endpoints.
ZONE_TO_COUNTRY = {
    "DE-LU": "de", "DE-AT-LU": "de", "AT": "at", "BE": "be", "CH": "ch", "CZ": "cz", "DK1": "dk",
    "DK2": "dk", "EE": "ee", "ES": "es", "FI": "fi", "FR": "fr", "GR": "gr", "HR": "hr", "HU": "hu",
    "LT": "lt", "LV": "lv", "NL": "nl", "NO1": "no", "NO2": "no", "NO3": "no", "NO4": "no",
    "NO5": "no", "PL": "pl", "PT": "pt", "RO": "ro", "RS": "rs", "SE1": "se", "SE2": "se",
    "SE3": "se", "SE4": "se", "SI": "si", "SK": "sk", "IT-North": "it", "IT-South": "it",
}

FORECAST_TYPES = ("solar", "wind_onshore", "wind_offshore", "load")
FORECAST_HORIZONS = {"day_ahead": "day-ahead", "intraday": "intraday", "current": "current"}
ACTUAL_TYPES = (
    "load", "residual_load", "solar", "wind_onshore", "wind_offshore", "fossil_gas",
    "fossil_hard_coal", "fossil_brown_coal_lignite", "nuclear", "hydro_pumped_storage",
    "hydro_pumped_storage_consumption", "cross_border_electricity_trading",
)


class EnergyChartsSource(DataSource):
    name = "energy_charts"

    def __init__(self, chunk_days: int = 366, pause_s: float = 1.0, retries: int = 5, timeout: int = 120):
        # The API rate-limits aggressively (HTTP 429). One request per year per series keeps a
        # 3-year backfill under ~30 requests; 429 responses honour Retry-After (min 30 s).
        self.chunk_days = chunk_days
        self.pause_s = pause_s
        self.retries = retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "powerarb/0.1 (research)"

    # ---- public --------------------------------------------------------------------------
    def available_series(self, zone: str) -> list[str]:
        out = ["price.day_ahead"]
        if zone in ZONE_TO_COUNTRY:
            out += [f"forecast.{t}.{h}" for t in FORECAST_TYPES for h in FORECAST_HORIZONS]
            out += [f"actual.{t}" for t in ACTUAL_TYPES]
        return out

    def fetch(self, zone: str, series: str, start: datetime, end: datetime) -> pd.DataFrame:
        frames = []
        for s, e in _chunks(start, end, self.chunk_days):
            frames.append(self._fetch_chunk(zone, series, s, e))
            time.sleep(self.pause_s)
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return empty_long()
        df = pd.concat(frames, ignore_index=True)
        # chunks overlap by one day at the boundary; keep the last observation
        return df.drop_duplicates(subset=["zone", "series", "ts_utc"], keep="last")

    # ---- internals -----------------------------------------------------------------------
    def _fetch_chunk(self, zone: str, series: str, start: datetime, end: datetime) -> pd.DataFrame:
        parts = series.split(".")
        if series == "price.day_ahead":
            payload = self._get("/v2/price", {"bzn": zone, "start": _day(start), "end": _day(end)})
            return self._parse(payload, "day_ahead_price", zone, series)
        country = ZONE_TO_COUNTRY.get(zone)
        if country is None:
            raise ValueError(f"{series} needs a country mapping for zone {zone}")
        if parts[0] == "forecast" and len(parts) == 3:
            _, ptype, horizon = parts
            payload = self._get(
                "/v2/public_power_forecast",
                {"country": country, "production_type": ptype,
                 "forecast_type": FORECAST_HORIZONS[horizon], "start": _day(start), "end": _day(end)},
            )
            return self._parse(payload, ptype, zone, series)
        if parts[0] == "actual" and len(parts) == 2:
            payload = self._get("/v2/public_power",
                                {"country": country, "start": _day(start), "end": _day(end)})
            return self._parse(payload, parts[1], zone, series)
        raise ValueError(f"unknown series {series}")

    def _get(self, path: str, params: dict) -> dict:
        url = BASE_URL + path
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    wait = max(30, int(r.headers.get("Retry-After", "0") or 0))
                    log.warning("energy-charts rate limited on %s %s, sleeping %ss", path, params, wait)
                    time.sleep(wait)
                    continue
                if r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code} from {url} {params}")
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:  # ValueError = bad JSON body
                last_err = e
                wait = 2 ** attempt
                log.warning("energy-charts %s %s failed (%s), retry in %ss", path, params, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"energy-charts request failed after retries: {last_err}")

    @staticmethod
    def _parse(payload: dict, value_key: str, zone: str, series: str) -> pd.DataFrame:
        data = payload.get("data") or []
        if not data:
            return empty_long()
        ts = [row["timestamp"] for row in data]
        vals = [row.get("values", {}).get(value_key) for row in data]
        df = to_long(ts, vals, zone, series, "energy_charts")
        return df.dropna(subset=["value"])


def _day(d: datetime) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _chunks(start: datetime, end: datetime, days: int):
    s = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while s < end:
        e = min(s + timedelta(days=days), end)
        yield s, e
        s = e
