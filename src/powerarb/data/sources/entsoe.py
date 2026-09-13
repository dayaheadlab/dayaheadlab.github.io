"""ENTSO-E Transparency Platform via entsoe-py. Needs ENTSOE_API_KEY in .env.

Covers every European bidding zone, including forecasts/actuals that Energy-Charts only
provides for some countries. Zone codes use entsoe-py conventions (``DE_LU``, ``NO_2`` ...);
we map from the Energy-Charts style used elsewhere in the project.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..store import empty_long
from .base import DataSource, to_long

ZONE_MAP = {"DE-LU": "DE_LU", "DE-AT-LU": "DE_AT_LU", "NO1": "NO_1", "NO2": "NO_2", "NO3": "NO_3",
            "NO4": "NO_4", "NO5": "NO_5", "SE1": "SE_1", "SE2": "SE_2", "SE3": "SE_3", "SE4": "SE_4",
            "DK1": "DK_1", "DK2": "DK_2", "IT-North": "IT_NORD", "IT-South": "IT_SUD"}

_RES_COLS = {"solar": "Solar", "wind_onshore": "Wind Onshore", "wind_offshore": "Wind Offshore"}


class EntsoeSource(DataSource):
    name = "entsoe"

    def __init__(self, api_key: str | None):
        if not api_key:
            raise RuntimeError("ENTSOE_API_KEY missing; see .env.example")
        from entsoe import EntsoePandasClient  # lazy import: optional at runtime

        self.client = EntsoePandasClient(api_key=api_key)

    def available_series(self, zone: str) -> list[str]:
        return ["price.day_ahead", "actual.load", "forecast.load.day_ahead",
                "forecast.solar.day_ahead", "forecast.wind_onshore.day_ahead",
                "forecast.wind_offshore.day_ahead"]

    def fetch(self, zone: str, series: str, start: datetime, end: datetime) -> pd.DataFrame:
        code = ZONE_MAP.get(zone, zone.replace("-", "_"))
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        if series == "price.day_ahead":
            out = self.client.query_day_ahead_prices(code, start=s, end=e)
            return to_long(out.index, out.values, zone, series, self.name)
        if series == "actual.load":
            out = self.client.query_load(code, start=s, end=e)
            return to_long(out.index, out.iloc[:, 0].values, zone, series, self.name)
        if series == "forecast.load.day_ahead":
            out = self.client.query_load_forecast(code, start=s, end=e)
            return to_long(out.index, out.iloc[:, 0].values, zone, series, self.name)
        if series.startswith("forecast.") and series.endswith(".day_ahead"):
            ptype = series.split(".")[1]
            out = self.client.query_wind_and_solar_forecast(code, start=s, end=e)
            col = _RES_COLS[ptype]
            if col not in out.columns:
                return empty_long()
            return to_long(out.index, out[col].values, zone, series, self.name)
        raise ValueError(f"unknown series {series}")
