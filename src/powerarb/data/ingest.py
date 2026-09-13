"""Backfill and incremental update orchestration."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..config import Settings
from .sources import EnergyChartsSource, OpenMeteoSource
from .store import TimeSeriesStore

log = logging.getLogger(__name__)

# What a standard backfill pulls from Energy-Charts for a zone it covers.
CORE_SERIES = [
    "price.day_ahead",
    "forecast.load.day_ahead",
    "forecast.solar.day_ahead",
    "forecast.wind_onshore.day_ahead",
    "forecast.wind_offshore.day_ahead",
    "actual.load",
    "actual.solar",
    "actual.wind_onshore",
    "actual.wind_offshore",
]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def backfill(
    store: TimeSeriesStore,
    settings: Settings,
    zone: str,
    start: datetime,
    end: datetime | None = None,
    series: list[str] | None = None,
    weather: bool = False,
) -> dict[str, int]:
    end = end or (_now() + timedelta(days=2))
    ec = EnergyChartsSource()
    wanted = series or [s for s in CORE_SERIES if s in ec.available_series(zone)]
    counts: dict[str, int] = {}
    for s in wanted:
        log.info("backfill %s %s %s -> %s", zone, s, start, end)
        try:
            df = ec.fetch(zone, s, start, end)
        except Exception as e:  # keep going; -1 marks a failed series in the report
            log.error("failed %s %s: %s", zone, s, e)
            counts[s] = -1
            continue
        counts[s] = store.upsert(df)
    if weather and zone in settings.weather_points:
        om = OpenMeteoSource(settings.weather_points)
        df = om.fetch_all(zone, start, end)
        counts["weather.*"] = store.upsert(df)
    return counts


def update(store: TimeSeriesStore, settings: Settings, zone: str, lookback_days: int = 3,
           series: list[str] | None = None) -> dict[str, int]:
    """Refresh series from (last timestamp - lookback) to now + 2 days.

    ``series`` defaults to CORE_SERIES. The cloud runner passes just ``price.day_ahead``:
    under the leak-free feature policy that is the only series a forecast needs.
    """
    ec = EnergyChartsSource()
    counts: dict[str, int] = {}
    end = _now() + timedelta(days=2)
    for s in (series or CORE_SERIES):
        if s not in ec.available_series(zone):
            continue
        last = store.last_timestamp(zone, s)
        if last is not None:
            start = (last - pd.Timedelta(days=lookback_days)).tz_localize(None).to_pydatetime()
        else:
            start = end - timedelta(days=30)
        df = ec.fetch(zone, s, start, end)
        counts[s] = store.upsert(df)
    return counts
