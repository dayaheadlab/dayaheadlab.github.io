from __future__ import annotations

import abc
from datetime import datetime

import pandas as pd

from ..store import LONG_COLUMNS


def to_long(ts, values, zone: str, series: str, source: str) -> pd.DataFrame:
    """Build a long-format frame with the store's canonical columns."""
    ts_index = pd.to_datetime(pd.Index(ts), utc=True)
    df = pd.DataFrame({"ts_utc": ts_index, "value": list(values)})
    df["zone"] = zone
    df["series"] = series
    df["source"] = source
    return df[LONG_COLUMNS]


class DataSource(abc.ABC):
    """A source produces long frames for (zone, series, start, end)."""

    name: str = "base"

    @abc.abstractmethod
    def available_series(self, zone: str) -> list[str]: ...

    @abc.abstractmethod
    def fetch(self, zone: str, series: str, start: datetime, end: datetime) -> pd.DataFrame: ...
