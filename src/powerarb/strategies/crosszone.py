"""Cross-zone spread: what the price difference between two bidding zones is worth.

Europe's day-ahead markets are coupled: when there is enough interconnector capacity between
two zones, the algorithm equalises their prices and the spread collapses to zero. A non-zero
spread is therefore a congestion signal, and its size is what transmission capacity between
those two zones was worth in that hour.

Who can actually capture it:

- Holders of physical or financial transmission rights (PTR/FTR) on that border. A directional
  right A->B pays ``max(0, P_B - P_A)`` per MWh, which is the "option value" computed here.
- Not a merchant trader without rights: implicit coupling means you cannot simply buy in one
  zone and sell in the other. The analysis is still useful for siting a battery, valuing a
  border, or timing an auction bid for rights, but it is not a strategy an outsider can run.

This module computes the value and its predictability; it does not model the rights auction,
ramping limits, or losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Physically interconnected pairs among the zones we hold prices for. Ordered (A, B); the
# option value is reported for both directions.
NEIGHBOUR_PAIRS: list[tuple[str, str]] = [
    ("DE-LU", "FR"), ("DE-LU", "NL"), ("DE-LU", "BE"), ("DE-LU", "AT"), ("DE-LU", "CZ"),
    ("DE-LU", "PL"), ("DE-LU", "DK1"), ("DE-LU", "DK2"), ("DE-LU", "CH"),
    ("FR", "BE"), ("FR", "ES"), ("FR", "CH"), ("FR", "IT-North"),
    ("NL", "BE"), ("NL", "DK1"),
    ("AT", "CZ"), ("AT", "HU"), ("AT", "CH"), ("AT", "IT-North"),
    ("PL", "CZ"), ("PL", "SE4"),
    ("HU", "RO"),
    ("ES", "PT"),
    ("IT-North", "IT-South"), ("IT-South", "GR"),
    ("DK1", "DK2"), ("DK2", "SE4"),
    ("CH", "IT-North"),
]

CONVERGED_EPS = 0.01  # |spread| below this counts as fully coupled, i.e. no congestion


@dataclass
class SpreadStats:
    pair: str
    hours: int
    mean_abs: float          # average |P_B - P_A|, the size of the dislocation
    converged_share: float   # share of hours the coupling equalised the two prices
    option_ab: float         # EUR/MW/year for a directional right A->B
    option_ba: float         # EUR/MW/year for a directional right B->A
    sign_persistence: float  # share of hours whose sign matches the same hour a day earlier
    naive_capture: float     # what "yesterday's spread, same hour" captures of the ceiling


def price_matrix(store, zones: list[str], start, end=None, freq: str = "1h") -> pd.DataFrame:
    """One column of day-ahead prices per zone on a single explicit hourly grid.

    Do NOT build this with ``pd.DataFrame({zone: series})``. Each zone's index comes back from
    a resample carrying ``freq=<Hour>``, and when two such tz-aware indexes cover slightly
    different spans pandas' union collapses to a handful of rows instead of the superset
    (observed 2026-09-16: AT 14974 rows unioned with BE 14950 produced 16). Reindexing every
    series onto one date_range sidesteps the alignment entirely.
    """
    series = {}
    for z in zones:
        w = store.read_wide(z, ["price.day_ahead"], start=start, end=end, freq=freq)
        if not w.empty and "price.day_ahead" in w:
            series[z] = w["price.day_ahead"]
    if not series:
        return pd.DataFrame()
    lo = min(s.index.min() for s in series.values())
    hi = max(s.index.max() for s in series.values())
    grid = pd.date_range(lo, hi, freq=freq, tz="UTC")
    out = pd.DataFrame(index=grid)
    for z, s in series.items():
        out[z] = s.reindex(grid)
    return out


def spread_series(prices: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Hourly P_b - P_a, restricted to hours both zones priced."""
    return (prices[b] - prices[a]).dropna().rename(f"{a}->{b}")


def analyse_pair(prices: pd.DataFrame, a: str, b: str) -> SpreadStats | None:
    if a not in prices or b not in prices:
        return None
    s = spread_series(prices, a, b)
    if len(s) < 24 * 300:
        return None
    hours_per_year = 8760
    scale = hours_per_year / len(s)

    # A directional right pays only when the flow direction is profitable.
    option_ab = float(s.clip(lower=0).sum() * scale)
    option_ba = float((-s).clip(lower=0).sum() * scale)

    prev = s.shift(24, freq="h").reindex(s.index)
    both = pd.DataFrame({"now": s, "prev": prev}).dropna()
    sign_persistence = float((np.sign(both["now"]) == np.sign(both["prev"])).mean())

    # Naive strategy: hold the direction yesterday's same hour would have paid. Ceiling is the
    # perfect-foresight option value on the same hours.
    naive_pay = both["now"].where(both["prev"] > 0, -both["now"]).clip(lower=None)
    ceiling = both["now"].abs().sum()
    naive_capture = float(naive_pay.sum() / ceiling * 100) if ceiling > 0 else float("nan")

    return SpreadStats(
        pair=f"{a} | {b}", hours=len(s), mean_abs=float(s.abs().mean()),
        converged_share=float((s.abs() < CONVERGED_EPS).mean() * 100),
        option_ab=option_ab, option_ba=option_ba,
        sign_persistence=sign_persistence, naive_capture=naive_capture,
    )


def analyse_all(prices: pd.DataFrame,
                pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    rows = [r for a, b in (pairs or NEIGHBOUR_PAIRS) if (r := analyse_pair(prices, a, b))]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r.__dict__ for r in rows])
    df["best_option"] = df[["option_ab", "option_ba"]].max(axis=1) / 1000  # k EUR/MW/year
    return df.sort_values("best_option", ascending=False).reset_index(drop=True)
