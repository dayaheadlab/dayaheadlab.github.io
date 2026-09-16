"""Feature matrix for day-ahead price forecasting.

Design rule: every feature for delivery hour t on day D must be known at the day-ahead gate
closure (12:00 CET on D-1). The DA curve for D-1 was published at ~13:00 on D-2, so at gate
closure the most recent full day of prices is D-1: a 24h lag is legal, anything shorter is
leakage. The only information about D itself is the D-1 published forecasts for D.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

FEATURE_TARGET = "price.day_ahead"

ZONE_HOLIDAY_COUNTRY = {"DE-LU": "DE", "AT": "AT", "FR": "FR", "NL": "NL", "BE": "BE", "CH": "CH",
                        "PL": "PL", "CZ": "CZ", "ES": "ES", "IT-North": "IT", "DK1": "DK", "DK2": "DK",
                        "NO2": "NO", "SE3": "SE", "FI": "FI"}


def build_features(wide: pd.DataFrame, zone: str, tz: str = "Europe/Berlin",
                   allowed_forecasts: set[str] | None = None,
                   weather_raw: bool = False) -> pd.DataFrame:
    """Take an hourly wide frame from the store and return features + ``target``.

    Rows where the target is NaN are kept so the same function can produce next-day inputs.

    ``allowed_forecasts`` restricts which ``forecast.*.day_ahead`` columns may be used, by
    production type (e.g. ``{"load"}``). Default ``None`` keeps all of them. This matters:
    ENTSO-E only requires day-ahead wind/solar forecasts to be published by 18:00 on D-1,
    which is AFTER the 12:00 day-ahead gate closure, so using them is leakage. Measured on
    2026-09-13 at 13:18 Berlin, the wind and solar forecasts for D+1 were still absent while
    the load forecast was already published.
    """
    df = wide.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    local = df.index.tz_convert(tz)
    out = pd.DataFrame(index=df.index)

    # ---- calendar --------------------------------------------------------------------
    out["hour"] = local.hour
    out["dow"] = local.dayofweek
    out["month"] = local.month
    out["doy"] = local.dayofyear
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    country = ZONE_HOLIDAY_COUNTRY.get(zone, "DE")
    hol = holidays.country_holidays(country, years=sorted(set(local.year)))
    out["is_holiday"] = np.array([d in hol for d in local.date]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)

    # ---- D-1 published forecasts for D (known at gate closure) ------------------------
    fc_cols = [c for c in df.columns if c.startswith("forecast.") and c.endswith(".day_ahead")]
    if allowed_forecasts is not None:
        fc_cols = [c for c in fc_cols if c.split(".")[1] in allowed_forecasts]
    for c in fc_cols:
        out[c] = df[c]

    def _fc(ptype: str) -> pd.Series | None:
        col = f"forecast.{ptype}.day_ahead"
        return df[col] if col in fc_cols and col in df else None

    zero = pd.Series(0.0, index=df.index)
    load = _fc("load")
    won, woff, sol = _fc("wind_onshore"), _fc("wind_offshore"), _fc("solar")
    # Residual load only makes sense when both load and the renewables are in the feature set.
    if load is not None and (won is not None or woff is not None or sol is not None):
        wind = (won if won is not None else zero).fillna(0) + (woff if woff is not None else zero).fillna(0)
        solar = (sol if sol is not None else zero).fillna(0)
        out["fc_residual_load"] = load - wind - solar
        out["fc_ren_share"] = (wind + solar) / load.replace(0, np.nan)

    # ---- weather forecast (available at any hour, so always legal at gate closure) --------
    # Measured 2026-09-16: feeding all 12 raw point series plus aggregates helped in winter
    # (+3.5pp capture) but hurt the rest of the year (-1.2pp), i.e. it overfit. Only the zone
    # aggregates are kept by default; set weather_raw to put the point series back.
    wx_cols = [c for c in df.columns if c.startswith("wxfc.")]
    if weather_raw:
        for c in wx_cols:
            out[c] = df[c]
    # Zone aggregates: the model mostly cares about how much wind and sun the whole zone gets,
    # and the spread between reference points carries the "is the weather front across the
    # country or only in the north" information that drives congestion and ramping.
    for var in ("wind_speed_100m", "shortwave_radiation", "temperature_2m"):
        cols = [c for c in wx_cols if c.split(".")[1] == var]
        if len(cols) >= 2:
            out[f"wx_{var}_mean"] = df[cols].mean(axis=1)
            out[f"wx_{var}_spread"] = df[cols].max(axis=1) - df[cols].min(axis=1)
        elif cols:
            out[f"wx_{var}_mean"] = df[cols[0]]
    if "wx_wind_speed_100m_mean" in out:
        # turbine output rises roughly with the cube of wind speed up to rated power
        out["wx_wind_cubed"] = out["wx_wind_speed_100m_mean"].clip(upper=15) ** 3
        # rolling means capture whether a windy spell is building or fading
        out["wx_wind_24h_mean"] = out["wx_wind_speed_100m_mean"].rolling(24, min_periods=6).mean()

    # ---- price lags (>= 24h) -------------------------------------------------------------
    if FEATURE_TARGET in df.columns:
        p = df[FEATURE_TARGET]
    else:
        p = pd.Series(np.nan, index=df.index)
    for lag in (24, 48, 168):
        out[f"price_lag_{lag}h"] = p.shift(lag, freq="h").reindex(df.index)
    daily = p.resample("D").agg(["mean", "std", "min", "max"])
    daily.index = daily.index + pd.Timedelta(days=1)  # yesterday's stats are known today
    daily = daily.reindex(df.index, method="ffill")
    out["price_prevday_mean"] = daily["mean"]
    out["price_prevday_std"] = daily["std"]
    out["price_prevday_min"] = daily["min"]
    out["price_prevday_max"] = daily["max"]
    out["price_roll7d_mean"] = p.shift(24, freq="h").rolling("7D").mean().reindex(df.index)

    out["target"] = p
    return out


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c != "target"]
