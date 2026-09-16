"""Walk-forward forecasting + strategy settlement.

The DA auction for day D clears at 12:00 CET on D-1, so a forecast for D may use everything
published up to that moment. ``rolling_forecast`` retrains every ``retrain_every`` days on all
history before D and predicts D. ``battery_backtest`` dispatches on the forecast, settles at
the realised DA price, and compares to perfect-foresight dispatch.
"""
from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from ..features.build import feature_columns
from ..models.base import Forecaster
from ..strategies.battery import BatteryParams, optimize_dispatch, settle_dispatch

log = logging.getLogger(__name__)


def rolling_forecast(
    feats: pd.DataFrame,
    model: Forecaster,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    retrain_every: int = 7,
    min_train_days: int = 90,
    tz: str = "Europe/Berlin",
    target_mode: str = "level",
) -> pd.DataFrame:
    """Return frame [target, pred] over [start, end) at the feature index resolution.

    ``target_mode="shape"`` trains on the price minus that delivery day's mean instead of the
    price itself. Battery dispatch is invariant to adding a constant to all of a day's prices,
    so the level carries no decision-relevant information; predicting only the shape spends
    the whole model on the within-day ordering, which is what capture ratio depends on. The
    prediction is shifted back by a per-day constant (the previous day's mean) purely so that
    MAE stays comparable; that shift cannot change the dispatch.
    """
    cols = feature_columns(feats)
    if target_mode not in ("level", "shape"):
        raise ValueError(f"unknown target_mode {target_mode}")
    local_day = pd.Series(feats.index.tz_convert(tz).floor("D"), index=feats.index)
    days = pd.DatetimeIndex(sorted(local_day.unique()))
    start_ts = pd.Timestamp(start, tz=tz)
    end_ts = pd.Timestamp(end, tz=tz)
    test_days = days[(days >= start_ts) & (days < end_ts)]
    y = feats["target"]
    if target_mode == "shape":
        day_mean = y.groupby(local_day).transform("mean")
        y = y - day_mean

    preds = []
    fitted = None
    for i, day in enumerate(test_days):
        train_mask = (local_day < day) & y.notna()
        if train_mask.sum() < min_train_days * 20:
            continue
        if fitted is None or i % retrain_every == 0:
            fitted = model.fit(feats.loc[train_mask, cols], y[train_mask])
            log.info("retrained %s on %d rows up to %s", model.name, int(train_mask.sum()), day.date())
        test_mask = local_day == day
        X = feats.loc[test_mask, cols]
        p = pd.Series(fitted.predict(X).values, index=X.index)
        if target_mode == "shape":
            # per-day constant; leaves the dispatch unchanged, makes MAE readable
            p = p + feats.loc[test_mask, "price_prevday_mean"].fillna(0)
        preds.append(pd.DataFrame({"target": feats.loc[test_mask, "target"], "pred": p.values},
                                  index=X.index))
    if not preds:
        return pd.DataFrame(columns=["target", "pred"])
    return pd.concat(preds).sort_index()


def battery_backtest(
    forecasts: pd.DataFrame,
    params: BatteryParams,
    dt_hours: float = 1.0,
    tz: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Daily results: revenue on forecast dispatch vs perfect foresight."""
    rows = []
    day = forecasts.index.tz_convert(tz).floor("D")
    p = replace(params, soc_final_min_mwh=params.soc_initial_mwh)  # end each day where it began
    for d, g in forecasts.groupby(day):
        # sorted because optimize_dispatch walks the array in chronological order
        g = g.dropna(subset=["target"]).sort_index()
        if len(g) < 20:
            continue
        actual = g["target"].to_numpy()
        pred = g["pred"].fillna(float(np.mean(actual))).to_numpy()
        disp_fc = optimize_dispatch(pred, dt_hours, p)
        rev_fc = float(settle_dispatch(disp_fc, actual, dt_hours, p).sum())
        disp_pf = optimize_dispatch(actual, dt_hours, p)
        rev_pf = float(settle_dispatch(disp_pf, actual, dt_hours, p).sum())
        rows.append({
            "day": d.date(),
            "revenue_forecast_eur": rev_fc,
            "revenue_perfect_eur": rev_pf,
            "capture_ratio": rev_fc / rev_pf if rev_pf > 0 else np.nan,
            "da_spread_eur": float(actual.max() - actual.min()),
            "cycles": float(disp_fc["discharge_mw"].sum() * dt_hours / params.energy_mwh),
        })
    return pd.DataFrame(rows).set_index("day")
