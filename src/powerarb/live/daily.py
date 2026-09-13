"""Daily live forecast: issue tomorrow's day-ahead price forecast before gate closure, log it,
and later score it against the realised price.

Credibility rules (this is the public track record, so they are strict):
- A forecast is ``on_time`` only if issued before 12:00 local on D-1 (EPEX DA gate closure).
  Late forecasts are still logged but flagged, and the public page counts only on-time ones.
- Target-day prices are masked before feature building, so running the job late can never
  leak the published price into the forecast.
- Each day's model is retrained on all history up to D-1.
"""
from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from ..backtest.metrics import forecast_metrics
from ..config import Settings
from ..data.ingest import update
from ..data.store import TimeSeriesStore
from ..features.build import build_features, feature_columns
from ..models import LGBMForecaster, SeasonalNaive
from ..strategies.battery import BatteryParams, optimize_dispatch, settle_dispatch

log = logging.getLogger(__name__)

GATE_HOUR_LOCAL = 12
FORECAST_INPUTS = ["forecast.load.day_ahead", "forecast.solar.day_ahead",
                   "forecast.wind_onshore.day_ahead", "forecast.wind_offshore.day_ahead"]
# A published series counts as usable only if it covers essentially the whole target day.
MIN_AVAILABILITY = 0.95
# settings.forecast_features -> the production types the policy permits (None = all).
POLICY_SETS: dict[str, set[str] | None] = {"none": set(), "load": {"load"}, "all": None}


def gate_closure(target_day: pd.Timestamp) -> pd.Timestamp:
    """12:00 local on the day before ``target_day`` (a tz-aware local midnight)."""
    return (target_day - pd.Timedelta(days=1)) + pd.Timedelta(hours=GATE_HOUR_LOCAL)


def run_daily(
    store: TimeSeriesStore,
    settings: Settings,
    zone: str,
    now: pd.Timestamp | None = None,
    model_name: str = "lgbm",
    do_update: bool = True,
    target_day: pd.Timestamp | None = None,
    update_series: list[str] | None = None,
) -> dict:
    tz = settings.timezone
    now = (now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
    now_local = now.tz_convert(tz)
    if target_day is None:
        target_day = (now_local + pd.Timedelta(days=1)).normalize()
    gate = gate_closure(target_day)
    on_time = now_local < gate

    if do_update:
        counts = update(store, settings, zone, series=update_series)
        log.info("update: %s", counts)

    wide = store.read_wide(zone, freq="1h")
    day_start = target_day.tz_convert("UTC")
    day_end = (target_day + pd.Timedelta(days=1)).tz_convert("UTC")
    target_idx = pd.date_range(day_start, day_end - pd.Timedelta(hours=1), freq="1h")
    wide = wide.reindex(wide.index.union(target_idx))

    # Never let the target day's published price (or anything later) into the features.
    if "price.day_ahead" in wide:
        wide.loc[wide.index >= day_start, "price.day_ahead"] = np.nan

    availability = {}
    for c in FORECAST_INPUTS:
        availability[c] = float(wide.loc[target_idx, c].notna().mean()) if c in wide else 0.0

    # Two filters, in order.
    # 1. Policy (settings.forecast_features): the published methodology. Default "none",
    #    because ENTSO-E may publish day-ahead wind/solar as late as 18:00 on D-1, after the
    #    12:00 gate closure. This keeps the live job identical to the published backtest.
    # 2. Measured availability: even a policy-allowed series is dropped if it is not actually
    #    published for the target day, from BOTH training and prediction, so the model never
    #    sees a feature at fit time that is blank at predict time.
    policy = POLICY_SETS[settings.forecast_features]
    if policy is None:
        policy = {c.split(".")[1] for c in FORECAST_INPUTS}
    allowed = {c.split(".")[1] for c in FORECAST_INPUTS
               if c.split(".")[1] in policy and availability[c] >= MIN_AVAILABILITY}
    dropped = sorted(policy - allowed)
    if dropped:
        log.warning("day-ahead forecasts unavailable for %s, excluded from features: %s",
                    target_day.date(), dropped)

    feats = build_features(wide, zone, tz, allowed_forecasts=allowed)
    cols = feature_columns(feats)
    train = feats[feats["target"].notna()]
    model = LGBMForecaster() if model_name == "lgbm" else SeasonalNaive()
    model.fit(train[cols], train["target"])
    model_label = f"{model.name}[{'+'.join(sorted(allowed)) if allowed else 'no-fc'}]"
    X = feats.loc[target_idx, cols]
    pred = model.predict(X)
    pred.index = target_idx

    params = replace(BatteryParams(**settings.battery.model_dump()),
                     soc_final_min_mwh=settings.battery.soc_initial_mwh)
    dispatch = optimize_dispatch(pred.to_numpy(), 1.0, params)
    dispatch.index = target_idx

    store.log_forecast(zone, target_day, now, pred, model_label, on_time)
    log.info("forecast for %s issued at %s (on_time=%s) model=%s mean=%.1f", target_day.date(), now,
             on_time, model_label, float(pred.mean()))
    return {
        "zone": zone, "target_day": target_day, "issued_at": now, "on_time": on_time,
        "gate": gate, "pred": pred, "dispatch": dispatch, "availability": availability,
        "train_rows": int(len(train)), "model": model_label, "dropped_forecasts": dropped,
    }


def score_pending(store: TimeSeriesStore, settings: Settings, zone: str) -> list[dict]:
    """Score every logged target day whose realised hourly prices are now complete."""
    tz = settings.timezone
    scored_days = set(pd.to_datetime(store.read_scores(zone)["target_day"]).dt.date) \
        if not store.read_scores(zone).empty else set()
    params = replace(BatteryParams(**settings.battery.model_dump()),
                     soc_final_min_mwh=settings.battery.soc_initial_mwh)
    results = []
    for day in store.forecast_days(zone):
        if day in scored_days:
            continue
        fc = store.read_forecast(zone, day)
        if fc.empty:
            continue
        local_day = pd.Timestamp(day, tz=tz)
        actual = store.read_wide(zone, ["price.day_ahead"], start=local_day,
                                 end=local_day + pd.Timedelta(days=1), freq="1h")
        if actual.empty or actual["price.day_ahead"].notna().sum() < 20:
            continue
        joined = fc.set_index("ts_utc")[["pred"]].join(actual, how="inner").dropna()
        if len(joined) < 20:
            continue
        m = forecast_metrics(joined["price.day_ahead"], joined["pred"])
        a = joined["price.day_ahead"].to_numpy()
        p = joined["pred"].to_numpy()
        rev_fc = float(settle_dispatch(optimize_dispatch(p, 1.0, params), a, 1.0, params).sum())
        rev_pf = float(settle_dispatch(optimize_dispatch(a, 1.0, params), a, 1.0, params).sum())
        row = {
            "zone": zone, "target_day": day, "issued_at": fc["issued_at"].iloc[0],
            "on_time": bool(fc["on_time"].iloc[0]), "n": m["n"], "mae": m["mae"], "rmse": m["rmse"],
            "rank_corr": m["intraday_rank_corr"], "rev_forecast": rev_fc, "rev_perfect": rev_pf,
            "capture": rev_fc / rev_pf if rev_pf > 0 else None,
            "scored_at": pd.Timestamp.now(tz="UTC"),
        }
        store.log_score(row)
        results.append(row)
    return results
