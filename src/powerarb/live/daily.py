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
from ..data.sources import OpenMeteoForecastSource
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
# How much weather-forecast history to pull for training. Two years spans two winters, which
# is where the weather features earn their keep.
WEATHER_TRAIN_DAYS = 760
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

    # Weather forecasts are not carried in the repo state: one request per reference point
    # covers years, so they are re-fetched each run. A failure here must never break the
    # record, so it degrades to the weather-free feature set rather than raising.
    wx_used = False
    if settings.weather_features and zone in settings.weather_points:
        try:
            src = OpenMeteoForecastSource(settings.weather_points)
            hist_start = (now - pd.Timedelta(days=WEATHER_TRAIN_DAYS)).tz_convert(None)
            n = store.upsert(src.fetch_history(zone, hist_start, now.tz_convert(None)))
            n += store.upsert(src.fetch_live(zone))
            wx_used = n > 0
            log.info("weather forecast rows: %d", n)
        except Exception as e:
            log.error("weather fetch failed, continuing without it: %s", e)

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

    # Train on the within-day shape when configured: battery dispatch is unchanged by adding a
    # constant to all of a day's prices, so the level is not decision-relevant and modelling it
    # only spends capacity. See backtest/engine.rolling_forecast for the same transform.
    local_all = pd.Series(feats.index.tz_convert(tz).floor("D"), index=feats.index)
    y = feats["target"]
    if settings.target_mode == "shape":
        y = y - y.groupby(local_all).transform("mean")
    train = feats[y.notna()]
    model = LGBMForecaster() if model_name == "lgbm" else SeasonalNaive()
    model.fit(train[cols], y[y.notna()])
    tags = sorted(allowed) if allowed else ["no-fc"]
    if wx_used:
        tags.append("wx")
    if settings.target_mode == "shape":
        tags.append("shape")
    model_label = f"{model.name}[{'+'.join(tags)}]"
    X = feats.loc[target_idx, cols]
    pred = pd.Series(model.predict(X).values, index=target_idx)
    if settings.target_mode == "shape":
        # per-day constant: leaves the dispatch identical, restores a readable price level
        level = feats.loc[target_idx, "price_prevday_mean"].ffill().fillna(0)
        pred = pred + level.values

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


def _shape_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    """MAE after removing each series' own daily mean: how well the within-day shape was
    predicted, independent of the level. This is the error that matters for dispatch."""
    return float(np.abs((pred - pred.mean()) - (actual - actual.mean())).mean())


def backfill_shape_mae(store: TimeSeriesStore, settings: Settings, zone: str) -> int:
    """Fill shape_mae for rows scored before the column existed. Idempotent."""
    tz = settings.timezone
    scores = store.read_scores(zone)
    if scores.empty or "shape_mae" not in scores:
        return 0
    todo = scores[scores["shape_mae"].isna()]
    n = 0
    for r in todo.itertuples():
        fc = store.read_forecast(zone, r.target_day, issued_at=r.issued_at)
        if fc.empty:
            continue
        local_day = pd.Timestamp(r.target_day, tz=tz)
        actual = store.read_wide(zone, ["price.day_ahead"], start=local_day,
                                 end=local_day + pd.Timedelta(days=1), freq="1h")
        joined = fc.set_index("ts_utc")[["pred"]].join(actual, how="inner").dropna().sort_index()
        if len(joined) < 20:
            continue
        sm = _shape_mae(joined["pred"].to_numpy(), joined["price.day_ahead"].to_numpy())
        store.conn.execute(
            "UPDATE score_log SET shape_mae = ? WHERE zone = ? AND target_day = ? AND issued_at = ?",
            [sm, zone, pd.Timestamp(r.target_day).date(),
             pd.Timestamp(r.issued_at).tz_localize(None) if pd.Timestamp(r.issued_at).tzinfo is None
             else pd.Timestamp(r.issued_at).tz_convert("UTC").tz_localize(None)])
        n += 1
    return n


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
        # sort_index is load-bearing: optimize_dispatch walks the array in order, so an
        # out-of-order join would silently score a different dispatch
        joined = fc.set_index("ts_utc")[["pred"]].join(actual, how="inner").dropna().sort_index()
        if len(joined) < 20:
            continue
        m = forecast_metrics(joined["price.day_ahead"], joined["pred"])
        a = joined["price.day_ahead"].to_numpy()
        p = joined["pred"].to_numpy()
        shape_mae = _shape_mae(p, a)
        rev_fc = float(settle_dispatch(optimize_dispatch(p, 1.0, params), a, 1.0, params).sum())
        rev_pf = float(settle_dispatch(optimize_dispatch(a, 1.0, params), a, 1.0, params).sum())
        row = {
            "zone": zone, "target_day": day, "issued_at": fc["issued_at"].iloc[0],
            "on_time": bool(fc["on_time"].iloc[0]), "n": m["n"], "mae": m["mae"], "rmse": m["rmse"],
            "rank_corr": m["intraday_rank_corr"], "rev_forecast": rev_fc, "rev_perfect": rev_pf,
            "capture": rev_fc / rev_pf if rev_pf > 0 else None,
            "scored_at": pd.Timestamp.now(tz="UTC"),
            "model": str(fc["model"].iloc[0]) if "model" in fc else None,
            "shape_mae": shape_mae,
        }
        store.log_score(row)
        results.append(row)
    return results
