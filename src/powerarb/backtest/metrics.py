from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    m = y_true.notna() & y_pred.notna()
    yt, yp = y_true[m], y_pred[m]
    err = yp - yt
    # For arbitrage the within-day ranking of hours matters more than the level.
    frame = pd.DataFrame({"t": yt, "p": yp})
    daily_rank = frame.groupby(frame.index.floor("D")).apply(
        lambda g: g["t"].corr(g["p"], method="spearman") if len(g) > 3 else np.nan
    )
    return {
        "n": int(m.sum()),
        "mae": round(float(err.abs().mean()), 3),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        "bias": round(float(err.mean()), 3),
        "intraday_rank_corr": round(float(daily_rank.mean()), 3),
    }


def pnl_metrics(daily_pnl: pd.Series, capacity_mwh: float | None = None) -> dict[str, float]:
    d = daily_pnl.dropna()
    std = d.std()
    out = {
        "days": int(len(d)),
        "total_eur": round(float(d.sum()), 1),
        "mean_daily_eur": round(float(d.mean()), 1),
        "std_daily_eur": round(float(std), 1),
        "sharpe_daily_ann": round(float(d.mean() / std * np.sqrt(365)), 2) if std > 0 else float("nan"),
        "win_rate": round(float((d > 0).mean()), 3),
        "worst_day_eur": round(float(d.min()), 1),
        "max_drawdown_eur": round(float((d.cumsum() - d.cumsum().cummax()).min()), 1),
    }
    if capacity_mwh:
        out["eur_per_mwh_capacity_per_year"] = round(float(d.mean() * 365 / capacity_mwh), 0)
    return out
