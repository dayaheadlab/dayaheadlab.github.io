"""Day-ahead vs intraday spread strategy (virtual position, default 1 MW).

Signal: a model's prediction of (intraday_ref - day_ahead) for each delivery hour.
If predicted spread > +threshold: buy in DA, sell back in ID (position +1).
If predicted spread < -threshold: sell in DA, buy back in ID (position -1).
PnL_t = position_t * (ID_t - DA_t) * volume - fees.

Data note: free public sources do not carry intraday prices. The ``price.intraday_ref``
series must come from EPEX (paid), Netztransparenz ID-AEP (DE, free with registration), or
a broker feed. The strategy is data-ready; the adapter is on the roadmap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DayAheadIntradaySpread:
    threshold_eur: float = 3.0
    volume_mw: float = 1.0
    fee_eur_per_mwh: float = 0.1  # round-trip exchange fees, approximate

    def positions(self, predicted_spread: pd.Series) -> pd.Series:
        pos = np.sign(predicted_spread.where(predicted_spread.abs() > self.threshold_eur, 0.0))
        return pos.fillna(0.0).rename("position")

    def pnl(self, positions: pd.Series, da: pd.Series, intraday: pd.Series,
            dt_hours: float = 1.0) -> pd.Series:
        spread = (intraday - da).reindex(positions.index)
        gross = positions * spread * self.volume_mw * dt_hours
        fees = positions.abs() * self.fee_eur_per_mwh * self.volume_mw * dt_hours
        return (gross - fees).rename("pnl_eur")
