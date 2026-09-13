from __future__ import annotations

import pandas as pd

from .base import Forecaster


class SeasonalNaive(Forecaster):
    """Predict the price of the same hour ``lag`` hours ago (default: one week)."""

    def __init__(self, lag_col: str = "price_lag_168h", fallback_col: str = "price_lag_24h"):
        self.lag_col, self.fallback_col = lag_col, fallback_col
        self.name = f"naive_{lag_col}"

    def fit(self, X, y):
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X[self.lag_col].fillna(X[self.fallback_col]).rename("pred")
