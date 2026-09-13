from __future__ import annotations

import pandas as pd

from .base import Forecaster


class LGBMForecaster(Forecaster):
    name = "lgbm"

    def __init__(self, **params):
        self.params = {
            "n_estimators": 600, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 40,
            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
            "verbose": -1, "random_state": 42,
        } | params
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import lightgbm as lgb

        mask = y.notna()
        self.model = lgb.LGBMRegressor(**self.params).fit(X[mask], y[mask])
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index, name="pred")

    def feature_importance(self, columns) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=columns).sort_values(ascending=False)
