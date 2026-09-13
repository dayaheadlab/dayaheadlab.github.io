from __future__ import annotations

import abc

import pandas as pd


class Forecaster(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series: ...
