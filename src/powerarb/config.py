"""Project settings: YAML defaults + .env secrets."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class BatteryConfig(BaseModel):
    power_mw: float = 10.0
    energy_mwh: float = 20.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_initial_mwh: float = 0.0
    cycle_cost_eur_per_mwh: float = 2.0
    max_cycles_per_day: float | None = None
    availability: float = 1.0


class Settings(BaseModel):
    default_zone: str = "DE-LU"
    timezone: str = "Europe/Berlin"
    weather_points: dict[str, dict[str, list[float]]] = Field(default_factory=dict)
    battery: BatteryConfig = Field(default_factory=BatteryConfig)
    battery_unconstrained: BatteryConfig = Field(default_factory=BatteryConfig)
    # Which D-1 published forecasts may enter the features: "none" | "load" | "all".
    # "none" is the published policy: wind/solar appear only after the 12:00 gate closure,
    # and load adds ~0.1pp of capture, so dropping every one of them removes any dependence
    # on publication timing and makes the live job identical to the published backtest.
    forecast_features: str = "none"
    # Weather forecasts are legal at gate closure (they exist at any hour) and, combined with
    # the shape target, lift winter capture by ~7pp. Measured 2026-09-16.
    weather_features: bool = True
    # "shape": train on price minus that day's mean. Dispatch is invariant to a per-day
    # constant, so this spends the whole model on the within-day ordering that capture depends
    # on. Worth ~+1.1pp overall and ~+7pp in winter versus predicting the level.
    target_mode: str = "shape"
    brand: dict[str, str] = Field(default_factory=lambda: {"name_zh": "powerarb", "name_en": "powerarb"})
    db_path: Path = PROJECT_ROOT / "data" / "powerarb.duckdb"
    entsoe_api_key: str | None = None


def load_settings(path: Path | None = None) -> Settings:
    path = path or PROJECT_ROOT / "config" / "settings.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    s = Settings(**(raw or {}))
    if env_db := os.getenv("POWERARB_DB_PATH"):
        s.db_path = Path(env_db) if Path(env_db).is_absolute() else PROJECT_ROOT / env_db
    s.entsoe_api_key = os.getenv("ENTSOE_API_KEY") or None
    return s
