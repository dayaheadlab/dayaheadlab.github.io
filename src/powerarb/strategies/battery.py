"""Battery / pumped-storage arbitrage: LP dispatch against a price curve.

Variables per period t (dt hours): charge c_t [MW], discharge d_t [MW], soc_t [MWh].
    maximise  sum_t p_t * (d_t - c_t) * dt  -  cycle_cost * d_t * dt
    s.t.      soc_t = soc_{t-1} + eta_c * c_t * dt - d_t * dt / eta_d
              0 <= soc_t <= E,  0 <= c_t, d_t <= P
Solved with scipy HiGHS. Simultaneous charge and discharge is never optimal when prices are
finite and round-trip efficiency < 1, so no binary variables are needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass
class BatteryParams:
    power_mw: float = 10.0
    energy_mwh: float = 20.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_initial_mwh: float = 0.0
    cycle_cost_eur_per_mwh: float = 2.0
    soc_final_min_mwh: float | None = None  # if set, require soc_T >= this (e.g. == initial)
    max_cycles_per_day: float | None = None  # cap on discharged energy per horizon, in multiples of E
    availability: float = 1.0  # fraction of time the asset is available; scales revenue linearly


def optimize_dispatch(prices: np.ndarray, dt_hours: float, params: BatteryParams) -> pd.DataFrame:
    """Return per-period charge, discharge (MW), soc (MWh) and revenue (EUR) for given prices."""
    p = np.asarray(prices, dtype=float)
    T = len(p)
    if T == 0:
        return pd.DataFrame(columns=["charge_mw", "discharge_mw", "soc_mwh", "revenue_eur"])
    P, E = params.power_mw, params.energy_mwh
    ec, ed = params.eta_charge, params.eta_discharge

    # x = [c_0..c_{T-1}, d_0..d_{T-1}, soc_0..soc_{T-1}]; linprog minimises, so negate revenue
    n = 3 * T
    cost = np.zeros(n)
    cost[:T] = p * dt_hours
    cost[T:2 * T] = -(p - params.cycle_cost_eur_per_mwh) * dt_hours

    # SOC balance: soc_t - soc_{t-1} - ec*dt*c_t + dt/ed*d_t = 0  (soc_{-1} = soc_initial)
    rows, cols, vals = [], [], []
    rhs = np.zeros(T)
    for t in range(T):
        rows += [t, t, t]
        cols += [2 * T + t, t, T + t]
        vals += [1.0, -ec * dt_hours, dt_hours / ed]
        if t > 0:
            rows.append(t)
            cols.append(2 * T + t - 1)
            vals.append(-1.0)
        else:
            rhs[t] = params.soc_initial_mwh
    A_eq = coo_matrix((vals, (rows, cols)), shape=(T, n)).tocsr()

    bounds = [(0, P)] * (2 * T) + [(0, E)] * T
    if params.soc_final_min_mwh is not None:
        bounds[-1] = (min(params.soc_final_min_mwh, E), E)

    A_ub = b_ub = None
    if params.max_cycles_per_day is not None:
        # total discharged energy over the horizon <= cycles * E (horizon scaled to days)
        horizon_days = T * dt_hours / 24.0
        row = np.zeros(n)
        row[T:2 * T] = dt_hours
        A_ub = row.reshape(1, -1)
        b_ub = np.array([params.max_cycles_per_day * E * horizon_days])

    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=rhs, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"battery LP failed: {res.message}")
    x = res.x
    c, d, soc = x[:T], x[T:2 * T], x[2 * T:]
    revenue = (d - c) * p * dt_hours * params.availability
    return pd.DataFrame({"charge_mw": c, "discharge_mw": d, "soc_mwh": soc, "revenue_eur": revenue})


def settle_dispatch(dispatch: pd.DataFrame, actual_prices: np.ndarray, dt_hours: float,
                    params: BatteryParams) -> pd.Series:
    """Revenue when a dispatch planned on forecast prices is settled at actual prices."""
    net = (dispatch["discharge_mw"] - dispatch["charge_mw"]).to_numpy()
    rev = net * np.asarray(actual_prices, dtype=float) * dt_hours
    rev -= dispatch["discharge_mw"].to_numpy() * params.cycle_cost_eur_per_mwh * dt_hours
    return pd.Series(rev * params.availability, index=dispatch.index, name="revenue_eur")
