"""Command line: ``python -m powerarb.cli --help`` (or ``powerarb`` after ``pip install -e .``)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer
from rich import print as rprint
from rich.markup import escape
from rich.table import Table

from .config import load_settings
from .data.ingest import backfill as _backfill
from .data.ingest import update as _update
from .data.store import TimeSeriesStore

app = typer.Typer(help="powerarb: European power market arbitrage research")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _store() -> TimeSeriesStore:
    return TimeSeriesStore(load_settings().db_path)


@app.command()
def backfill(
    zone: str = typer.Option(None, help="bidding zone, e.g. DE-LU"),
    start: str = typer.Option("2023-01-01"),
    end: str = typer.Option(None),
    series: list[str] = typer.Option(None, help="restrict to these series"),
    weather: bool = typer.Option(False, help="also pull Open-Meteo weather points"),
):
    """Pull history from Energy-Charts (and optionally Open-Meteo) into the local store."""
    s = load_settings()
    zone = zone or s.default_zone
    store = _store()
    counts = _backfill(store, s, zone, pd.Timestamp(start).to_pydatetime(),
                       pd.Timestamp(end).to_pydatetime() if end else None, series or None, weather)
    for k, v in counts.items():
        rprint(f"  {k:40s} {v:>8} rows")


@app.command()
def update(zone: str = typer.Option(None)):
    """Incremental refresh of all core series."""
    s = load_settings()
    counts = _update(_store(), s, zone or s.default_zone)
    for k, v in counts.items():
        rprint(f"  {k:40s} {v:>8} rows")


@app.command()
def coverage():
    """Show what is in the store."""
    df = _store().coverage()
    t = Table("zone", "series", "start", "end", "rows")
    for r in df.itertuples():
        t.add_row(r.zone, r.series, str(r.start)[:16], str(r.end)[:16], str(r.n))
    rprint(t)


@app.command()
def backtest(
    zone: str = typer.Option(None),
    start: str = typer.Option("2025-01-01", help="first test day (local)"),
    end: str = typer.Option(None, help="last test day exclusive (local)"),
    model: str = typer.Option("lgbm", help="lgbm | naive"),
    retrain_every: int = typer.Option(7),
    out: str = typer.Option(None, help="CSV path for daily results"),
    forecasts_csv: str = typer.Option(None, help="reuse saved hourly forecasts instead of re-running the model"),
    save_forecasts: str = typer.Option(None, help="save hourly forecasts to this CSV for later re-evaluation"),
    battery: str = typer.Option("battery", help="settings key: battery | battery_unconstrained"),
    forecast_features: str = typer.Option(
        "all", help="which D-1 forecasts to use: all | load | none  (wind/solar are published "
                    "after the 12:00 gate closure, so 'all' leaks)"),
):
    """Walk-forward DA price forecast + battery arbitrage settlement."""
    from .backtest import battery_backtest, forecast_metrics, pnl_metrics, rolling_forecast
    from .features import build_features
    from .models import LGBMForecaster, SeasonalNaive
    from .strategies import BatteryParams

    s = load_settings()
    zone = zone or s.default_zone
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if forecasts_csv:
        fc = pd.read_csv(forecasts_csv, index_col=0, parse_dates=True)
        fc.index = pd.to_datetime(fc.index, utc=True)
    else:
        wide = _store().read_wide(zone, freq="1h")
        if wide.empty or "price.day_ahead" not in wide:
            rprint("[red]no price data; run `backfill` first[/red]")
            raise typer.Exit(1)
        allowed = {"all": None, "load": {"load"}, "none": set()}[forecast_features]
        feats = build_features(wide, zone, s.timezone, allowed_forecasts=allowed)
        m = LGBMForecaster() if model == "lgbm" else SeasonalNaive()
        fc = rolling_forecast(feats, m, start, end, retrain_every=retrain_every, tz=s.timezone)
    if fc.empty:
        rprint("[red]no forecasts produced (not enough training history?)[/red]")
        raise typer.Exit(1)
    if save_forecasts:
        Path(save_forecasts).parent.mkdir(parents=True, exist_ok=True)
        fc.to_csv(save_forecasts)
        rprint(f"wrote {save_forecasts}")
    rprint("[bold]Forecast quality[/bold]", forecast_metrics(fc["target"], fc["pred"]))
    cfg = getattr(s, battery)
    params = BatteryParams(**cfg.model_dump())
    rprint(f"[bold]Battery case '{battery}'[/bold]: {params}")
    daily = battery_backtest(fc, params, dt_hours=1.0, tz=s.timezone)
    per_mw = lambda d: round(float(d.mean() * 365 / params.power_mw), 0)  # noqa: E731
    rprint("[bold]Battery on forecast[/bold]",
           pnl_metrics(daily["revenue_forecast_eur"], params.energy_mwh)
           | {"eur_per_mw_per_year": per_mw(daily["revenue_forecast_eur"])})
    rprint("[bold]Battery perfect foresight[/bold]",
           pnl_metrics(daily["revenue_perfect_eur"], params.energy_mwh)
           | {"eur_per_mw_per_year": per_mw(daily["revenue_perfect_eur"])})
    cap_w = daily["revenue_forecast_eur"].sum() / daily["revenue_perfect_eur"].sum()
    rprint(f"capture ratio: {cap_w * 100:.1f}% revenue-weighted / "
           f"{daily['capture_ratio'].mean() * 100:.1f}% daily-mean")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(out)
        rprint(f"wrote {out}")


@app.command()
def daily(
    zone: str = typer.Option(None),
    model: str = typer.Option("lgbm"),
    no_update: bool = typer.Option(False, help="skip the data refresh (offline / tests)"),
    site: bool = typer.Option(True, help="rebuild the static site afterwards"),
):
    """Live job: refresh data, score yesterday's forecast, issue tomorrow's, rebuild the site."""
    from .live import build_site, run_daily, score_pending

    s = load_settings()
    zone = zone or s.default_zone
    store = _store()
    scored = score_pending(store, s, zone)
    for r in scored:
        rprint(f"  scored {r['target_day']}: MAE {r['mae']:.1f}, rank {r['rank_corr']:.3f}, "
               f"capture {(r['capture'] or 0) * 100:.1f}% (on_time={r['on_time']})")
    res = run_daily(store, s, zone, model_name=model, do_update=not no_update)
    rprint(f"[bold]forecast {res['target_day'].date()}[/bold] issued {res['issued_at']} "
           f"on_time={res['on_time']} model={escape(res['model'])} (gate {res['gate']})")
    rprint(f"  inputs available for target day: {res['availability']}")
    rprint(f"  mean {res['pred'].mean():.1f}  min {res['pred'].min():.1f}  max {res['pred'].max():.1f} EUR/MWh; "
           f"planned battery revenue {res['dispatch']['revenue_eur'].sum():,.0f} EUR")
    if site:
        path = build_site(store, s, zone)
        rprint(f"site written to {path}")


@app.command()
def review(
    power_mw: float = typer.Option(..., help="battery power"),
    energy_mwh: float = typer.Option(..., help="battery energy"),
    zone: str = typer.Option(None),
    eta: float = typer.Option(0.95, help="one-way efficiency (charge = discharge)"),
    cycles: float = typer.Option(1.5, help="max cycles per day; 0 = unlimited"),
    availability: float = typer.Option(0.95),
    cycle_cost: float = typer.Option(2.0, help="EUR per MWh discharged"),
    client: str = typer.Option("项目", help="label printed on the page (no personal data)"),
    forecasts_csv: str = typer.Option(None, help="saved hourly forecasts; default outputs/<zone>_lgbm_forecasts.csv"),
    out: str = typer.Option(None),
):
    """One-page review: revenue of a client-specified battery on the saved walk-forward forecasts."""
    from .backtest import battery_backtest
    from .strategies import BatteryParams

    s = load_settings()
    zone = zone or s.default_zone
    key = zone.lower().replace("-", "_")
    forecasts_csv = forecasts_csv or f"outputs/{key}_lgbm_forecasts.csv"
    fc = pd.read_csv(forecasts_csv, index_col=0, parse_dates=True)
    fc.index = pd.to_datetime(fc.index, utc=True)
    params = BatteryParams(power_mw=power_mw, energy_mwh=energy_mwh, eta_charge=eta, eta_discharge=eta,
                           cycle_cost_eur_per_mwh=cycle_cost, availability=availability,
                           max_cycles_per_day=cycles if cycles > 0 else None)
    daily = battery_backtest(fc, params, dt_hours=1.0, tz=s.timezone)
    n = len(daily)
    pf_yr = daily["revenue_perfect_eur"].sum() / n * 365 / power_mw / 1000
    fc_yr = daily["revenue_forecast_eur"].sum() / n * 365 / power_mw / 1000
    cap = daily["revenue_forecast_eur"].sum() / daily["revenue_perfect_eur"].sum()
    cap_d = float(daily["capture_ratio"].mean())
    daily["month"] = pd.to_datetime(daily.index).strftime("%Y-%m")
    g = daily.groupby("month").agg(days=("capture_ratio", "size"), pf=("revenue_perfect_eur", "sum"),
                                   fcr=("revenue_forecast_eur", "sum"))
    g["pf_yr"] = g["pf"] / g["days"] * 365 / power_mw / 1000
    g["fc_yr"] = g["fcr"] / g["days"] * 365 / power_mw / 1000
    g["cap"] = g["fcr"] / g["pf"]
    month_rows = "".join(f"<tr><td>{m}</td><td>{r.pf_yr:.1f}</td><td>{r.fc_yr:.1f}</td><td>{r.cap * 100:.1f}%</td></tr>"
                         for m, r in g.iterrows())
    worst = daily.nsmallest(3, "capture_ratio")
    worst_rows = "".join(f"<tr><td>{d}</td><td>{r.revenue_perfect_eur:,.0f}</td><td>{r.revenue_forecast_eur:,.0f}</td>"
                         f"<td>{r.capture_ratio * 100:.1f}%</td></tr>" for d, r in worst.iterrows())
    duration = energy_mwh / power_mw
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>一页评审 · {client}</title>
<style>body{{font:14px/1.6 system-ui,"PingFang SC","Microsoft YaHei",sans-serif;color:#1b1f24;max-width:820px;margin:30px auto;padding:0 20px}}
h1{{font-size:20px;color:#0f2a5a}} h2{{font-size:16px;margin-top:22px}} .k{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
.k div{{border:1px solid #e5e7eb;border-radius:8px;padding:10px}} .k span{{display:block;color:#6b7280;font-size:12px}} .k b{{font-size:22px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid #e5e7eb;padding:5px 8px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.m{{color:#6b7280;font-size:12px}}</style></head><body>
<h1>一页评审 · {client} · {zone}</h1>
<p class="m">口径：{power_mw:g} MW / {energy_mwh:g} MWh（{duration:g} 小时），单向效率 {eta:.0%}，每日最多 {cycles:g} 次循环{'（不限）' if cycles <= 0 else ''}，可用率 {availability:.0%}，退化成本 {cycle_cost:g} €/MWh。
仅日前市场，价格接受者，不含日内、辅助服务、电网费、税。回测区间 {daily.index.min()} 至 {daily.index.max()}，{n} 天，每 7 天重训练。</p>
<div class="k"><div><span>完美预见收益</span><b>{pf_yr:.1f}</b> k€/MW/年</div>
<div><span>关门前预测调度收益</span><b>{fc_yr:.1f}</b> k€/MW/年</div>
<div><span>捕获率（收益加权）</span><b>{cap * 100:.1f}%</b><br><span>每日平均 {cap_d * 100:.1f}%</span></div></div>
<h2>逐月（k€/MW/年折算）</h2><table><tr><th>月份</th><th>完美预见</th><th>预测调度</th><th>捕获率</th></tr>{month_rows}</table>
<h2>捕获率最低的 3 天（EUR）</h2><table><tr><th>日期</th><th>完美预见</th><th>预测调度</th><th>捕获率</th></tr>{worst_rows}</table>
<h2>参照</h2><p class="m">Modo Energy 德国 2h 基准（日前 + 日内 + FCR + aFRR 叠加）2025 年约 240、2026 年 4 月约 218 k€/MW/年。日前套利通常占叠加收益的四到五成；本页只算日前，不能直接与叠加数字相比。
完整报告包含日内与辅助服务叠加、退化与融资情景、15 分钟分辨率影响。</p>
<p class="m">日前实验室 · DayAhead Lab · 数据：Bundesnetzagentur | SMARD.de、energy-charts.info（CC BY 4.0）、ENTSO-E · 仅供研究参考，不构成投资或交易建议 · 逐日记录 dayaheadlab.github.io</p>
</body></html>"""
    out = out or f"outputs/review_{key}_{power_mw:g}mw_{energy_mwh:g}mwh.html"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html, encoding="utf-8")
    rprint(f"perfect {pf_yr:.1f}  forecast {fc_yr:.1f} kEUR/MW/yr  capture {cap * 100:.1f}%  -> {out}")


@app.command("cloud-daily")
def cloud_daily(
    zone: str = typer.Option(None),
    state_dir: str = typer.Option("state", help="CSV state directory in the repo"),
    site_dir: str = typer.Option(".", help="where index.html and data.json are written"),
    backtest_csv: str = typer.Option(None, help="daily backtest results for the benchmark table"),
    model: str = typer.Option("lgbm"),
):
    """Stateless daily run for CI: CSV state in, forecast + site + CSV state out.

    Rebuilds a throwaway DuckDB from ``state/``, refreshes the price series from the public
    API, scores whatever has cleared, issues tomorrow's forecast, regenerates the site and
    writes the state back. Nothing outside the repository is required.
    """
    import tempfile

    from .live import build_site, export_state, import_state, run_daily, score_pending
    from .live.portable import PRICE_SERIES

    s = load_settings()
    zone = zone or s.default_zone
    with tempfile.TemporaryDirectory() as tmp:
        store = TimeSeriesStore(Path(tmp) / "run.duckdb")
        rprint("[bold]import[/bold]", import_state(store, state_dir))

        scored = score_pending(store, s, zone)
        for r in scored:
            rprint(f"  scored {r['target_day']}: MAE {r['mae']:.1f}, rank {r['rank_corr']:.3f}, "
                   f"capture {(r['capture'] or 0) * 100:.1f}% (on_time={r['on_time']})")

        res = run_daily(store, s, zone, model_name=model, update_series=[PRICE_SERIES])
        # escape: the model label contains [...] which rich would otherwise eat as markup
        rprint(f"[bold]forecast {res['target_day'].date()}[/bold] on_time={res['on_time']} "
               f"model={escape(res['model'])} gate={res['gate']}")
        rprint(f"  mean {res['pred'].mean():.1f}  min {res['pred'].min():.1f}  "
               f"max {res['pred'].max():.1f} EUR/MWh")
        if not res["on_time"]:
            rprint("[yellow]warning: issued after gate closure, excluded from the record[/yellow]")

        bt = Path(backtest_csv) if backtest_csv else None
        rprint(f"site -> {build_site(store, s, zone, out_dir=Path(site_dir), backtest_csv=bt)}")
        rprint("[bold]export[/bold]", export_state(store, state_dir, zone))
        store.close()


@app.command("export-state")
def export_state_cmd(zone: str = typer.Option(None), state_dir: str = typer.Option("state")):
    """Dump the daily job's state from the local DuckDB to CSV (to seed the cloud repo)."""
    from .live import export_state

    s = load_settings()
    rprint(export_state(_store(), state_dir, zone or s.default_zone))


@app.command()
def site(zone: str = typer.Option(None)):
    """Rebuild the static track-record site from the store."""
    from .live import build_site

    s = load_settings()
    rprint(f"site written to {build_site(_store(), s, zone or s.default_zone)}")


if __name__ == "__main__":
    app()
