"""Static site generator for the public track record (GitHub Pages friendly, no JS deps)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, Settings
from ..data.store import TimeSeriesStore


def _svg_line_chart(x_labels: list[str], series: dict[str, list[float]], width=860, height=300,
                    colors=("#1f5fbf", "#c0392b", "#7f8c8d"), y_unit="EUR/MWh",
                    bars: list[float] | None = None) -> str:
    """Tiny dependency-free SVG chart: lines for price series, optional bars for dispatch."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 16, 34
    W, H = width - pad_l - pad_r, height - pad_t - pad_b
    all_vals = [v for s in series.values() for v in s if v is not None and not np.isnan(v)]
    if not all_vals:
        return "<p>no data</p>"
    lo, hi = min(all_vals + [0]), max(all_vals)
    if hi == lo:
        hi = lo + 1
    n = max(len(v) for v in series.values())
    xs = lambda i: pad_l + (i / max(n - 1, 1)) * W  # noqa: E731
    ys = lambda v: pad_t + (1 - (v - lo) / (hi - lo)) * H  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
           f'style="width:100%;height:auto;font-family:system-ui,sans-serif;font-size:12px">']
    # grid + y labels
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = ys(v)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
                   f'stroke="#ddd" stroke-width="1"/>')
        out.append(f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#666">{v:.0f}</text>')
    out.append(f'<text x="{pad_l - 6}" y="{pad_t - 4}" text-anchor="end" fill="#666">{y_unit}</text>')
    # bars (dispatch): positive = discharge, negative = charge, scaled to ±H/3
    if bars:
        bmax = max(abs(b) for b in bars) or 1
        zero_y = ys(0) if lo <= 0 <= hi else pad_t + H
        bw = W / n * 0.8
        for i, b in enumerate(bars):
            if abs(b) < 1e-9:
                continue
            h = abs(b) / bmax * (H / 3)
            y = zero_y - h if b > 0 else zero_y
            color = "#27ae60" if b > 0 else "#e67e22"
            out.append(f'<rect x="{xs(i) - bw / 2:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                       f'fill="{color}" opacity="0.35"/>')
    for (name, vals), color in zip(series.items(), colors):
        pts = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(vals)
                       if v is not None and not np.isnan(v))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    # x labels
    step = max(1, n // 12)
    for i in range(0, n, step):
        out.append(f'<text x="{xs(i):.1f}" y="{height - 12}" text-anchor="middle" fill="#666">'
                   f'{x_labels[i]}</text>')
    # legend
    lx = pad_l
    for (name, _), color in zip(series.items(), colors):
        out.append(f'<rect x="{lx}" y="{height - 8}" width="12" height="3" fill="{color}"/>')
        out.append(f'<text x="{lx + 16}" y="{height - 4}" fill="#444">{name}</text>')
        lx += 16 + 8 * len(name) + 20
    out.append("</svg>")
    return "\n".join(out)


def _fmt(v, nd=1, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:,.{nd}f}{suffix}"


def build_site(store: TimeSeriesStore, settings: Settings, zone: str,
               out_dir: Path | None = None, backtest_csv: Path | None = None) -> Path:
    tz = settings.timezone
    out_dir = out_dir or PROJECT_ROOT / "site"
    out_dir.mkdir(parents=True, exist_ok=True)
    brand = settings.brand
    b = settings.battery

    # ---- latest forecast ----------------------------------------------------------------
    days = store.forecast_days(zone)
    latest_day = days[-1] if days else None
    fc_html, fc_meta = "<p>尚无预测。</p>", {}
    if latest_day is not None:
        fc = store.read_forecast(zone, latest_day)
        local = fc["ts_utc"].dt.tz_convert(tz)
        labels = [t.strftime("%H:%M") for t in local]
        actual = store.read_wide(zone, ["price.day_ahead"], start=pd.Timestamp(latest_day, tz=tz),
                                 end=pd.Timestamp(latest_day, tz=tz) + pd.Timedelta(days=1), freq="1h")
        series = {"预测 forecast": fc["pred"].round(2).tolist()}
        if not actual.empty:
            act = actual["price.day_ahead"].reindex(fc["ts_utc"]).round(2).tolist()
            if any(not np.isnan(v) for v in act):
                series["实际 actual"] = act
        from dataclasses import replace

        from ..strategies.battery import BatteryParams, optimize_dispatch
        params = replace(BatteryParams(**b.model_dump()), soc_final_min_mwh=b.soc_initial_mwh)
        disp = optimize_dispatch(fc["pred"].to_numpy(), 1.0, params)
        bars = (disp["discharge_mw"] - disp["charge_mw"]).round(3).tolist()
        fc_html = _svg_line_chart(labels, series, bars=bars)
        issued = fc["issued_at"].iloc[0].tz_convert(tz)
        fc_meta = {
            "target_day": str(latest_day), "issued_at_local": issued.strftime("%Y-%m-%d %H:%M %Z"),
            "on_time": bool(fc["on_time"].iloc[0]), "model": fc["model"].iloc[0],
            "mean": float(fc["pred"].mean()), "min": float(fc["pred"].min()),
            "max": float(fc["pred"].max()),
            "planned_revenue": float(disp["revenue_eur"].sum()),
        }

    # ---- track record --------------------------------------------------------------------
    scores = store.read_scores(zone)
    if not scores.empty:
        scores["target_day"] = pd.to_datetime(scores["target_day"]).dt.date
        scores = scores.sort_values("target_day", ascending=False)
    on_time = scores[scores["on_time"]] if not scores.empty else scores
    summary = {}
    if not on_time.empty:
        rev_fc, rev_pf = float(on_time["rev_forecast"].sum()), float(on_time["rev_perfect"].sum())
        summary = {
            "days": int(len(on_time)), "mae": float(on_time["mae"].mean()),
            "rank_corr": float(on_time["rank_corr"].mean()),
            "capture": rev_fc / rev_pf if rev_pf > 0 else float("nan"),  # revenue-weighted
            "capture_daily_mean": float(on_time["capture"].mean()),
            "rev_forecast": rev_fc, "rev_perfect": rev_pf,
        }
    # The table IS the record, so it lists on-time forecasts only. Late ones are reported as a
    # count below it rather than filling the table with rows that do not count.
    n_late = int(len(scores) - len(on_time)) if not scores.empty else 0
    rows_html = "".join(
        f"<tr><td>{r.target_day}</td><td>{r.model if hasattr(r, 'model') else ''}</td>"
        f"<td>{_fmt(r.mae)}</td>"
        f"<td>{_fmt(r.rank_corr, 3)}</td><td>{_fmt(r.rev_forecast, 0)}</td><td>{_fmt(r.rev_perfect, 0)}</td>"
        f"<td>{_fmt(r.capture * 100 if r.capture is not None else None, 1, '%')}</td></tr>"
        for r in on_time.head(60).itertuples()) if not on_time.empty else \
        "<tr><td colspan=7>首条关门前发布的预测将在次日出清后自动打分并出现在这里。</td></tr>"
    late_note = (f"<p class='muted'>另有 {n_late} 条关门后才发出的预测，已记录但不计入本表和上方统计。</p>"
                 if n_late else "")

    # ---- backtest benchmark (monthly) ----------------------------------------------------
    bt_html = ""
    backtest_csv = backtest_csv or PROJECT_ROOT / "outputs" / f"{zone.lower().replace('-', '_')}_lgbm.csv"
    if Path(backtest_csv).exists():
        bt = pd.read_csv(backtest_csv, parse_dates=["day"])
        bt["month"] = bt["day"].dt.to_period("M").astype(str)
        g = bt.groupby("month").agg(rev_fc=("revenue_forecast_eur", "sum"),
                                    rev_pf=("revenue_perfect_eur", "sum"),
                                    spread=("da_spread_eur", "mean"), days=("day", "count"))
        g["capture"] = g["rev_fc"] / g["rev_pf"]
        # annualised EUR per MW of power, the unit the industry quotes (Modo, Aurora, Pexapark)
        g["pf_per_mw_yr"] = g["rev_pf"] / g["days"] * 365 / b.power_mw
        g["fc_per_mw_yr"] = g["rev_fc"] / g["days"] * 365 / b.power_mw
        bt_rows = "".join(
            f"<tr><td>{m}</td><td>{int(r.days)}</td><td>{_fmt(r.spread, 0)}</td>"
            f"<td>{_fmt(r.pf_per_mw_yr / 1000, 1)}</td><td>{_fmt(r.fc_per_mw_yr / 1000, 1)}</td>"
            f"<td>{_fmt(r.capture * 100, 1, '%')}</td></tr>"
            for m, r in g.sort_index(ascending=False).iterrows())
        total_days = int(g["days"].sum())
        pf_yr = float(g["rev_pf"].sum() / total_days * 365 / b.power_mw / 1000)
        fc_yr = float(g["rev_fc"].sum() / total_days * 365 / b.power_mw / 1000)
        cap_w = float(g["rev_fc"].sum() / g["rev_pf"].sum() * 100)
        cap_d = float(bt["capture_ratio"].mean() * 100)
        cyc = f"每日最多 {b.max_cycles_per_day:g} 次循环" if b.max_cycles_per_day else "不限循环"
        unc_txt = ""
        unc_csv = Path(backtest_csv).with_name(Path(backtest_csv).stem + "_unconstrained.csv")
        if unc_csv.exists():
            u = pd.read_csv(unc_csv)
            u_pf = float(u["revenue_perfect_eur"].sum() / len(u) * 365 / b.power_mw / 1000)
            u_fc = float(u["revenue_forecast_eur"].sum() / len(u) * 365 / b.power_mw / 1000)
            unc_txt = f"对照口径（不限循环、可用率 100%）：完美预见 {u_pf:.1f}，预测调度 {u_fc:.1f} k€/MW/年。"
        bt_html = f"""
<h2>历史回测基准 · Walk-forward backtest</h2>
<p class="muted">同一模型在 {bt['day'].min().date()} 至 {bt['day'].max().date()} 的逐日滚动回测（每 7 天重训练）。
保守口径：{b.power_mw:g} MW / {b.energy_mwh:g} MWh，{cyc}，可用率 {b.availability:.0%}，仅日前市场。
整段年化：完美预见 {pf_yr:.1f} k€/MW/年，预测调度 {fc_yr:.1f} k€/MW/年，
捕获率 {cap_w:.1f}%（收益加权）／{cap_d:.1f}%（每日平均，等权重计入低收益日，更保守）。{unc_txt}
参照：Modo Energy 德国 2h 基准（日前 + 日内 + FCR + aFRR 叠加）2025 年约 240、2026 年 4 月约 218 k€/MW/年。
这是回测，不是实盘记录，与上表分开统计。</p>
<div class="scroll"><table>
<tr><th>月份</th><th>天数</th><th>日均峰谷价差 (€/MWh)</th><th>完美预见 (k€/MW/年)</th><th>预测调度 (k€/MW/年)</th><th>捕获率</th></tr>
{bt_rows}
</table></div>"""

    generated = pd.Timestamp.now(tz=tz).strftime("%Y-%m-%d %H:%M %Z")
    ontime_txt = "是（关门前发布）" if fc_meta.get("on_time") else "否（关门后发布，不计入记录）"
    summary_html = (
        f"<div class='kpis'>"
        f"<div><span>已打分天数</span><b>{summary['days']}</b></div>"
        f"<div><span>平均 MAE</span><b>{_fmt(summary['mae'])}</b></div>"
        f"<div><span>日内排序相关性</span><b>{_fmt(summary['rank_corr'], 3)}</b></div>"
        f"<div><span>捕获率（收益加权）</span><b>{_fmt(summary['capture'] * 100, 1, '%')}</b></div>"
        f"<div><span>捕获率（每日平均）</span><b>{_fmt(summary['capture_daily_mean'] * 100, 1, '%')}</b></div>"
        f"<div><span>累计收益 / 完美预见</span><b>{_fmt(summary['rev_forecast'], 0)} / {_fmt(summary['rev_perfect'], 0)}</b></div>"
        f"</div>") if summary else "<p class='muted'>记录从第一个被打分的预测开始累计。</p>"

    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{brand['name_zh']} · {brand['name_en']} · {zone}</title>
<style>
:root{{--fg:#1b1f24;--muted:#6b7280;--bg:#fafaf9;--card:#fff;--line:#e5e7eb;--acc:#1f5fbf}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}}
header{{background:#0f2a5a;color:#fff;padding:22px 20px}} header h1{{margin:0;font-size:22px}} header p{{margin:4px 0 0;opacity:.8}}
main{{max-width:960px;margin:0 auto;padding:20px}}
h2{{font-size:18px;margin:28px 0 8px}} .muted{{color:var(--muted);font-size:13px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}
.kpis div{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}}
.kpis span{{display:block;color:var(--muted);font-size:12px}} .kpis b{{font-size:20px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} .scroll{{overflow-x:auto}}
dl{{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;font-size:13px}} dt{{color:var(--muted)}}
footer{{color:var(--muted);font-size:12px;padding:30px 20px;text-align:center}}
</style></head><body>
<header><h1>{brand['name_zh']} · {brand['name_en']}</h1>
<p>{zone} 日前电价预测公开记录 · 每日关门前发布，出清后自动打分 · Public day-ahead forecast track record</p></header>
<main>
<h2>次日预测 · {fc_meta.get('target_day', '—')}</h2>
<div class="card">
{fc_html}
<dl>
<dt>发布时间</dt><dd>{fc_meta.get('issued_at_local', '—')}</dd>
<dt>关门前发布</dt><dd>{ontime_txt}</dd>
<dt>预测均价 / 最低 / 最高</dt><dd>{_fmt(fc_meta.get('mean'))} / {_fmt(fc_meta.get('min'))} / {_fmt(fc_meta.get('max'))} EUR/MWh</dd>
<dt>储能计划收益</dt><dd>{_fmt(fc_meta.get('planned_revenue'), 0)} EUR（{b.power_mw:g} MW / {b.energy_mwh:g} MWh，绿色放电、橙色充电）</dd>
<dt>模型</dt><dd>{fc_meta.get('model', '—')}，每日用全部历史重训练</dd>
</dl></div>

<h2>实盘记录 · Live track record</h2>
<p class="muted">只列关门前（柏林 12:00）发布、且交割日已出清并打分的预测。储能收益按下方保守口径计算。</p>
{summary_html}
<div class="scroll card"><table>
<tr><th>交割日</th><th>模型</th><th>MAE</th><th>排序相关性</th><th>预测调度收益</th><th>完美预见收益</th><th>捕获率</th></tr>
{rows_html}
</table>
{late_note}</div>
{bt_html}

<h2>方法 · Method</h2>
<div class="card">
<p>数据：ENTSO-E / Energy-Charts 公开数据（日前价格、负荷与风光日前预测），CC BY 4.0，来源 Bundesnetzagentur | SMARD.de 与 energy-charts.info。
模型：梯度提升树，特征严格限制在 D-1 12:00 前可得的信息。
储能收益按 {b.power_mw:g} MW / {b.energy_mwh:g} MWh、充放效率 {b.eta_charge:.0%}、循环成本 {b.cycle_cost_eur_per_mwh:g} EUR/MWh、
{('每日最多 ' + format(b.max_cycles_per_day, 'g') + ' 次循环、') if b.max_cycles_per_day else ''}可用率 {b.availability:.0%} 的线性规划调度计算。
价格接受者假设，仅日前市场（不含日内、FCR、aFRR），不含电网费与平衡结算。2025-10 起日前市场为 15 分钟分辨率，本页按小时均价计算，会低估部分价差。</p>
<p><b>特征口径</b>：不使用 ENTSO-E 的 D-1 风光预测。实测（2026-09-13 13:16 柏林）显示该数据在 12:00
日前关门时刻尚未发布，用它属于泄漏。若使用，回测捕获率会从 91.4% 虚增到 96.2%。
每日实盘任务会实测各序列可用性，把关门时刻拿不到的从训练和预测两侧同时剔除，
实际使用的特征集记在上表的模型名里。</p>
<p><b>记录起点</b>：正式发布之前的调试运行（均为关门后发布）未纳入本记录。
记录从第一条关门前发布的预测开始累计，此后每一天都在，包括错得离谱的日子。</p>
<p>捕获率 = 按预测调度的收益 ÷ 事后完美预见的收益。</p><p>它衡量预测对储能套利的实际价值，比 MAE 更有意义。
两种算法都列出：<b>收益加权</b>是区间内总收益之比，回答"可赚的钱captured了多少"；<b>每日平均</b>是逐日比值的算术平均，
把低收益日与高收益日等权重看待，数值更低也更保守。引用时请注明用的是哪一种。</p>
<p class="muted">仅供研究参考，不构成投资或交易建议。Research only; not trading advice.</p>
</div>
</main>
<footer>{brand['name_zh']} · 生成于 {generated}</footer>
</body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    data = {"zone": zone, "generated": generated, "latest": fc_meta, "summary": summary,
            "scores": scores.assign(target_day=scores["target_day"].astype(str),
                                    issued_at=scores["issued_at"].astype(str),
                                    scored_at=scores["scored_at"].astype(str)).to_dict("records")
            if not scores.empty else []}
    (out_dir / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str),
                                       encoding="utf-8")
    return out_dir / "index.html"
