"""
PE factor research:
  - IC vs horizon: 1m, 6m, 12m, 24m forward returns
  - IC vs market regimes: up/down, high/low vol (from index-based monthly regimes)
  - PE variants:
      1) plain: score = -PE
      2) threshold: score = -PE, but if mom <= 0 then assign floor score
      3) industry-neutral: within each industry each month, z-score PE then negate

Inputs (parquet/csv):
  --pe: columns date,symbol,pe  (month-end snapshots)
  --prices: columns date,symbol,close  (month-end closes for all stocks; used for momentum gating)
  --prices-monthstart-open: columns month,date,symbol,open (month-start opens; used for open-to-open forward returns)
  --industry: columns symbol,industry
  --regimes: columns month,mkt_ret,up,vol,high_vol

Output:
  - ic_series.parquet: date, horizon, variant, ic
  - ic_summary.parquet: horizon, variant, regime, mean_ic, std_ic, ic_ir, t_stat, n
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


Variant = Literal["plain", "threshold", "industry_neutral"]


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suf == ".csv":
        return pd.read_csv(p)
    raise SystemExit("Unsupported input format. Use .csv or .parquet")


def _write_any(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    if suf == ".csv":
        df.to_csv(path, index=False)
        return
    raise SystemExit("Unsupported output format. Use .csv or .parquet")


def _to_month_end(d: pd.Series) -> pd.Series:
    return pd.to_datetime(d).dt.to_period("M").dt.to_timestamp("M")


def _prepare_monthly_panels(pe_long: pd.DataFrame, px_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pe = pe_long.copy()
    pe["date"] = pd.to_datetime(pe["date"])
    pe["symbol"] = pe["symbol"].astype(str).str.strip()
    pe["pe"] = pd.to_numeric(pe["pe"], errors="coerce")

    px = px_long.copy()
    px["date"] = pd.to_datetime(px["date"])
    px["symbol"] = px["symbol"].astype(str).str.strip()
    # accept close or adj_close; prefer adjusted close when present (qfq-equivalent)
    price_col = "adj_close" if "adj_close" in px.columns else ("close" if "close" in px.columns else None)
    if price_col is None:
        raise SystemExit("prices file must contain column 'close' or 'adj_close'.")
    px["close"] = pd.to_numeric(px[price_col], errors="coerce")

    pe_w = pe.pivot(index="date", columns="symbol", values="pe").sort_index()
    px_w = px.pivot(index="date", columns="symbol", values="close").sort_index()

    # Align dates and symbols intersection
    common_dates = pe_w.index.intersection(px_w.index)
    common_syms = pe_w.columns.intersection(px_w.columns)
    pe_w = pe_w.reindex(index=common_dates, columns=common_syms)
    px_w = px_w.reindex(index=common_dates, columns=common_syms)
    return pe_w, px_w


def _momentum(px_w: pd.DataFrame, lookback_m: int) -> pd.DataFrame:
    # simple momentum over lookback months
    return px_w / px_w.shift(lookback_m) - 1.0


def _industry_neutral_score(pe_row: pd.Series, industry_map: pd.Series) -> pd.Series:
    """
    Cross-sectional score for one date:
      score = -zscore(PE within industry)
    """
    pe = pd.to_numeric(pe_row, errors="coerce")
    ind = industry_map.reindex(pe.index)
    out = pd.Series(np.nan, index=pe.index, dtype="float64")

    for g, idx in ind.dropna().groupby(ind.dropna()).groups.items():
        vals = pe.loc[list(idx)].astype("float64")
        m = vals.notna() & np.isfinite(vals) & (vals > 0)
        if int(m.sum()) < 3:
            continue
        x = vals[m]
        std = float(x.std(ddof=1))
        if not np.isfinite(std) or std == 0.0:
            continue
        z = (x - float(x.mean())) / std
        out.loc[x.index] = (-z).astype("float64")
    return out


def _compute_ic_series(
    score: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    *,
    min_names: int = 30,
) -> pd.Series:
    ics = []
    for d in score.index:
        s = score.loc[d]
        r = fwd_ret.loc[d]
        m = s.notna() & r.notna() & np.isfinite(s) & np.isfinite(r)
        if int(m.sum()) < int(min_names):
            ics.append(np.nan)
            continue
        x = s[m].astype(float).values
        y = r[m].astype(float).values
        if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
            ics.append(np.nan)
            continue
        ic, _ = spearmanr(x, y)
        ics.append(float(ic) if np.isfinite(ic) else np.nan)
    return pd.Series(ics, index=score.index, name="ic").astype("float64")


def _summarize_ic(x: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(x, errors="coerce").dropna().astype("float64")
    if s.empty:
        return {"mean_ic": float("nan"), "std_ic": float("nan"), "ic_ir": float("nan"), "t_stat": float("nan"), "n": 0.0}
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    ic_ir = float(mean / std) if std and np.isfinite(std) else float("nan")
    t = float(mean / (std / np.sqrt(len(s)))) if std and np.isfinite(std) else float("nan")
    return {"mean_ic": mean, "std_ic": std, "ic_ir": ic_ir, "t_stat": t, "n": float(len(s))}


def _summarize_performance(rets: pd.Series, *, periods_per_year: int = 12) -> dict[str, float]:
    """
    Simple performance summary for periodic returns.
    Assumes rets is indexed by period timestamps (monthly in this project).
    """
    r = pd.to_numeric(rets, errors="coerce").fillna(0.0).astype("float64")
    n = int(len(r))
    if n == 0:
        return {
            "cagr": float("nan"),
            "vol_ann": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "total_return": float("nan"),
            "n_periods": 0.0,
        }
    eq = (1.0 + r).cumprod()
    total = float(eq.iloc[-1] - 1.0)
    years = float(n) / float(periods_per_year)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol_ann = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else float("nan")
    sharpe = float((r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year)) if n > 1 and r.std(ddof=1) != 0 else float("nan")
    dd = (eq / eq.cummax() - 1.0).astype("float64")
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if np.isfinite(cagr) and np.isfinite(max_dd) and max_dd < 0 else float("nan")
    return {
        "cagr": cagr,
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "total_return": total,
        "n_periods": float(n),
    }


def _build_equal_weight_top_quantile(score_row: pd.Series, *, top_quantile: float) -> pd.Series:
    s = pd.to_numeric(score_row, errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.Series(0.0, index=score_row.index, dtype="float64")
    n = int(len(s))
    k = max(1, int(np.floor(n * float(top_quantile))))
    top = s.nlargest(k).index
    w = pd.Series(0.0, index=score_row.index, dtype="float64")
    w.loc[top] = 1.0 / float(k)
    return w


def _simulate_drift_rebalance(
    *,
    open_panel: pd.DataFrame,
    score_panel: pd.DataFrame,
    rebalance_every_m: int,
    top_quantile: float,
    cost_bps: float,
    max_stale_months: int,
    vol_target_ann: float | None = None,
    vol_lookback_m: int = 12,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """
    Drift-aware long-only backtest using month-start opens.

    Execution model:
      - signal/score observed at month-end t (score_panel index uses month-end timestamps)
      - execute at next month open, which is represented by open_panel at month-end (t+1)
      - returns are open-to-open (month start to next month start)
      - rebalance every `rebalance_every_m` months (execute schedule), equal-weight top quantile
      - anti-zombie rule: if a currently-held name is non-tradable for more than
        `max_stale_months` consecutive months, force-drop it to cash.
      - optional vol targeting: scale total exposure at rebalance to target annualized vol.

    Inputs:
      - open_panel: index=month_end timestamps, values=open at month start of that month
      - score_panel: index=month_end timestamps, cross-sectional score at that month end
    """
    idx = open_panel.index
    cols = open_panel.columns
    if not idx.equals(score_panel.index):
        score_panel = score_panel.reindex(index=idx, columns=cols)
    else:
        score_panel = score_panel.reindex(columns=cols)

    # monthly open-to-open returns aligned at current month (dt -> next dt)
    r1m = (open_panel.shift(-1) / open_panel - 1.0).replace([np.inf, -np.inf], np.nan).astype("float64")

    cost_rate = float(cost_bps) / 10000.0 if cost_bps and cost_bps > 0 else 0.0

    # execution dates are every h months starting at idx[1] (first date with prior score)
    exec_pos = list(range(1, len(idx), int(rebalance_every_m)))
    exec_dates = set(pd.DatetimeIndex([idx[p] for p in exec_pos if p < len(idx)]))

    w = pd.Series(0.0, index=cols, dtype="float64")  # weights AFTER trade at current open
    stale_months = pd.Series(0, index=cols, dtype="int64")  # consecutive non-tradable months while held
    equity = 1.0
    ret_hist: list[float] = []

    out_rows = []
    for t in range(0, len(idx) - 1):
        dt = pd.Timestamp(idx[t])
        tradable_now = open_panel.loc[dt].notna()

        # drift to current open using previous month's realized return (t-1 -> t)
        if t > 0:
            r_prev = r1m.iloc[t - 1].copy()
            # If return is unobservable for a name (e.g., suspended on month-start), assume 0% for that month.
            r_prev = r_prev.fillna(0.0)
            gross_prev = float((w * r_prev).sum())
            value_after = 1.0 + gross_prev
            if np.isfinite(value_after) and value_after > 0:
                w = (w * (1.0 + r_prev)) / value_after
            else:
                w = pd.Series(0.0, index=cols, dtype="float64")

        # Track stale non-tradable streak for currently held names.
        held = w.abs() > 0.0
        stale_months = stale_months.where(~held, other=0)
        stale_months = stale_months.where(~(held & ~tradable_now), other=stale_months + 1)
        stale_months = stale_months.where(~(held & tradable_now), other=0)

        # Anti-zombie: if held and stale too long, force-drop to cash (no explicit trade cost).
        force_drop = held & (stale_months > int(max_stale_months))
        n_forced = int(force_drop.sum())
        if n_forced > 0:
            w = w.where(~force_drop, other=0.0)
            stale_months = stale_months.where(~force_drop, other=0)

        # rebalance at dt (current open) if scheduled
        to = 0.0
        cost = 0.0
        lev = float(w.sum())
        vol_est_ann = float("nan")
        scale = 1.0
        if dt in exec_dates:
            score_date = pd.Timestamp(idx[t - 1])  # previous month-end
            srow = score_panel.loc[score_date].copy()
            # tradable = has open at execution (if no open, assume cannot trade at that month-start)
            tradable = tradable_now

            # Vol targeting (optional): compute realized vol from past net returns, then scale exposure.
            if vol_target_ann is not None and float(vol_target_ann) > 0:
                lb = int(vol_lookback_m)
                if lb < 2:
                    lb = 2
                hist = np.asarray(ret_hist[-lb:], dtype="float64")
                if hist.size >= 2:
                    vol_m = float(np.nanstd(hist, ddof=1))
                    vol_est_ann = float(vol_m * np.sqrt(12.0)) if np.isfinite(vol_m) else float("nan")
                    if np.isfinite(vol_est_ann) and vol_est_ann > 0:
                        scale = float(float(vol_target_ann) / vol_est_ann)
                # clip leverage scale
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
                scale = float(min(scale, float(max_leverage)))

            # Build target weights with execution constraints:
            # - untradable names: cannot change, must carry
            # - tradable names: we can set to match the desired total exposure
            w_untradable = w.where(~tradable, other=0.0).astype("float64")
            fixed_sum = float(w_untradable.sum())

            desired_total = float(scale) if (vol_target_ann is not None and float(vol_target_ann) > 0) else 1.0
            desired_total = float(min(desired_total, float(max_leverage)))
            if desired_total < 0:
                desired_total = 0.0

            # If untradable exposure already exceeds desired, sell all tradable to minimize risk.
            if not np.isfinite(fixed_sum) or fixed_sum >= desired_total:
                w_tradable_final = pd.Series(0.0, index=cols, dtype="float64")
                w_target = (w_untradable + w_tradable_final).astype("float64")
            else:
                budget = float(desired_total - fixed_sum)
                # allocate budget to top-quantile among tradable names
                srow_t = srow.where(tradable)
                w_tradable_unit = _build_equal_weight_top_quantile(srow_t, top_quantile=top_quantile)
                # if no tradable names selected, stay with only untradable
                if float(w_tradable_unit.sum()) <= 0:
                    w_target = w_untradable.astype("float64")
                else:
                    w_target = (w_untradable + w_tradable_unit * budget).astype("float64")

            cash_before = 1.0 - float(w.sum())
            cash_target = 1.0 - float(w_target.sum())
            to = 0.5 * (float((w_target - w).abs().sum()) + abs(cash_target - cash_before))
            cost = float(to * cost_rate)
            w = w_target
            lev = float(w.sum())

        # realize return over next month (dt -> dt_next)
        r_cur = r1m.iloc[t].fillna(0.0)
        gross = float((w * r_cur).sum())
        net = gross - cost
        equity = float(equity * (1.0 + net))
        ret_hist.append(float(net))

        untradable_exposure = float(w.where(~tradable_now, other=0.0).sum())
        leverage_now = float(w.sum())
        untradable_share = float(untradable_exposure / leverage_now) if leverage_now > 0 else 0.0

        out_rows.append(
            {
                "date": dt,
                "return": float(net),
                "equity": float(equity),
                "turnover": float(to),
                "n_holdings": int((w.abs() > 0).sum()),
                "n_forced_drops": n_forced,
                "leverage": float(lev),
                "vol_est_ann": float(vol_est_ann),
                "vol_scale": float(scale),
                "untradable_exposure": float(untradable_exposure),
                "untradable_share": float(untradable_share),
            }
        )

    return pd.DataFrame(out_rows)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PE factor IC research.")
    p.add_argument("--pe", type=str, required=True, help="Month-end PE snapshots: date,symbol,pe")
    p.add_argument("--prices", type=str, required=True, help="Month-end prices: date,symbol,close (all stocks)")
    p.add_argument(
        "--prices-monthstart-open",
        type=str,
        required=True,
        help="Month-start open snapshots: month,date,symbol,open. Forward returns use open-to-open.",
    )
    p.add_argument("--industry", type=str, required=True, help="Stock industry mapping: symbol,industry")
    p.add_argument("--regimes", type=str, required=True, help="Monthly regimes: month,up,high_vol,...")
    p.add_argument("--out-dir", type=str, default="reports/pe_research")
    p.add_argument("--min-names", type=int, default=30, help="Minimum cross-sectional names per month to compute IC")
    p.add_argument("--floor", type=float, default=-1e12, help="Floor score for ineligible names (threshold variant)")
    p.add_argument("--mom-lookback-m", type=int, default=6, help="Momentum lookback in months for threshold PE")
    p.add_argument("--horizons", type=str, default="1,6,12,24", help="Comma-separated forward horizons in months")
    p.add_argument(
        "--max-stale-months",
        type=int,
        default=3,
        help="Force-drop held names to cash after this many consecutive non-tradable months (anti-zombie).",
    )
    p.add_argument("--vol-target-ann", type=float, default=0.0, help="Annualized volatility target (0 disables).")
    p.add_argument("--vol-lookback-m", type=int, default=12, help="Lookback months for realized-vol estimate.")
    p.add_argument("--max-leverage", type=float, default=2.0, help="Max leverage when vol targeting is enabled.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pe_long = _read_any(args.pe)
    px_long = _read_any(args.prices)
    ind_df = _read_any(args.industry)
    reg_df = _read_any(args.regimes)

    pe_w, px_w = _prepare_monthly_panels(pe_long, px_long)
    # Month-start open panel for open-to-open forward returns (required)
    mo = _read_any(args.prices_monthstart_open)
    if not {"month", "symbol", "open"}.issubset(set(mo.columns)):
        raise SystemExit("--prices-monthstart-open must contain columns: month,symbol,open (date optional).")
    mo = mo.copy()
    mo["month"] = mo["month"].astype(str)
    mo["symbol"] = mo["symbol"].astype(str).str.strip()
    mo["open"] = pd.to_numeric(mo["open"], errors="coerce")
    # Map month -> month-start open, index it by month-end timestamps (same as pe_w/px_w index)
    mo["month_end"] = pd.PeriodIndex(mo["month"], freq="M").to_timestamp("M")
    open_w = mo.pivot(index="month_end", columns="symbol", values="open").sort_index()
    # align to pe/px panels intersection
    open_w = open_w.reindex(index=pe_w.index, columns=pe_w.columns)

    # industry map
    ind_df = ind_df.rename(columns={"ts_code": "symbol"}).copy()
    ind_df["symbol"] = ind_df["symbol"].astype(str).str.strip()
    ind_df["industry"] = ind_df["industry"].astype(str).str.strip()
    industry_map = ind_df.drop_duplicates("symbol").set_index("symbol")["industry"]

    # regimes by month
    reg = reg_df.copy()
    if "month" not in reg.columns:
        raise SystemExit("regimes file must contain 'month' column (YYYY-MM).")
    reg["month"] = reg["month"].astype(str)
    reg = reg.set_index("month")

    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    floor = float(args.floor)
    min_names = int(args.min_names)

    # Precompute momentum for threshold variant
    mom = _momentum(px_w, lookback_m=int(args.mom_lookback_m))

    results_series = []
    results_summary = []

    for h in horizons:
        # Forward returns for IC MUST align with the execution model:
        #   score observed at month-end t, execute at next month open (t+1 open),
        #   so forward return horizon h starts at (t+1 open) and ends at (t+1+h open).
        # open_w is indexed by month-end timestamps but contains the month-start open of that month.
        # Therefore, for score date t:
        #   start = open_w.shift(-1).loc[t]
        #   end   = open_w.shift(-(h+1)).loc[t]
        fwd = open_w.shift(-(h + 1)) / open_w.shift(-1) - 1.0

        # build scores per variant
        scores: dict[Variant, pd.DataFrame] = {}
        # Plain PE: prefer lower PE, but treat invalid PE as the lowest score for comparability.
        valid_pe = pe_w.notna() & (pe_w > 0) & np.isfinite(pe_w)
        scores["plain"] = (-pe_w).where(valid_pe, other=floor).astype("float64")

        eligible = (mom > 0) & pe_w.notna() & (pe_w > 0) & np.isfinite(pe_w)
        scores["threshold"] = (-pe_w).where(eligible, other=floor).astype("float64")

        ind_scores = []
        for d in pe_w.index:
            ind_scores.append(_industry_neutral_score(pe_w.loc[d], industry_map).rename(d))
        scores["industry_neutral"] = pd.DataFrame(ind_scores)
        scores["industry_neutral"].index = pe_w.index
        scores["industry_neutral"] = scores["industry_neutral"].reindex(columns=pe_w.columns).fillna(floor).astype("float64")

        for variant, sc in scores.items():
            ic = _compute_ic_series(sc, fwd, min_names=min_names)

            # attach to long series output
            tmp = pd.DataFrame({"date": ic.index, "horizon_m": h, "variant": variant, "ic": ic.values})
            results_series.append(tmp)

            # Backtest for this (h, variant): long top 20%, rebalance every h months, drift-aware.
            bt_curve = _simulate_drift_rebalance(
                open_panel=open_w,
                score_panel=sc,
                rebalance_every_m=int(h),
                top_quantile=0.2,
                cost_bps=10.0,
                max_stale_months=int(args.max_stale_months),
                vol_target_ann=(float(args.vol_target_ann) if float(args.vol_target_ann) > 0 else None),
                vol_lookback_m=int(args.vol_lookback_m),
                max_leverage=float(args.max_leverage),
            )
            bt_dir = out_dir / "backtests" / f"h{int(h)}m" / str(variant)
            _write_any(bt_curve, bt_dir / "equity_curve.parquet")
            perf = _summarize_performance(pd.Series(bt_curve["return"].values, index=pd.to_datetime(bt_curve["date"])))
            (bt_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "horizon_m": int(h),
                        "variant": str(variant),
                        "top_quantile": 0.2,
                        "cost_bps": 10.0,
                        "max_stale_months": int(args.max_stale_months),
                        "periods_per_year": 12,
                        **perf,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            # summaries overall + by regimes
            summ_all = _summarize_ic(ic)
            results_summary.append(
                {
                    "horizon_m": h,
                    "variant": variant,
                    "regime": "all",
                    **summ_all,
                }
            )

            # by market states
            months = ic.index.to_period("M").astype(str)
            ic_by_month = pd.Series(ic.values, index=months)
            joined = pd.DataFrame({"ic": ic_by_month}).join(reg[["up", "high_vol"]], how="left")

            def _summ(name: str, mask: pd.Series) -> None:
                s = joined.loc[mask, "ic"]
                results_summary.append({"horizon_m": h, "variant": variant, "regime": name, **_summarize_ic(s)})

            _summ("up", joined["up"] == True)  # noqa: E712
            _summ("down", joined["up"] == False)  # noqa: E712
            _summ("high_vol", joined["high_vol"] == True)  # noqa: E712
            _summ("low_vol", joined["high_vol"] == False)  # noqa: E712
            _summ("up_high_vol", (joined["up"] == True) & (joined["high_vol"] == True))  # noqa: E712
            _summ("up_low_vol", (joined["up"] == True) & (joined["high_vol"] == False))  # noqa: E712
            _summ("down_high_vol", (joined["up"] == False) & (joined["high_vol"] == True))  # noqa: E712
            _summ("down_low_vol", (joined["up"] == False) & (joined["high_vol"] == False))  # noqa: E712

    ic_series = pd.concat(results_series, ignore_index=True)
    ic_summary = pd.DataFrame(results_summary)

    _write_any(ic_series, out_dir / "ic_series.parquet")
    _write_any(ic_summary, out_dir / "ic_summary.parquet")

    print(f"Wrote: {out_dir / 'ic_series.parquet'} rows={len(ic_series):,}")
    print(f"Wrote: {out_dir / 'ic_summary.parquet'} rows={len(ic_summary):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

