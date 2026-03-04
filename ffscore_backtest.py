from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_engine import performance_stats, run_weight_execution_engine


def _read_parquet_folder(path: Path, pattern: str) -> pd.DataFrame:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found under {path} with pattern={pattern}")
    frames: list[pd.DataFrame] = []
    for fp in files:
        df = pd.read_parquet(fp)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError(f"All parquet files are empty under {path} with pattern={pattern}")
    return pd.concat(frames, ignore_index=True)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num.astype("float64") / den.astype("float64")
    return out.replace([np.inf, -np.inf], np.nan)


def _to_month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp("M")


def _asof_join_monthly(
    monthly_dates: pd.DatetimeIndex,
    stock_rows: pd.DataFrame,
    cols: Iterable[str],
) -> pd.DataFrame:
    if stock_rows.empty:
        return pd.DataFrame(columns=["month_end", "ts_code", *cols])
    left = pd.DataFrame({"month_end": monthly_dates}).sort_values("month_end")
    right = stock_rows.sort_values("ann_date").copy()
    out = pd.merge_asof(
        left,
        right[["ann_date", *cols]],
        left_on="month_end",
        right_on="ann_date",
        direction="backward",
        allow_exact_matches=True,
    )
    out["ts_code"] = str(stock_rows["ts_code"].iloc[0])
    return out.drop(columns=["ann_date"])


@dataclass
class BacktestConfig:
    data_dir: Path
    out_dir: Path
    start: str
    end: str
    top_quantile: float
    min_tradable_days: int
    min_score_names: int
    cost_bps: float
    max_vol_lag: int
    use_pb_filter: bool
    pb_quantile: float
    pb_proxy_path: Path | None
    vol_target_ann: float
    vol_lookback_m: int
    max_leverage: float
    ffscore_carry_months: int
    pb_carry_months: int
    ffscore_full_score_only: bool
    rebalance_freq: str


def load_monthly_market_data(data_dir: Path, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = _read_parquet_folder(data_dir / "daily_qfq", "daily_qfq_*.parquet")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"].astype(str))
    daily = daily.sort_values(["ts_code", "trade_date"])
    daily = daily[(daily["trade_date"] >= pd.Timestamp(start)) & (daily["trade_date"] <= pd.Timestamp(end))]
    if daily.empty:
        raise RuntimeError("No daily rows in requested date range.")

    px_col = "close_qfq" if "close_qfq" in daily.columns else "close"
    op_col = "open_qfq" if "open_qfq" in daily.columns else "open"
    hi_col = "high_qfq" if "high_qfq" in daily.columns else "high"
    lo_col = "low_qfq" if "low_qfq" in daily.columns else "low"
    for c in [px_col, op_col, hi_col, lo_col]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")

    daily["month_end"] = _to_month_end(daily["trade_date"])
    g = daily.groupby(["ts_code", "month_end"], sort=True)

    monthly = g.agg(
        close_m=(px_col, "last"),
        open_m=(op_col, "first"),
        high_m=(hi_col, "max"),
        low_m=(lo_col, "min"),
        tradable_days=("trade_date", "nunique"),
    ).reset_index()

    monthly["ret_1m"] = (
        monthly.sort_values(["ts_code", "month_end"])
        .groupby("ts_code", sort=False)["close_m"]
        .pct_change()
    )
    monthly["next_ret_1m"] = (
        monthly.sort_values(["ts_code", "month_end"])
        .groupby("ts_code", sort=False)["ret_1m"]
        .shift(-1)
    )
    mkt = monthly.sort_values(["month_end", "ts_code"]).reset_index(drop=True)
    return mkt, daily


def load_fundamental_snapshot(data_dir: Path) -> pd.DataFrame:
    bs = pd.read_parquet(data_dir / "fundamental" / "balancesheet.parquet")
    inc = pd.read_parquet(data_dir / "fundamental" / "income.parquet")
    fi = pd.read_parquet(data_dir / "fundamental" / "fina_indicator.parquet")

    for df in [bs, inc, fi]:
        for c in ["ts_code", "ann_date", "end_date"]:
            if c not in df.columns:
                raise RuntimeError(f"Missing required fundamental column: {c}")
        df["ann_date"] = pd.to_datetime(df["ann_date"].astype(str), errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"].astype(str), errors="coerce")
        df.dropna(subset=["ts_code", "ann_date", "end_date"], inplace=True)

    keep_bs = ["ts_code", "ann_date", "end_date", "total_assets", "total_cur_assets", "total_nca", "total_ncl"]
    keep_inc = ["ts_code", "ann_date", "end_date", "revenue"]
    keep_fi = ["ts_code", "ann_date", "end_date", "roe"]

    bs = bs[[c for c in keep_bs if c in bs.columns]].copy()
    inc = inc[[c for c in keep_inc if c in inc.columns]].copy()
    fi = fi[[c for c in keep_fi if c in fi.columns]].copy()

    base = bs.merge(inc, on=["ts_code", "ann_date", "end_date"], how="outer")
    base = base.merge(fi, on=["ts_code", "ann_date", "end_date"], how="outer")
    base = base.sort_values(["ts_code", "end_date", "ann_date"]).drop_duplicates(
        ["ts_code", "end_date"], keep="last"
    )

    for c in ["roe", "revenue", "total_assets", "total_cur_assets", "total_nca", "total_ncl"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    # Quarter-aligned lag mapping avoids date-mismatch bugs (e.g. 03-30 vs 03-31).
    base["q"] = base["end_date"].dt.to_period("Q")
    base["q_prev_y"] = base["q"] - 4
    base["q_lag1q"] = base["q"] - 1
    base["q_prev_y_lag1q"] = base["q"] - 5

    prev = base[
        ["ts_code", "q", "end_date", "roe", "revenue", "total_assets", "total_cur_assets", "total_nca", "total_ncl"]
    ].copy()

    cur = base.copy()
    cur = cur.merge(
        prev.add_suffix("_prev_y"),
        left_on=["ts_code", "q_prev_y"],
        right_on=["ts_code_prev_y", "q_prev_y"],
        how="left",
    )
    cur = cur.merge(
        prev.add_suffix("_lag1q"),
        left_on=["ts_code", "q_lag1q"],
        right_on=["ts_code_lag1q", "q_lag1q"],
        how="left",
    )
    cur = cur.merge(
        prev.add_suffix("_prev_y_lag1q"),
        left_on=["ts_code", "q_prev_y_lag1q"],
        right_on=["ts_code_prev_y_lag1q", "q_prev_y_lag1q"],
        how="left",
    )

    cur["lever"] = _safe_div(cur["total_ncl"], cur["total_nca"])
    cur["lever_prev_y"] = _safe_div(cur["total_ncl_prev_y"], cur["total_nca_prev_y"])
    cur["delta_roe"] = _safe_div(cur["roe"], cur["roe_prev_y"]) - 1.0

    cur_assets_avg = (cur["total_cur_assets"] + cur["total_cur_assets_lag1q"]) / 2.0
    cur_assets_avg_prev = (cur["total_cur_assets_prev_y"] + cur["total_cur_assets_prev_y_lag1q"]) / 2.0
    tot_assets_avg = (cur["total_assets"] + cur["total_assets_lag1q"]) / 2.0
    tot_assets_avg_prev = (cur["total_assets_prev_y"] + cur["total_assets_prev_y_lag1q"]) / 2.0

    cur["caturn"] = _safe_div(cur["revenue"], cur_assets_avg)
    cur["caturn_prev_y"] = _safe_div(cur["revenue_prev_y"], cur_assets_avg_prev)
    cur["delta_caturn"] = _safe_div(cur["caturn"], cur["caturn_prev_y"]) - 1.0

    cur["turn"] = _safe_div(cur["revenue"], tot_assets_avg)
    cur["turn_prev_y"] = _safe_div(cur["revenue_prev_y"], tot_assets_avg_prev)
    cur["delta_turn"] = _safe_div(cur["turn"], cur["turn_prev_y"]) - 1.0

    cur["delta_lever"] = -(_safe_div(cur["lever"], cur["lever_prev_y"]) - 1.0)

    cur["ff_roe"] = (cur["roe"] > 0).astype("float64")
    cur["ff_delta_roe"] = (cur["delta_roe"] > 0).astype("float64")
    cur["ff_delta_caturn"] = (cur["delta_caturn"] > 0).astype("float64")
    cur["ff_delta_turn"] = (cur["delta_turn"] > 0).astype("float64")
    cur["ff_delta_lever"] = (cur["delta_lever"] > 0).astype("float64")
    cur["ffscore"] = cur[
        ["ff_roe", "ff_delta_roe", "ff_delta_caturn", "ff_delta_turn", "ff_delta_lever"]
    ].sum(axis=1, min_count=5)

    keep = [
        "ts_code",
        "ann_date",
        "end_date",
        "roe",
        "delta_roe",
        "delta_caturn",
        "delta_turn",
        "delta_lever",
        "ffscore",
    ]
    return cur[keep].copy().sort_values(["ts_code", "ann_date"])


def build_monthly_panel(cfg: BacktestConfig) -> pd.DataFrame:
    mkt, _ = load_monthly_market_data(cfg.data_dir, cfg.start, cfg.end)
    fund = load_fundamental_snapshot(cfg.data_dir)

    month_ends = pd.DatetimeIndex(sorted(mkt["month_end"].dropna().unique()))
    snapshots: list[pd.DataFrame] = []
    for ts_code, g in fund.groupby("ts_code", sort=False):
        s = _asof_join_monthly(
            monthly_dates=month_ends,
            stock_rows=g,
            cols=["ts_code", "end_date", "roe", "delta_roe", "delta_caturn", "delta_turn", "delta_lever", "ffscore"],
        )
        snapshots.append(s)
    fund_monthly = pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()
    fund_monthly = fund_monthly.rename(columns={"end_date": "fund_end_date"})
    panel = mkt.merge(fund_monthly, on=["month_end", "ts_code"], how="left")

    if cfg.pb_proxy_path is not None and cfg.pb_proxy_path.exists():
        pb = pd.read_parquet(cfg.pb_proxy_path)
        required_pb_cols = {"month_end", "ts_code", "pb_proxy"}
        if not required_pb_cols.issubset(set(pb.columns)):
            raise RuntimeError(f"PB proxy file missing columns {required_pb_cols}: {cfg.pb_proxy_path}")
        pb = pb[["month_end", "ts_code", "pb_proxy"]].copy()
        pb["month_end"] = pd.to_datetime(pb["month_end"])
        pb["pb_proxy"] = pd.to_numeric(pb["pb_proxy"], errors="coerce")
        panel = panel.merge(pb, on=["month_end", "ts_code"], how="left")
    else:
        panel["pb_proxy"] = np.nan

    panel = panel.sort_values(["ts_code", "month_end"]).reset_index(drop=True)
    panel["ffscore_raw"] = panel["ffscore"]
    carry_m = max(0, int(cfg.ffscore_carry_months))
    panel["ffscore_used"] = (
        panel.groupby("ts_code", sort=False)["ffscore_raw"]
        .transform(lambda s: s.ffill(limit=carry_m))
        .astype("float64")
    )
    pb_carry_m = max(0, int(cfg.pb_carry_months))
    panel["pb_used"] = (
        panel.groupby("ts_code", sort=False)["pb_proxy"]
        .transform(lambda s: s.ffill(limit=pb_carry_m))
        .astype("float64")
    )

    # tradability and executable-return availability are still mandatory.
    required = ["close_m", "next_ret_1m", "ffscore_used"]
    panel["has_required_data"] = panel[required].notna().all(axis=1)
    panel["tradable_flag"] = panel["tradable_days"] >= int(cfg.min_tradable_days)
    panel["eligible"] = panel["tradable_flag"] & panel["has_required_data"]
    return panel.sort_values(["month_end", "ts_code"]).reset_index(drop=True)


def _build_equal_weight_top_quantile(score_row: pd.Series, top_quantile: float) -> pd.Series:
    s = pd.to_numeric(score_row, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = pd.Series(0.0, index=score_row.index, dtype="float64")
    if s.empty:
        return out
    n = len(s)
    k = max(1, int(np.floor(n * float(top_quantile))))
    pick = s.nlargest(k).index
    out.loc[pick] = 1.0 / float(k)
    return out


def _build_equal_weight_full_score(score_row: pd.Series) -> pd.Series:
    s = pd.to_numeric(score_row, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = pd.Series(0.0, index=score_row.index, dtype="float64")
    if s.empty:
        return out
    pick = s.index[s >= 5.0]
    n = int(len(pick))
    if n <= 0:
        return out
    out.loc[pick] = 1.0 / float(n)
    return out


def _build_rebalance_mask(dates: pd.DatetimeIndex, freq: str) -> pd.Series:
    if len(dates) == 0:
        return pd.Series(dtype=bool)
    f = str(freq).lower()
    if f == "daily":
        return pd.Series(True, index=dates)
    if f == "weekly":
        p = dates.to_period("W-FRI")
    elif f == "monthly":
        p = dates.to_period("M")
    else:
        raise ValueError(f"Unsupported rebalance_freq={freq}, expected daily|weekly|monthly")
    prev_p = pd.Series(p, index=dates).shift(1)
    # Rebalance on the first trading day of each new period.
    out = pd.Series(p != prev_p, index=dates).fillna(False).astype(bool)
    return out


def build_signal_period_end_by_trade_date(cal: pd.DataFrame, period_col: str = "period_end") -> pd.Series:
    trade_dates = pd.DatetimeIndex(cal.index)
    period_last_trade_dates = set(
        cal.reset_index().groupby(period_col, sort=True)["trade_date"].max().tolist()
    )
    out: dict[pd.Timestamp, pd.Timestamp | pd.NaT] = {}
    latest_signal_period_end: pd.Timestamp | pd.NaT = pd.NaT
    for i, dt in enumerate(trade_dates):
        if i > 0:
            prev_dt = pd.Timestamp(trade_dates[i - 1])
            if prev_dt in period_last_trade_dates:
                latest_signal_period_end = pd.Timestamp(cal.at[prev_dt, period_col])
        out[pd.Timestamp(dt)] = (
            pd.Timestamp(latest_signal_period_end) if pd.notna(latest_signal_period_end) else pd.NaT
        )
    return pd.Series(out).sort_index()


def _build_ffscore_rebalance_targets(
    score_panel: pd.DataFrame,
    pb_panel: pd.DataFrame,
    eligible_panel: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    rebalance_mask: pd.Series,
    signal_period_end_by_date: pd.Series,
    symbols: pd.Index,
    cfg: BacktestConfig,
) -> tuple[dict[pd.Timestamp, pd.Series], dict[pd.Timestamp, dict], dict[pd.Timestamp, pd.Series]]:
    target_by_date: dict[pd.Timestamp, pd.Series] = {}
    meta_by_date: dict[pd.Timestamp, dict] = {}
    elig_by_date: dict[pd.Timestamp, pd.Series] = {}
    zero_w = pd.Series(0.0, index=symbols, dtype="float64")

    for dt in trade_dates:
        dt = pd.Timestamp(dt)
        if not bool(rebalance_mask.loc[dt]):
            continue

        signal_period_end = signal_period_end_by_date.get(dt, pd.NaT)
        meta = {
            "signal_period_end": pd.Timestamp(signal_period_end) if pd.notna(signal_period_end) else pd.NaT,
            "n_eligible": 0,
            "n_pb_pool": 0,
        }
        if pd.isna(signal_period_end) or signal_period_end not in score_panel.index:
            target_by_date[dt] = zero_w.copy()
            meta_by_date[dt] = meta
            elig_by_date[dt] = pd.Series(False, index=symbols, dtype=bool)
            continue

        elig = eligible_panel.loc[signal_period_end].astype(bool)
        score = score_panel.loc[signal_period_end].where(elig)
        meta["n_eligible"] = int(elig.sum())

        if bool(cfg.use_pb_filter):
            pb_row = pd.to_numeric(pb_panel.loc[signal_period_end], errors="coerce")
            pb_valid = elig & pb_row.notna() & (pb_row > 0) & np.isfinite(pb_row)
            if int(pb_valid.sum()) > 0:
                thr = float(pb_row[pb_valid].quantile(float(cfg.pb_quantile)))
                low_pb_mask = pb_valid & (pb_row <= thr)
                meta["n_pb_pool"] = int(low_pb_mask.sum())
                score = score.where(low_pb_mask)
            else:
                score = pd.Series(np.nan, index=symbols, dtype="float64")

        if int(score.notna().sum()) >= int(cfg.min_score_names):
            if bool(cfg.ffscore_full_score_only):
                w_target = _build_equal_weight_full_score(score)
            else:
                w_target = _build_equal_weight_top_quantile(score, top_quantile=cfg.top_quantile)
        else:
            w_target = zero_w.copy()

        target_by_date[dt] = w_target.astype("float64")
        meta_by_date[dt] = meta
        elig_by_date[dt] = elig.astype(bool)

    return target_by_date, meta_by_date, elig_by_date


def run_backtest(panel: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = pd.DatetimeIndex(sorted(panel["month_end"].dropna().unique()))
    symbols = pd.Index(sorted(panel["ts_code"].dropna().unique()))
    if len(months) < 2 or len(symbols) == 0:
        raise RuntimeError("Too few rows for backtest.")

    score_panel = panel.pivot(index="month_end", columns="ts_code", values="ffscore_used").reindex(index=months, columns=symbols)
    pb_panel = panel.pivot(index="month_end", columns="ts_code", values="pb_used").reindex(index=months, columns=symbols)
    eligible_panel = (
        panel.pivot(index="month_end", columns="ts_code", values="eligible")
        .reindex(index=months, columns=symbols)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    _, daily = load_monthly_market_data(cfg.data_dir, cfg.start, cfg.end)
    px_col = "close_qfq" if "close_qfq" in daily.columns else "close"
    daily[px_col] = pd.to_numeric(daily[px_col], errors="coerce")
    daily["ret_d"] = daily.groupby("ts_code", sort=False)[px_col].pct_change()
    daily["month_end"] = _to_month_end(daily["trade_date"])
    daily["period_end"] = daily["month_end"]

    ret_d = daily.pivot(index="trade_date", columns="ts_code", values="ret_d").sort_index().reindex(columns=symbols)
    if ret_d.empty:
        raise RuntimeError("No daily returns in requested date range.")
    trade_dates = pd.DatetimeIndex(ret_d.index)
    rebalance_mask = _build_rebalance_mask(trade_dates, cfg.rebalance_freq)

    cal = daily[["trade_date", "period_end"]].drop_duplicates("trade_date").set_index("trade_date").sort_index()
    signal_period_end_by_date = build_signal_period_end_by_trade_date(cal, period_col="period_end")
    target_by_date, meta_by_date, elig_by_date = _build_ffscore_rebalance_targets(
        score_panel=score_panel,
        pb_panel=pb_panel,
        eligible_panel=eligible_panel,
        trade_dates=trade_dates,
        rebalance_mask=rebalance_mask,
        signal_period_end_by_date=signal_period_end_by_date,
        symbols=symbols,
        cfg=cfg,
    )
    return run_weight_execution_engine(
        ret_d=ret_d,
        cal=cal,
        symbols=symbols,
        rebalance_mask=rebalance_mask,
        target_by_date=target_by_date,
        meta_by_date=meta_by_date,
        elig_by_date=elig_by_date,
        cost_bps=float(cfg.cost_bps),
        vol_target_ann=float(cfg.vol_target_ann),
        vol_lookback_m=int(cfg.vol_lookback_m),
        max_leverage=float(cfg.max_leverage),
        period_col="period_end",
    )

def build_strategy_monthly_realized_vol(daily_bt: pd.DataFrame) -> pd.DataFrame:
    if daily_bt.empty:
        return pd.DataFrame(columns=["period_end", "realized_vol_ann", "month_ret"])
    rows: list[dict] = []
    for p_end, g in daily_bt.groupby("period_end", sort=True):
        r = pd.to_numeric(g["net_ret"], errors="coerce").fillna(0.0).astype("float64")
        month_ret = float((1.0 + r).prod() - 1.0)
        realized_vol_ann = float(r.std(ddof=1) * np.sqrt(252.0)) if len(r) > 1 else np.nan
        rows.append(
            {
                "period_end": pd.Timestamp(p_end),
                "realized_vol_ann": realized_vol_ann,
                "month_ret": month_ret,
                "n_days": int(len(r)),
            }
        )
    return pd.DataFrame(rows).sort_values("period_end")


def build_strategy_vol_lag_corr(strategy_monthly_vol: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    if strategy_monthly_vol.empty:
        return pd.DataFrame(columns=["lag_month", "corr", "n_pairs"])
    s = (
        strategy_monthly_vol.sort_values("period_end")
        .set_index("period_end")["realized_vol_ann"]
        .astype("float64")
    )
    rows: list[dict] = []
    for lag in range(1, int(max_lag) + 1):
        x = s
        y = s.shift(lag)
        m = x.notna() & y.notna()
        if int(m.sum()) > 2:
            xv = x[m].astype("float64")
            yv = y[m].astype("float64")
            if float(xv.std(ddof=1)) == 0.0 or float(yv.std(ddof=1)) == 0.0:
                corr = np.nan
            else:
                corr = float(xv.corr(yv))
        else:
            corr = np.nan
        rows.append({"lag_month": int(lag), "corr": corr, "n_pairs": int(m.sum())})
    return pd.DataFrame(rows)


def save_plots(
    daily_bt: pd.DataFrame,
    strategy_monthly_realized_vol: pd.DataFrame,
    strategy_vol_lag_corr: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_bt["trade_date"], daily_bt["equity"], lw=1.2, color="tab:blue", label="FFScore strategy")
    ax.set_title("FFScore Backtest PnL (Equity Curve)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pnl_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(strategy_vol_lag_corr["lag_month"], strategy_vol_lag_corr["corr"], marker="o", lw=1.5, color="tab:green")
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.4)
    ax.set_title("Strategy Vol Memory: corr(vol_t, vol_{t-lag})")
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Correlation")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_vol_lag_correlation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(strategy_monthly_realized_vol["period_end"], strategy_monthly_realized_vol["realized_vol_ann"], lw=1.4, color="tab:green")
    ax.set_title("Strategy Monthly Realized Vol (from Daily Net Returns)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Vol")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_realized_vol_timeseries.png", dpi=160)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FFScore backtest with monthly tradability filters and drift-aware costs.")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out-dir", type=str, default="reports/ffscore_backtest")
    p.add_argument("--start", type=str, default="2013-01-01")
    p.add_argument("--end", type=str, default="2026-01-31")
    p.add_argument("--top-quantile", type=float, default=0.2, help="Select top FFScore names by this quantile monthly.")
    p.add_argument("--min-tradable-days", type=int, default=2, help="Non-tradable if monthly tradable days < this.")
    p.add_argument("--min-score-names", type=int, default=0, help="If fewer eligible names, stay in cash. Not activated if 0.")
    p.add_argument("--cost-bps", type=float, default=10.0, help="One-way transaction cost in bps via turnover * rate.")
    p.add_argument("--max-vol-lag", type=int, default=24, help="Max month lag for vol memory correlation plot.")
    p.add_argument(
        "--use-pb-filter",
        action="store_true",
        default=True,
        help="Filter to low PB names before FFScore ranking. Default: enabled.",
    )
    p.add_argument("--pb-quantile", type=float, default=0.2, help="Low-PB quantile threshold when PB filter enabled.")
    p.add_argument(
        "--pb-proxy-path",
        type=str,
        default="data/derived/pb_proxy_monthly.parquet",
        help="PB proxy parquet path with columns month_end,ts_code,pb_proxy.",
    )
    p.add_argument("--vol-target-ann", type=float, default=0.0, help="Annualized vol target. 0 disables.")
    p.add_argument("--vol-lookback-m", type=int, default=12, help="Lookback months for realized-vol estimate.")
    p.add_argument("--max-leverage", type=float, default=1.0, help="Cap total portfolio leverage.")
    p.add_argument(
        "--ffscore-carry-months",
        type=int,
        default=2,
        help="Carry forward last available FFScore for at most this many months.",
    )
    p.add_argument(
        "--pb-carry-months",
        type=int,
        default=2,
        help="Carry forward last available PB proxy for at most this many months.",
    )
    p.add_argument(
        "--ffscore-full-score-only",
        action="store_true",
        default=True,
        help="Only buy names with FFScore >= 5. Default: enabled.",
    )
    p.add_argument(
        "--rebalance-freq",
        type=str,
        default="monthly",
        choices=["daily", "weekly", "monthly"],
        help="Rebalance frequency for daily execution engine.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = BacktestConfig(
        data_dir=Path(args.data_dir).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        start=args.start,
        end=args.end,
        top_quantile=float(args.top_quantile),
        min_tradable_days=int(args.min_tradable_days),
        min_score_names=int(args.min_score_names),
        cost_bps=float(args.cost_bps),
        max_vol_lag=int(args.max_vol_lag),
        use_pb_filter=bool(args.use_pb_filter),
        pb_quantile=float(args.pb_quantile),
        pb_proxy_path=(Path(args.pb_proxy_path).resolve() if args.pb_proxy_path else None),
        vol_target_ann=float(args.vol_target_ann),
        vol_lookback_m=int(args.vol_lookback_m),
        max_leverage=float(args.max_leverage),
        ffscore_carry_months=int(args.ffscore_carry_months),
        pb_carry_months=int(args.pb_carry_months),
        ffscore_full_score_only=bool(args.ffscore_full_score_only),
        rebalance_freq=str(args.rebalance_freq),
    )

    panel = build_monthly_panel(cfg)
    rebalance_info, daily_bt, w_daily = run_backtest(panel, cfg)
    stats = performance_stats(daily_bt["net_ret"], periods_per_year=252)
    strategy_monthly_realized_vol = build_strategy_monthly_realized_vol(daily_bt)
    strategy_vol_lag_corr = build_strategy_vol_lag_corr(strategy_monthly_realized_vol, cfg.max_vol_lag)
    save_plots(daily_bt, strategy_monthly_realized_vol, strategy_vol_lag_corr, cfg.out_dir)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cfg.out_dir / "monthly_panel.parquet", index=False)
    daily_bt.to_parquet(cfg.out_dir / "daily_strategy_pnl.parquet", index=False)
    daily_bt.to_parquet(cfg.out_dir / "backtest_curve.parquet", index=False)
    w_daily.reset_index().to_parquet(cfg.out_dir / "weights_daily.parquet", index=False)
    rebalance_info.to_parquet(cfg.out_dir / "rebalance_info.parquet", index=False)
    strategy_monthly_realized_vol.to_csv(cfg.out_dir / "strategy_monthly_realized_vol.csv", index=False)
    strategy_vol_lag_corr.to_csv(cfg.out_dir / "strategy_vol_lag_correlation.csv", index=False)
    pd.DataFrame([stats]).to_csv(cfg.out_dir / "performance_summary.csv", index=False)

    print(f"[done] outputs saved to: {cfg.out_dir}")
    print(f"[perf] annualized_return={stats['annualized_return']:.4%}")
    print(f"[perf] annualized_vol={stats['annualized_vol']:.4%}")
    print(f"[perf] sharpe={stats['sharpe']:.3f}")
    print(f"[perf] max_drawdown={stats['max_drawdown']:.4%}")
    print(f"[perf] total_return={stats['total_return']:.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

