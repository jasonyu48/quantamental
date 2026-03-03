from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
        raise RuntimeError(f"All files are empty under {path} with pattern={pattern}")
    return pd.concat(frames, ignore_index=True)


def _to_month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp("M")


def load_monthly_price(data_dir: Path, start: str, end: str) -> pd.DataFrame:
    daily = _read_parquet_folder(data_dir / "daily_qfq", "daily_qfq_*.parquet")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"].astype(str))
    daily = daily[(daily["trade_date"] >= pd.Timestamp(start)) & (daily["trade_date"] <= pd.Timestamp(end))].copy()
    if daily.empty:
        raise RuntimeError("No daily rows in requested period.")

    px_col = "close_qfq" if "close_qfq" in daily.columns else "close"
    daily[px_col] = pd.to_numeric(daily[px_col], errors="coerce")
    daily["month_end"] = _to_month_end(daily["trade_date"])

    monthly_px = (
        daily.sort_values(["ts_code", "trade_date"])
        .groupby(["ts_code", "month_end"], as_index=False)[px_col]
        .last()
        .rename(columns={px_col: "close_m"})
    )
    return monthly_px


def load_book_value(data_dir: Path, share_unit_multiplier: float) -> pd.DataFrame:
    bs = pd.read_parquet(data_dir / "fundamental" / "balancesheet.parquet")
    need = ["ts_code", "ann_date", "end_date", "total_assets", "total_cur_liab", "total_ncl", "total_share"]
    miss = [c for c in need if c not in bs.columns]
    if miss:
        raise RuntimeError(f"balancesheet missing required columns: {miss}")
    bs = bs[need].copy()
    bs["ann_date"] = pd.to_datetime(bs["ann_date"].astype(str), errors="coerce")
    bs["end_date"] = pd.to_datetime(bs["end_date"].astype(str), errors="coerce")
    bs = bs.dropna(subset=["ts_code", "ann_date", "end_date"])

    for c in ["total_assets", "total_cur_liab", "total_ncl", "total_share"]:
        bs[c] = pd.to_numeric(bs[c], errors="coerce")

    # Approximate total liabilities from available fields.
    bs["total_liab_proxy"] = bs["total_cur_liab"] + bs["total_ncl"]
    bs["book_equity_proxy"] = bs["total_assets"] - bs["total_liab_proxy"]

    # Tushare `total_share` is typically in 10k shares, convert to shares by default.
    shares = bs["total_share"] * float(share_unit_multiplier)
    bs["bps_proxy"] = bs["book_equity_proxy"] / shares

    # Keep the latest announcement for each statement period.
    out = (
        bs.sort_values(["ts_code", "end_date", "ann_date"])
        .drop_duplicates(["ts_code", "end_date"], keep="last")
        .sort_values(["ts_code", "ann_date"])
    )
    keep = ["ts_code", "ann_date", "end_date", "book_equity_proxy", "bps_proxy"]
    return out[keep]


def join_pit_pb(monthly_px: pd.DataFrame, book_df: pd.DataFrame) -> pd.DataFrame:
    out_frames: list[pd.DataFrame] = []
    months = pd.DatetimeIndex(sorted(monthly_px["month_end"].unique()))
    for ts_code, px_g in monthly_px.groupby("ts_code", sort=False):
        b_g = book_df[book_df["ts_code"] == ts_code].sort_values("ann_date")
        if b_g.empty:
            continue
        left = pd.DataFrame({"month_end": months})
        asof = pd.merge_asof(
            left.sort_values("month_end"),
            b_g[["ann_date", "end_date", "book_equity_proxy", "bps_proxy"]].sort_values("ann_date"),
            left_on="month_end",
            right_on="ann_date",
            direction="backward",
            allow_exact_matches=True,
        )
        asof["ts_code"] = ts_code
        px_this = px_g[["month_end", "close_m"]]
        one = asof.merge(px_this, on="month_end", how="left")
        one["pb_proxy"] = one["close_m"] / one["bps_proxy"]
        out_frames.append(one)

    if not out_frames:
        return pd.DataFrame(columns=["month_end", "ts_code", "close_m", "pb_proxy"])
    out = pd.concat(out_frames, ignore_index=True)
    out["pb_proxy"] = pd.to_numeric(out["pb_proxy"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["month_end", "ts_code"])


def main() -> int:
    p = argparse.ArgumentParser(description="Derive PB proxy from existing daily+balancesheet data (no Tushare call).")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="data/derived")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument(
        "--share-unit-multiplier",
        type=float,
        default=1.0,
        help="Multiply total_share by this value before computing BPS. Default 1 for this local dataset.",
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly_px = load_monthly_price(data_dir, args.start, args.end)
    book_df = load_book_value(data_dir, share_unit_multiplier=float(args.share_unit_multiplier))
    pb = join_pit_pb(monthly_px, book_df)

    # Sanity clip only for output stats (not for saved value).
    s = pb["pb_proxy"].replace([np.inf, -np.inf], np.nan).dropna()
    s = s[(s > 0) & (s < 200)]
    print(f"[info] pb_proxy valid rows={len(s):,}, median={s.median():.3f}, p10={s.quantile(0.1):.3f}, p90={s.quantile(0.9):.3f}")

    out_fp = out_dir / "pb_proxy_monthly.parquet"
    pb.to_parquet(out_fp, index=False)
    print(f"[done] wrote {out_fp} rows={len(pb):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

