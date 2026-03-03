"""
Download A-share daily OHLCV (2005-2026-01) with 前复权 (qfq),
plus fundamentals needed by FFScore.ipynb.

Data sources:
  - Tushare Pro: daily, adj_factor, daily_basic(pb), balancesheet, income, cashflow, fina_indicator

Outputs (default under ./data):
  - data/meta/stock_basic.parquet
  - data/raw/daily/daily_raw_YYYY.parquet
  - data/raw/adj_factor/adj_factor_YYYY.parquet
  - data/raw/daily_basic/daily_basic_pb_YYYY.parquet
  - data/daily_qfq/daily_qfq_YYYY.parquet
  - data/meta/base_adj_factor.parquet
  - data/fundamental/{balancesheet,income,cashflow,fina_indicator}.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def _read_token(token_path: Path) -> str:
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"Empty token file: {token_path}")
    return token


def _ymd(s: str) -> str:
    """Accept YYYY-MM-DD or YYYYMMDD -> YYYYMMDD."""
    s = s.strip()
    if "-" in s:
        return s.replace("-", "")
    return s


def _year_chunks(start_ymd: str, end_ymd: str) -> List[Tuple[str, str, int]]:
    start_dt = pd.to_datetime(start_ymd)
    end_dt = pd.to_datetime(end_ymd)
    years = list(range(start_dt.year, end_dt.year + 1))
    chunks: List[Tuple[str, str, int]] = []
    for y in years:
        c_start = max(start_dt, pd.Timestamp(year=y, month=1, day=1))
        c_end = min(end_dt, pd.Timestamp(year=y, month=12, day=31))
        chunks.append((c_start.strftime("%Y%m%d"), c_end.strftime("%Y%m%d"), y))
    return chunks


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """
    Write parquet atomically (best-effort): write to temp file then replace.
    Prevents corrupted/empty parquet when interrupted.
    """
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def combine_parquet_files(files: List[Path], out_path: Path) -> None:
    """
    Combine parquet files to one parquet.
    Handles schema mismatches (e.g. null vs double columns) by reading via pandas
    and concatenating, which coerces types automatically.
    """
    _ensure_dir(out_path.parent)
    if not files:
        save_parquet_atomic(pd.DataFrame(), out_path)
        return

    frames: List[pd.DataFrame] = []
    for fp in tqdm(files, desc=f"combine {out_path.stem}", leave=False):
        try:
            df = pd.read_parquet(fp)
        except Exception:  # noqa: BLE001
            continue
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        save_parquet_atomic(pd.DataFrame(), out_path)
        return

    combined = pd.concat(frames, ignore_index=True)
    save_parquet_atomic(combined, out_path)


@dataclass(frozen=True)
class Retry:
    max_tries: int = 6
    base_sleep: float = 1.2
    max_sleep: float = 30.0


def _call_with_retry(fn, *, retry: Retry, desc: str):
    last_err: Optional[BaseException] = None
    for i in range(retry.max_tries):
        try:
            return fn()
        except BaseException as e:  # noqa: BLE001
            last_err = e
            sleep = min(retry.max_sleep, retry.base_sleep * (2**i))
            print(f"[warn] {desc} failed ({type(e).__name__}): {e}. sleep {sleep:.1f}s", file=sys.stderr)
            time.sleep(sleep)
    assert last_err is not None
    raise last_err


def _is_transport_empty_df(df: Any) -> bool:
    """
    Tushare python client returns an *empty DataFrame with 0 columns* when HTTP status >= 400
    (e.g. 504 Gateway Time-out) because `requests.Response` is falsy in that case.
    """
    return isinstance(df, pd.DataFrame) and df.empty and len(df.columns) == 0


def fetch_paginated(
    pro,
    api_name: str,
    params: Dict[str, Any],
    *,
    fields: Optional[str] = None,
    limit: int = 4000,
    max_offset: Optional[int] = 100000,
    retry: Retry = Retry(),
    sleep_s: float = 0.25,
) -> pd.DataFrame:
    """
    Fetch tushare pro data with offset/limit pagination.
    Works for endpoints that support server-side pagination.
    """
    out: List[pd.DataFrame] = []
    offset = 0
    while True:
        if max_offset is not None and offset >= max_offset:
            raise RuntimeError(
                f"{api_name} offset reached {offset} (>= {max_offset}). "
                "This endpoint likely has an offset cap; split the date range into smaller windows."
            )

        def _once():
            kw = dict(params)
            kw["offset"] = offset
            kw["limit"] = limit
            if fields is not None:
                kw["fields"] = fields
            df0 = pro.query(api_name, **kw)
            if _is_transport_empty_df(df0):
                raise RuntimeError(f"{api_name} returned empty(0 cols) - possible HTTP timeout (e.g. 504)")
            return df0

        df = _call_with_retry(_once, retry=retry, desc=f"{api_name} offset={offset}")
        if df is None or df.empty:
            break
        out.append(df)
        got = len(df)
        offset += got
        time.sleep(sleep_s)
        if got < limit:
            break
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def compute_base_adj_factor(adj_paths: Iterable[Path], out_path: Path) -> pd.DataFrame:
    """
    Compute per-ts_code latest adj_factor across all saved chunks.
    Returns DataFrame(ts_code, trade_date, adj_factor) where trade_date is latest available.
    """
    latest: Dict[str, Tuple[str, float]] = {}
    for p in tqdm(list(adj_paths), desc="compute base adj_factor"):
        df = load_parquet(p)
        if df.empty:
            continue
        df = df[["ts_code", "trade_date", "adj_factor"]].copy()
        df["trade_date"] = df["trade_date"].astype(str)
        # process in ascending trade_date so later overwrites
        df.sort_values(["ts_code", "trade_date"], inplace=True)
        for ts_code, g in df.groupby("ts_code", sort=False):
            td = str(g["trade_date"].iloc[-1])
            af = float(g["adj_factor"].iloc[-1])
            prev = latest.get(ts_code)
            if prev is None or td >= prev[0]:
                latest[ts_code] = (td, af)

    base = pd.DataFrame(
        [{"ts_code": k, "trade_date": v[0], "adj_factor": v[1]} for k, v in latest.items()]
    )
    base.sort_values("ts_code", inplace=True)
    save_parquet(base, out_path)
    return base


def apply_qfq(daily: pd.DataFrame, adj: pd.DataFrame, base_adj: pd.DataFrame) -> pd.DataFrame:
    """
    前复权: qfq_price = raw_price * adj_factor / base_adj_factor(ts_code).
    """
    if daily.empty:
        return daily
    d = daily.copy()
    a = adj.copy()
    b = base_adj.copy()

    d["trade_date"] = d["trade_date"].astype(str)
    a["trade_date"] = a["trade_date"].astype(str)

    d = d.merge(a[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
    d = d.merge(
        b[["ts_code", "adj_factor"]].rename(columns={"adj_factor": "base_adj_factor"}),
        on="ts_code",
        how="left",
    )
    d["qfq_ratio"] = d["adj_factor"] / d["base_adj_factor"]

    for col in ["open", "high", "low", "close", "pre_close"]:
        if col in d.columns:
            d[col + "_qfq"] = d[col] * d["qfq_ratio"]

    return d


def download_stock_basic(pro, out_dir: Path) -> pd.DataFrame:
    expected_cols = ["ts_code", "symbol", "name", "area", "industry", "market", "list_date", "delist_date", "list_status"]
    path = out_dir / "meta" / "stock_basic.parquet"

    def _read_existing() -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df0 = load_parquet(path)
        except Exception:  # noqa: BLE001
            return None
        if df0 is None or df0.empty or ("ts_code" not in df0.columns):
            return None
        return df0

    frames = []
    for status in ["L", "D", "P"]:
        df = fetch_paginated(
            pro,
            "stock_basic",
            {"exchange": "", "list_status": status},
            fields="ts_code,symbol,name,area,industry,market,list_date,delist_date",
            limit=5000,
        )
        if not df.empty:
            df["list_status"] = status
            frames.append(df)
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
    else:
        # If API returned nothing (network/permission/limit), do NOT overwrite an existing good file.
        existing = _read_existing()
        if existing is not None:
            print("[warn] stock_basic returned empty; keep existing local stock_basic.parquet", file=sys.stderr)
            return existing
        all_df = pd.DataFrame(columns=expected_cols)

    # ensure schema contains ts_code even if empty
    for c in expected_cols:
        if c not in all_df.columns:
            all_df[c] = pd.Series(dtype="object")
    all_df = all_df[expected_cols]
    save_parquet_atomic(all_df, path)
    return all_df


def download_daily_and_factors(
    pro,
    *,
    start_ymd: str,
    end_ymd: str,
    out_dir: Path,
    resume: bool = True,
    trade_window: int = 5,
) -> None:
    raw_daily_dir = out_dir / "raw" / "daily"
    raw_adj_dir = out_dir / "raw" / "adj_factor"
    raw_basic_dir = out_dir / "raw" / "daily_basic"
    qfq_dir = out_dir / "daily_qfq"
    meta_dir = out_dir / "meta"

    _ensure_dir(raw_daily_dir)
    _ensure_dir(raw_adj_dir)
    _ensure_dir(raw_basic_dir)
    _ensure_dir(qfq_dir)
    _ensure_dir(meta_dir)

    daily_fields = "ts_code,trade_date,open,high,low,close,pre_close,vol,amount"
    basic_fields = "ts_code,trade_date,pb"
    adj_fields = "ts_code,trade_date,adj_factor"

    def _get_trade_dates(s: str, e: str) -> List[str]:
        # Use SSE calendar as a practical proxy for A-share trading days.
        try:
            cal = fetch_paginated(
                pro,
                "trade_cal",
                {"exchange": "SSE", "start_date": s, "end_date": e, "is_open": "1"},
                fields="cal_date",
                limit=2000,
                max_offset=20000,
                sleep_s=0.1,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] trade_cal failed for {s}~{e}: {e}", file=sys.stderr)
            return []
        if cal.empty:
            return []
        return sorted(cal["cal_date"].astype(str).tolist())

    def _iter_calendar_windows(s: str, e: str, days: int = 7) -> List[Tuple[str, str]]:
        s_dt = pd.to_datetime(s)
        e_dt = pd.to_datetime(e)
        out: List[Tuple[str, str]] = []
        cur = s_dt
        step = pd.Timedelta(days=days)
        while cur <= e_dt:
            nxt = min(e_dt, cur + step)
            out.append((cur.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")))
            cur = nxt + pd.Timedelta(days=1)
        return out

    def _iter_windows(dates: List[str], n: int) -> Iterable[Tuple[str, str]]:
        if n <= 0:
            raise ValueError("trade_window must be positive")
        for i in range(0, len(dates), n):
            yield dates[i], dates[min(i + n - 1, len(dates) - 1)]

    chunks = _year_chunks(start_ymd, end_ymd)

    def _year_job(
        y: int,
        y_start: str,
        y_end: str,
        api_name: str,
        fields: str,
        out_file: Path,
        parts_root: Path,
    ) -> None:
        complete_marker = out_file.with_suffix(out_file.suffix + ".complete")
        if resume and out_file.exists() and complete_marker.exists():
            try:
                chk = pd.read_parquet(out_file, engine="pyarrow", columns=None)
            except Exception:  # noqa: BLE001
                chk = pd.DataFrame()
            # If previous run mistakenly marked complete with an empty/invalid parquet, redo it.
            if isinstance(chk, pd.DataFrame) and (chk.empty and len(chk.columns) == 0):
                try:
                    out_file.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001
                    pass
                try:
                    complete_marker.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001
                    pass
            else:
                return

        dates = _get_trade_dates(y_start, y_end)
        if dates:
            windows = list(_iter_windows(dates, trade_window))
        else:
            # Fallback when trade_cal unavailable: use calendar windows to continue downloading.
            # Do NOT mark complete with empty data.
            windows = _iter_calendar_windows(y_start, y_end, days=max(3, trade_window * 2))

        parts_dir = parts_root / str(y)
        _ensure_dir(parts_dir)

        expected_parts: List[Path] = []
        for w_start, w_end in tqdm(windows, desc=f"{y} {api_name}", leave=False):
            part_fp = parts_dir / f"{api_name}_{w_start}_{w_end}.parquet"
            expected_parts.append(part_fp)
            if resume and part_fp.exists():
                continue
            try:
                df = fetch_paginated(
                    pro,
                    api_name,
                    {"start_date": w_start, "end_date": w_end},
                    fields=fields,
                    limit=4000,
                    max_offset=100000,
                )
                # Don't persist transport-failure empties
                if _is_transport_empty_df(df):
                    raise RuntimeError(f"{api_name} window {w_start}~{w_end} empty(0 cols)")
                save_parquet_atomic(df, part_fp)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {api_name} {y} window {w_start}~{w_end} failed: {e}", file=sys.stderr)
                continue

        # Only combine when all expected parts exist
        if not all(p.exists() for p in expected_parts):
            missing = sum(1 for p in expected_parts if not p.exists())
            print(f"[warn] {api_name} {y}: missing {missing}/{len(expected_parts)} windows; skip combine", file=sys.stderr)
            return

        # combine parts -> yearly file
        part_files = sorted(parts_dir.glob(f"{api_name}_*.parquet"))
        combine_parquet_files(part_files, out_file)
        complete_marker.write_text("ok", encoding="utf-8")

    for y_start, y_end, y in tqdm(chunks, desc="download yearly chunks"):
        p_daily = raw_daily_dir / f"daily_raw_{y}.parquet"
        p_adj = raw_adj_dir / f"adj_factor_{y}.parquet"
        p_basic = raw_basic_dir / f"daily_basic_pb_{y}.parquet"

        _year_job(
            y,
            y_start,
            y_end,
            "daily",
            daily_fields,
            p_daily,
            raw_daily_dir / "parts",
        )
        _year_job(
            y,
            y_start,
            y_end,
            "adj_factor",
            adj_fields,
            p_adj,
            raw_adj_dir / "parts",
        )
        _year_job(
            y,
            y_start,
            y_end,
            "daily_basic",
            basic_fields,
            p_basic,
            raw_basic_dir / "parts",
        )

    # base adj_factor for qfq
    base_path = meta_dir / "base_adj_factor.parquet"
    if (not resume) or (not base_path.exists()):
        adj_paths = sorted(raw_adj_dir.glob("adj_factor_*.parquet"))
        base = compute_base_adj_factor(adj_paths, base_path)
    else:
        base = load_parquet(base_path)

    # compute qfq per year
    for _, _, y in tqdm(chunks, desc="compute qfq yearly"):
        p_qfq = qfq_dir / f"daily_qfq_{y}.parquet"
        if resume and p_qfq.exists():
            continue
        p_daily = raw_daily_dir / f"daily_raw_{y}.parquet"
        p_adj = raw_adj_dir / f"adj_factor_{y}.parquet"
        if not p_daily.exists() or not p_adj.exists():
            continue
        d = load_parquet(p_daily)
        a = load_parquet(p_adj)
        q = apply_qfq(d, a, base)
        save_parquet(q, p_qfq)


def download_fundamentals(
    pro,
    *,
    report_start: str,
    report_end: str,
    out_dir: Path,
    resume: bool = True,
    list_status: str = "L,D,P",
    combine: bool = True,
    sleep_s: float = 0.15,
    workers: int = 1,
) -> None:
    fund_dir = out_dir / "fundamental"
    meta_path = out_dir / "meta" / "stock_basic.parquet"
    _ensure_dir(fund_dir)

    def _infer_ts_codes_from_daily() -> List[str]:
        # Infer ts_code from already-downloaded market data files.
        # Prefer daily_qfq (it is derived and usually already exists).
        candidates = [
            ("daily_qfq", out_dir / "daily_qfq", "daily_qfq_*.parquet"),
            ("daily_raw", out_dir / "raw" / "daily", "daily_raw_*.parquet"),
        ]
        codes: set[str] = set()
        for label, base, pat in candidates:
            files = sorted(base.glob(pat))
            if not files:
                continue
            for fp in tqdm(files, desc=f"infer ts_code from {label}"):
                try:
                    col = pd.read_parquet(fp, columns=["ts_code"])
                except Exception:  # noqa: BLE001
                    continue
                if col is None or col.empty or "ts_code" not in col.columns:
                    continue
                codes.update(col["ts_code"].dropna().astype(str).unique().tolist())
            if codes:
                break
        return sorted(codes)

    # Prefer local market-data-derived ts_code list (more robust when API is flaky).
    ts_codes = _infer_ts_codes_from_daily()
    if not ts_codes:
        # Fallback to stock_basic (may fail if network/DNS/504 issues).
        stock_basic: Optional[pd.DataFrame] = None
        if meta_path.exists():
            try:
                stock_basic = load_parquet(meta_path)
            except Exception:  # noqa: BLE001
                stock_basic = None
        if stock_basic is None or stock_basic.empty or ("ts_code" not in stock_basic.columns):
            try:
                stock_basic = download_stock_basic(pro, out_dir)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] stock_basic download failed: {e}", file=sys.stderr)
                stock_basic = None

        if stock_basic is not None and ("ts_code" in stock_basic.columns):
            allowed = {s.strip() for s in list_status.split(",") if s.strip()}
            if "list_status" in stock_basic.columns and allowed:
                stock_basic = stock_basic[stock_basic["list_status"].isin(allowed)]
            ts_codes = sorted(stock_basic["ts_code"].dropna().astype(str).unique().tolist())

    if not ts_codes:
        raise RuntimeError(
            "No ts_code available (stock_basic empty and cannot infer from daily_raw). "
            "Please ensure daily data has been downloaded and/or Tushare API is reachable."
        )

    endpoints = [
        (
            "balancesheet",
            "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,total_assets,total_cur_assets,total_cur_liab,total_nca,total_ncl,total_share,paidin_capital",
        ),
        ("income", "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,revenue,total_profit,fin_exp"),
        ("cashflow", "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,n_cashflow_act"),
        ("fina_indicator", "ts_code,ann_date,end_date,roe,roa,grossprofit_margin"),
    ]

    def _safe_ts(ts_code: str) -> str:
        return ts_code.replace(".", "_").replace("/", "_")

    def _download_one(api_name: str, fields: str, ts_code: str, per_dir: Path) -> Optional[str]:
        out_fp = per_dir / f"{_safe_ts(ts_code)}.parquet"
        if resume and out_fp.exists():
            # Check that existing file has all requested columns; re-download if not.
            expected_cols = {c.strip() for c in fields.split(",")}
            try:
                existing_cols = set(pd.read_parquet(out_fp, columns=[]).columns)
                if expected_cols.issubset(existing_cols):
                    return None
            except Exception:  # noqa: BLE001
                pass

        def _once():
            df0 = pro.query(
                api_name,
                ts_code=ts_code,
                start_date=report_start,
                end_date=report_end,
                fields=fields,
            )
            if _is_transport_empty_df(df0):
                raise RuntimeError(f"{api_name} ts_code={ts_code} returned empty(0 cols) - possible HTTP timeout")
            return df0

        try:
            df = _call_with_retry(_once, retry=Retry(max_tries=6), desc=f"{api_name} ts_code={ts_code}")
            save_parquet_atomic(df if df is not None else pd.DataFrame(), out_fp)
        except Exception as e:  # noqa: BLE001
            return f"{api_name} ts_code={ts_code} failed: {e}"
        time.sleep(sleep_s)
        return None

    effective_workers = max(1, workers)
    print(f"[info] fundamental download workers: {effective_workers}")

    for api_name, fields in tqdm(endpoints, desc="download fundamentals"):
        per_dir = fund_dir / api_name / "by_ts_code"
        _ensure_dir(per_dir)
        combined_path = fund_dir / f"{api_name}.parquet"

        # Remove stale combined file if its schema is missing required columns.
        if combined_path.exists():
            expected_cols = {c.strip() for c in fields.split(",")}
            try:
                existing_cols = set(pd.read_parquet(combined_path, columns=[]).columns)
                if not expected_cols.issubset(existing_cols):
                    print(f"[info] {api_name}: combined file missing columns {expected_cols - existing_cols}, will re-download & re-combine")
                    combined_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

        if effective_workers <= 1:
            for ts_code in tqdm(ts_codes, desc=f"{api_name}", leave=False):
                err = _download_one(api_name, fields, ts_code, per_dir)
                if err:
                    print(f"[warn] {err}", file=sys.stderr)
        else:
            pbar = tqdm(total=len(ts_codes), desc=f"{api_name}", leave=False)
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futs = {
                    pool.submit(_download_one, api_name, fields, ts_code, per_dir): ts_code
                    for ts_code in ts_codes
                }
                for fut in as_completed(futs):
                    pbar.update(1)
                    try:
                        err = fut.result()
                        if err:
                            print(f"[warn] {err}", file=sys.stderr)
                    except Exception as e:  # noqa: BLE001
                        print(f"[warn] {futs[fut]} exception: {e}", file=sys.stderr)
            pbar.close()

        # combine (optional)
        if combine and ((not resume) or (not combined_path.exists())):
            files = sorted(per_dir.glob("*.parquet"))
            combine_parquet_files(files, combined_path)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--token-path", default="token", help="Path to tushare token file (default: ./token)")
    p.add_argument("--out-dir", default="data", help="Output directory (default: ./data)")
    p.add_argument(
        "--api-url",
        default="https://api.waditu.com/dataapi",
        help="Tushare Pro base URL (default: https://api.waditu.com/dataapi)",
    )
    p.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds for Tushare (default: 60)")
    p.add_argument("--start", default="20050101", help="Start date for daily data (YYYYMMDD or YYYY-MM-DD)")
    p.add_argument("--end", default="20260131", help="End date for daily data (YYYYMMDD or YYYY-MM-DD)")
    p.add_argument("--report-start", default="20040101", help="Start date for report end_date range (YYYYMMDD)")
    p.add_argument("--report-end", default="20251231", help="End date for report end_date range (YYYYMMDD)")
    p.add_argument("--trade-window", type=int, default=5, help="Trading-day window size for paging (default: 5)")
    p.add_argument(
        "--only-fundamentals",
        action="store_true",
        help="Only download fundamentals; skip daily/adj_factor/daily_basic/qfq computation",
    )
    p.add_argument(
        "--list-status",
        default="L,D,P",
        help="stock_basic list_status filter for fundamentals (default: L,D,P)",
    )
    p.add_argument(
        "--no-combine-fundamentals",
        action="store_true",
        help="Do not combine per-ts_code fundamentals into one parquet",
    )
    p.add_argument("--workers", type=int, default=4, help="Concurrent download threads for fundamentals (default: 4)")
    p.add_argument("--no-resume", action="store_true", help="Do not skip existing output files")
    args = p.parse_args(argv)

    start_ymd = _ymd(args.start)
    end_ymd = _ymd(args.end)

    out_dir = Path(args.out_dir).resolve()
    token_path = Path(args.token_path).resolve()
    resume = not args.no_resume

    import tushare as ts  # imported here so file can be parsed without tushare installed

    token = _read_token(token_path)
    ts.set_token(token)
    pro = ts.pro_api(timeout=args.timeout)
    # Switch to HTTPS endpoint by default (some networks block/unstable on plain HTTP).
    if getattr(args, "api_url", None):
        try:
            pro._DataApi__http_url = args.api_url  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to set api url: {e}", file=sys.stderr)

    print(f"[info] out_dir={out_dir}")
    print(f"[info] daily range: {start_ymd} ~ {end_ymd}")
    print(f"[info] report end_date range: {_ymd(args.report_start)} ~ {_ymd(args.report_end)}")

    if not args.only_fundamentals:
        download_daily_and_factors(
            pro,
            start_ymd=start_ymd,
            end_ymd=end_ymd,
            out_dir=out_dir,
            resume=resume,
            trade_window=args.trade_window,
        )
    download_fundamentals(
        pro,
        report_start=_ymd(args.report_start),
        report_end=_ymd(args.report_end),
        out_dir=out_dir,
        resume=resume,
        list_status=args.list_status,
        combine=not args.no_combine_fundamentals,
        workers=args.workers,
    )

    print("[done] download completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

