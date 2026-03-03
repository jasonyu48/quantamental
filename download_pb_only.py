from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import tushare as ts


def _ymd(s: str) -> str:
    s = s.strip()
    return s.replace("-", "")


def _read_token(token_path: Path) -> str:
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"Empty token file: {token_path}")
    return token


def _year_chunks(start_ymd: str, end_ymd: str) -> list[tuple[str, str, int]]:
    start_dt = pd.to_datetime(start_ymd)
    end_dt = pd.to_datetime(end_ymd)
    out: list[tuple[str, str, int]] = []
    for y in range(start_dt.year, end_dt.year + 1):
        s = max(start_dt, pd.Timestamp(year=y, month=1, day=1))
        e = min(end_dt, pd.Timestamp(year=y, month=12, day=31))
        out.append((s.strftime("%Y%m%d"), e.strftime("%Y%m%d"), y))
    return out


def fetch_paginated_daily_basic(
    pro: Any,
    start_date: str,
    end_date: str,
    *,
    fields: str = "ts_code,trade_date,pb",
    limit: int = 4000,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        df = pro.query(
            "daily_basic",
            start_date=start_date,
            end_date=end_date,
            fields=fields,
            offset=offset,
            limit=limit,
        )
        if df is None or df.empty:
            break
        frames.append(df)
        got = len(df)
        offset += got
        if got < limit:
            break
    if not frames:
        return pd.DataFrame(columns=fields.split(","))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Download PB from Tushare daily_basic only.")
    p.add_argument("--token-path", default="token")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--start", default="20050101")
    p.add_argument("--end", default="20260131")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--api-url", default="https://api.waditu.com/dataapi")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_yearly = out_dir / "raw" / "daily_basic"
    out_yearly.mkdir(parents=True, exist_ok=True)

    token = _read_token(Path(args.token_path).resolve())
    ts.set_token(token)
    pro = ts.pro_api(timeout=int(args.timeout))
    try:
        pro._DataApi__http_url = args.api_url  # type: ignore[attr-defined]
    except Exception:
        pass

    chunks = _year_chunks(_ymd(args.start), _ymd(args.end))
    all_frames: list[pd.DataFrame] = []

    for s, e, y in chunks:
        print(f"[info] downloading pb for {y}: {s}~{e}")
        df = fetch_paginated_daily_basic(pro, s, e, fields="ts_code,trade_date,pb")
        if not df.empty:
            df["trade_date"] = df["trade_date"].astype(str)
            df["pb"] = pd.to_numeric(df["pb"], errors="coerce")
            fp = out_yearly / f"daily_basic_pb_{y}.parquet"
            df.to_parquet(fp, index=False)
            all_frames.append(df)
            print(f"[ok] {fp} rows={len(df):,}")
        else:
            print(f"[warn] no data for {y}")

    if all_frames:
        all_df = pd.concat(all_frames, ignore_index=True)
        out_combined = out_dir / "raw" / "daily_basic_pb.parquet"
        out_combined.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_parquet(out_combined, index=False)
        print(f"[done] wrote combined: {out_combined} rows={len(all_df):,}")
    else:
        print("[done] no pb data downloaded")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

