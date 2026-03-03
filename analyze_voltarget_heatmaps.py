from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ffscore_backtest import BacktestConfig, build_monthly_panel, parse_args as parse_bt_args, performance_stats, run_backtest


def _calmar_from_daily(daily_bt: pd.DataFrame) -> float:
    s = performance_stats(daily_bt["net_ret"], periods_per_year=252)
    ann = float(s.get("annualized_return", np.nan))
    mdd = float(s.get("max_drawdown", np.nan))
    if not np.isfinite(ann) or not np.isfinite(mdd) or mdd >= 0:
        return np.nan
    return float(ann / abs(mdd))


def _pivot_metric(df: pd.DataFrame, metric: str, lookbacks: list[int], vol_targets: list[float]) -> np.ndarray:
    out = np.full((len(lookbacks), len(vol_targets)), np.nan, dtype="float64")
    for i, lb in enumerate(lookbacks):
        for j, vt in enumerate(vol_targets):
            m = df[(df["lookback_m"] == lb) & (np.isclose(df["vol_target_ann"], vt))]
            if not m.empty:
                out[i, j] = float(m.iloc[0][metric])
    return out


def _plot_heatmap(ax: plt.Axes, z: np.ndarray, lookbacks: list[int], vol_targets: list[float], title: str, cmap: str) -> None:
    im = ax.imshow(z, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("vol_target_ann")
    ax.set_ylabel("lookback_month")
    ax.set_xticks(np.arange(len(vol_targets)))
    ax.set_xticklabels([f"{x:.2f}" for x in vol_targets], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(lookbacks)))
    ax.set_yticklabels([str(x) for x in lookbacks])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def parse_args() -> argparse.Namespace:
    base = parse_bt_args([])
    p = argparse.ArgumentParser(description="Vol-target heatmap study using ffscore_backtest.run_backtest.")
    p.add_argument("--data-dir", type=str, default=str(base.data_dir))
    p.add_argument("--out-dir", type=str, default="reports/voltarget_heatmaps")
    p.add_argument("--start", type=str, default=base.start)
    p.add_argument("--end", type=str, default=base.end)
    p.add_argument("--top-quantile", type=float, default=float(base.top_quantile))
    p.add_argument("--min-tradable-days", type=int, default=int(base.min_tradable_days))
    p.add_argument("--min-score-names", type=int, default=int(base.min_score_names))
    p.add_argument("--cost-bps", type=float, default=float(base.cost_bps))
    p.add_argument("--use-pb-filter", action="store_true", default=bool(base.use_pb_filter))
    p.add_argument("--pb-quantile", type=float, default=float(base.pb_quantile))
    p.add_argument("--pb-proxy-path", type=str, default=str(base.pb_proxy_path))
    p.add_argument("--vol-target-ann", type=float, default=0.15)
    p.add_argument("--max-vol-lag", type=int, default=int(base.max_vol_lag))
    p.add_argument("--ffscore-carry-months", type=int, default=int(base.ffscore_carry_months))
    p.add_argument("--pb-carry-months", type=int, default=int(base.pb_carry_months))
    p.add_argument("--ffscore-full-score-only", action="store_true", default=bool(base.ffscore_full_score_only))
    p.add_argument("--rebalance-freq", type=str, default=str(base.rebalance_freq), choices=["daily", "weekly", "monthly"])
    p.add_argument("--lookback-months", type=str, default="3,6,9,12,18")
    p.add_argument("--vol-targets", type=str, default="0.08,0.10,0.12,0.15,0.18,0.21,0.25")
    p.add_argument("--max-leverage", type=float, default=1.0, help="Fixed leverage cap (not a heatmap axis).")
    p.add_argument("--output-png", type=str, default="reports/voltarget_heatmaps.png")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    lookbacks = [int(x.strip()) for x in str(args.lookback_months).split(",") if x.strip()]
    vol_targets = [float(x.strip()) for x in str(args.vol_targets).split(",") if x.strip()]
    if not lookbacks or not vol_targets:
        raise ValueError("lookback-months and vol-targets must be non-empty.")

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
        vol_lookback_m=12,
        max_leverage=float(args.max_leverage),
        ffscore_carry_months=int(args.ffscore_carry_months),
        pb_carry_months=int(args.pb_carry_months),
        ffscore_full_score_only=bool(args.ffscore_full_score_only),
        rebalance_freq=str(args.rebalance_freq),
    )
    panel = build_monthly_panel(cfg)

    rows: list[dict] = []
    for lb in lookbacks:
        for vt in vol_targets:
            cfg_i = replace(cfg, vol_lookback_m=int(lb), vol_target_ann=float(vt))
            _, daily_bt, _ = run_backtest(panel, cfg_i)
            s = performance_stats(daily_bt["net_ret"], periods_per_year=252)
            row = {
                "lookback_m": int(lb),
                "vol_target_ann": float(vt),
                "max_leverage": float(cfg_i.max_leverage),
                "annualized_return": float(s.get("annualized_return", np.nan)),
                "max_drawdown": float(s.get("max_drawdown", np.nan)),
                "calmar": float(_calmar_from_daily(daily_bt)),
            }
            rows.append(row)
            print(
                f"[grid] lookback_m={lb} vol_target_ann={vt:.2f} max_lev={cfg_i.max_leverage:.2f} "
                f"ann_ret={row['annualized_return']:.4%} mdd={row['max_drawdown']:.4%} calmar={row['calmar']:.3f}"
            )
    grid_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    metric_titles = [
        ("calmar", "Calmar", "viridis"),
        ("annualized_return", "Annualized Return", "viridis"),
        ("max_drawdown", "Max Drawdown", "magma_r"),
    ]
    for c_i, (metric, title, cmap) in enumerate(metric_titles):
        z = _pivot_metric(grid_df, metric, lookbacks, vol_targets)
        _plot_heatmap(
            axes[c_i],
            z,
            lookbacks,
            vol_targets,
            title=title,
            cmap=cmap,
        )
    fig.suptitle(
        "Vol Target Heatmaps (realized std only)\naxes: lookback_month x vol_target_ann",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = Path(args.output_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    out_csv = out_png.with_suffix(".csv")
    grid_df.to_csv(out_csv, index=False)
    print(f"[done] heatmap png: {out_png}")
    print(f"[done] grid metrics csv: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

