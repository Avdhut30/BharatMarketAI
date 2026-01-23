# src/backtest/oos_backtest_topk.py

import os
import pandas as pd
import numpy as np

from src.backtest.metrics import summary_stats
from src.backtest.strategy import StrategyConfig
from src.config import HORIZON_DAYS, ROUND_TRIP_COST, SL_ATR_MULT


def load_oos():
    path = os.path.join("reports", f"oos_predictions_{HORIZON_DAYS}d.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run: python -m src.models.walk_forward")
    oos = pd.read_csv(path)
    oos["Date"] = pd.to_datetime(oos["Date"], errors="coerce")
    oos = oos.dropna(subset=["Date"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return oos


def backtest_topk(oos: pd.DataFrame, cfg: StrategyConfig, top_k: int = 3):
    """
    Daily top-K portfolio (equal weight), holding for horizon days.
    Practical and usually improves stability.

    We simulate by:
    - each day select top K symbols meeting filters
    - each selected position is held for horizon days (no overlapping same symbol duplicates)
    - portfolio return = average of active positions' daily returns
    - apply a simple cost on entry/exit events
    """
    oos = oos.copy()
    oos["Date"] = pd.to_datetime(oos["Date"], errors="coerce")
    oos = oos.dropna(subset=["Date"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)

    dates = sorted(oos["Date"].unique())
    by_date = {d: oos[oos["Date"] == d] for d in dates}

    # active positions: symbol -> dict(entry_date, entry_price, exit_date)
    active = {}

    equity = []
    cash = 1.0

    for i, d in enumerate(dates):
        day = by_date[d]

        # 1) Close positions that reached horizon or model flip
        to_close = []
        for sym, pos in active.items():
            # if today >= exit date, close
            if d >= pos["exit_date"]:
                to_close.append(sym)
                continue

            # model flip exit (if symbol exists today)
            row = day[day["Symbol"] == sym]
            if not row.empty:
                p = float(row.iloc[0]["p_up_oos"])
                if p < cfg.p_up_exit:
                    to_close.append(sym)

        # apply exit cost
        if to_close:
            cash *= (1.0 - (cfg.round_trip_cost / 2.0))  # approx half cost on exits
            for sym in to_close:
                active.pop(sym, None)

        # 2) Select new entries (avoid duplicates already active)
        candidates = day.copy()
        if cfg.use_trend_200 and "trend_200" in candidates.columns:
            candidates = candidates[candidates["trend_200"] == 1]

        candidates = candidates[candidates["p_up_oos"] >= cfg.p_up_entry]
        candidates = candidates[~candidates["Symbol"].isin(active.keys())]

        candidates = candidates.sort_values("p_up_oos", ascending=False).head(max(0, top_k - len(active)))

        if not candidates.empty:
            cash *= (1.0 - (cfg.round_trip_cost / 2.0))  # approx half cost on entries

            for _, row in candidates.iterrows():
                sym = row["Symbol"]
                entry_price = float(row["Open"])
                exit_idx = min(i + cfg.horizon_days, len(dates) - 1)
                active[sym] = {
                    "entry_date": d,
                    "entry_price": entry_price,
                    "exit_date": dates[exit_idx],
                }

        # 3) Portfolio daily return from active positions (equal weight)
        if active:
            rets = []
            for sym in list(active.keys()):
                row = day[day["Symbol"] == sym]
                if row.empty:
                    continue
                # daily return approximated using Close-to-Close
                # (you can upgrade to Open/Close fills later)
                prev_day = dates[i - 1] if i > 0 else None
                if prev_day is None:
                    continue
                prev = by_date[prev_day]
                prev_row = prev[prev["Symbol"] == sym]
                if prev_row.empty:
                    continue

                close_today = float(row.iloc[0]["Close"])
                close_prev = float(prev_row.iloc[0]["Close"])
                if close_prev > 0:
                    rets.append(close_today / close_prev - 1.0)

            if rets:
                cash *= (1.0 + float(np.mean(rets)))

        equity.append((d, cash))

    equity_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    stats = summary_stats(equity_df["Equity"])
    return equity_df, stats


if __name__ == "__main__":
    from src.backtest.strategy import StrategyConfig

    oos = load_oos()

    # start with a reasonable threshold; after running optimize_threshold, set it here
    cfg = StrategyConfig(
        horizon_days=HORIZON_DAYS,
        p_up_entry=0.56,
        p_up_exit=0.50,
        use_trend_200=True,
        atr_stop_mult=SL_ATR_MULT,
        round_trip_cost=ROUND_TRIP_COST,
    )

    equity_df, stats = backtest_topk(oos, cfg, top_k=3)

    os.makedirs("reports", exist_ok=True)
    out = os.path.join("reports", "equity_curve_oos_topk.csv")
    equity_df.to_csv(out)

    print("✅ Top-K OOS backtest saved:", out)
    print("\n📊 Top-K OOS Performance:")
    for k, v in stats.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")
