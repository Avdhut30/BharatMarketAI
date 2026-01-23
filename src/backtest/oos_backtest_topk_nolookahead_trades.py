# src/backtest/oos_backtest_topk_nolookahead_trades.py

import os
import pandas as pd
import numpy as np

from src.backtest.metrics import summary_stats
from src.backtest.strategy import StrategyConfig
from src.config import HORIZON_DAYS, ROUND_TRIP_COST, SL_ATR_MULT, P_UP_THRESHOLD


def load_oos():
    path = os.path.join("reports", f"oos_predictions_{HORIZON_DAYS}d.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run: python -m src.models.walk_forward")
    oos = pd.read_csv(path)
    oos["Date"] = pd.to_datetime(oos["Date"], errors="coerce")
    oos = oos.dropna(subset=["Date"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return oos


def backtest_topk_nolookahead(oos: pd.DataFrame, cfg: StrategyConfig, top_k: int = 3):
    """
    NO-LOOKAHEAD top-K:
    - signal computed at day t close
    - enter at day t+1 open
    - hold horizon days (approx) or exit on model flip
    Equal-weight among active positions.
    Also logs trades with entry/exit.
    """
    oos = oos.sort_values(["Symbol", "Date"]).copy()

    # next day prices for fills
    oos["Open_next"] = oos.groupby("Symbol")["Open"].shift(-1)
    oos["Date_next"] = oos.groupby("Symbol")["Date"].shift(-1)

    oos = oos.dropna(subset=["Open_next", "Date_next"]).copy()
    oos = oos.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    dates = sorted(oos["Date"].unique())
    by_date = {d: oos[oos["Date"] == d] for d in dates}

    cash = 1.0
    equity = []

    # active positions: sym -> dict
    active = {}
    trades = []

    for i, d in enumerate(dates):
        day = by_date[d]

        # Exit rules first
        to_close = []
        for sym, pos in active.items():
            if d >= pos["planned_exit_date"]:
                to_close.append((sym, "TIME_EXIT"))
                continue

            # model flip exit
            row = day[day["Symbol"] == sym]
            if not row.empty:
                p = float(row.iloc[0]["p_up_oos"])
                if p < cfg.p_up_exit:
                    to_close.append((sym, "MODEL_FLIP"))

        if to_close:
            cash *= (1.0 - cfg.round_trip_cost / 2.0)  # exit cost
            for sym, reason in to_close:
                # close at today's Close
                row = day[day["Symbol"] == sym]
                if row.empty:
                    active.pop(sym, None)
                    continue

                close_price = float(row.iloc[0]["Close"])
                entry_price = float(active[sym]["entry_price"])
                pnl = close_price / entry_price - 1.0

                # update trade record
                for t in reversed(trades):
                    if t["Symbol"] == sym and t["ExitDate"] is None:
                        t["ExitDate"] = d
                        t["ExitPrice"] = close_price
                        t["Reason"] = reason
                        t["PnL_gross"] = pnl
                        t["PnL_net"] = (1.0 + pnl) * (1.0 - cfg.round_trip_cost / 2.0) - 1.0
                        break

                # remove from active
                active.pop(sym, None)

        # Entries
        slots = max(0, top_k - len(active))
        if slots > 0:
            candidates = day.copy()

            if cfg.use_trend_200 and "trend_200" in candidates.columns:
                candidates = candidates[candidates["trend_200"] == 1]

            candidates = candidates[candidates["p_up_oos"] >= cfg.p_up_entry]
            candidates = candidates[~candidates["Symbol"].isin(active.keys())]
            candidates = candidates.sort_values("p_up_oos", ascending=False).head(slots)

            if not candidates.empty:
                cash *= (1.0 - cfg.round_trip_cost / 2.0)  # entry cost

                for _, row in candidates.iterrows():
                    sym = row["Symbol"]
                    entry_date = row["Date_next"]
                    entry_price = float(row["Open_next"])

                    exit_i = min(i + cfg.horizon_days, len(dates) - 1)
                    planned_exit_date = dates[exit_i]

                    active[sym] = {
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "planned_exit_date": planned_exit_date,
                    }

                    trades.append({
                        "Symbol": sym,
                        "SignalDate": d,
                        "EntryDate": entry_date,
                        "EntryPrice": entry_price,
                        "PlannedExitDate": planned_exit_date,
                        "ExitDate": None,
                        "ExitPrice": None,
                        "Reason": None,
                        "PnL_gross": None,
                        "PnL_net": None,
                    })

        # Daily portfolio return (equal-weight, close-to-close for actives)
        if i > 0 and active:
            prev_day = dates[i - 1]
            prev = by_date[prev_day]
            rets = []

            for sym in active.keys():
                r_today = day[day["Symbol"] == sym]
                r_prev = prev[prev["Symbol"] == sym]
                if r_today.empty or r_prev.empty:
                    continue
                c1 = float(r_today.iloc[0]["Close"])
                c0 = float(r_prev.iloc[0]["Close"])
                if c0 > 0:
                    rets.append(c1 / c0 - 1.0)

            if rets:
                cash *= (1.0 + float(np.mean(rets)))

        equity.append((d, cash))

    equity_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    trades_df = pd.DataFrame(trades)

    stats = summary_stats(equity_df["Equity"])
    return equity_df, trades_df, stats


if __name__ == "__main__":
    oos = load_oos()

    cfg = StrategyConfig(
        horizon_days=HORIZON_DAYS,
        p_up_entry=P_UP_THRESHOLD,
        p_up_exit=0.50,
        use_trend_200=True,
        atr_stop_mult=SL_ATR_MULT,
        round_trip_cost=ROUND_TRIP_COST,
    )

    equity_df, trades_df, stats = backtest_topk_nolookahead(oos, cfg, top_k=3)

    os.makedirs("reports", exist_ok=True)
    equity_df.to_csv("reports/equity_curve_oos_topk_nolookahead.csv")
    trades_df.to_csv("reports/trades_topk_nolookahead.csv", index=False)

    print("✅ Generated Top-K No-lookahead backtest files")
    print("Saved: reports/equity_curve_oos_topk_nolookahead.csv")
    print("Saved: reports/trades_topk_nolookahead.csv")

    print("\n📊 Performance:")
    for k, v in stats.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")
