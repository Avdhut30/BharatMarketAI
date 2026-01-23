# src/backtest/oos_backtest.py

import os
import pandas as pd
import numpy as np

from src.backtest.metrics import summary_stats
from src.backtest.strategy import StrategyConfig
from src.config import HORIZON_DAYS, ROUND_TRIP_COST, SL_ATR_MULT, P_UP_THRESHOLD


def backtest_oos(oos: pd.DataFrame, cfg: StrategyConfig):
    """
    Same logic as earlier: pick 1 best trade per day, but using p_up_oos.
    """
    oos = oos.copy()
    oos["Date"] = pd.to_datetime(oos["Date"], errors="coerce")
    oos = oos.dropna(subset=["Date"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)

    dates = sorted(oos["Date"].unique())
    by_date = {d: oos[oos["Date"] == d] for d in dates}

    cash = 1.0
    in_pos = False
    entry_symbol = None
    entry_price = None
    entry_atr = None
    planned_exit_i = None

    equity = []
    trades = []

    i = 0
    while i < len(dates):
        d = dates[i]
        day = by_date[d]

        if not in_pos:
            candidates = day.copy()

            if cfg.use_trend_200 and "trend_200" in candidates.columns:
                candidates = candidates[candidates["trend_200"] == 1]

            candidates = candidates[candidates["p_up_oos"] >= cfg.p_up_entry]

            if not candidates.empty:
                pick = candidates.sort_values("p_up_oos", ascending=False).iloc[0]
                entry_symbol = pick["Symbol"]
                entry_price = float(pick["Open"])
                entry_atr = float(pick["atr_14"]) if "atr_14" in pick else np.nan

                cash *= (1.0 - cfg.round_trip_cost / 2.0)

                stop = None
                if np.isfinite(entry_atr):
                    stop = entry_price - cfg.atr_stop_mult * entry_atr

                planned_exit_i = min(i + cfg.horizon_days, len(dates) - 1)

                trades.append({
                    "Symbol": entry_symbol,
                    "EntryDate": d,
                    "EntryPrice": entry_price,
                    "StopPrice": stop,
                    "PlannedExitDate": dates[planned_exit_i],
                    "ExitDate": None,
                    "ExitPrice": None,
                    "Reason": None,
                    "PnL": None,
                })

                in_pos = True

        else:
            row = day[day["Symbol"] == entry_symbol]
            if row.empty:
                equity.append((d, cash))
                i += 1
                continue

            row = row.iloc[0]
            low = float(row["Low"])
            close = float(row["Close"])
            p = float(row["p_up_oos"])

            last = trades[-1]
            stop = last["StopPrice"]

            exit_now = False
            reason = None
            exit_price = None

            if stop is not None and low <= stop:
                exit_now = True
                reason = "STOP_ATR"
                exit_price = stop

            if (not exit_now) and (i >= planned_exit_i):
                exit_now = True
                reason = "TIME_EXIT"
                exit_price = close

            if (not exit_now) and (p < cfg.p_up_exit):
                exit_now = True
                reason = "MODEL_FLIP"
                exit_price = close

            if exit_now:
                pnl = (exit_price / entry_price) - 1.0
                cash *= (1.0 + pnl)
                cash *= (1.0 - cfg.round_trip_cost / 2.0)

                last["ExitDate"] = d
                last["ExitPrice"] = float(exit_price)
                last["Reason"] = reason
                last["PnL"] = float(pnl)

                in_pos = False
                entry_symbol = entry_price = entry_atr = None
                planned_exit_i = None

        equity.append((d, cash))
        i += 1

    equity_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    trades_df = pd.DataFrame(trades)
    stats = summary_stats(equity_df["Equity"])
    return equity_df, trades_df, stats


if __name__ == "__main__":
    path = os.path.join("reports", f"oos_predictions_{HORIZON_DAYS}d.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run: python -m src.models.walk_forward")

    oos = pd.read_csv(path)

    cfg = StrategyConfig(
        horizon_days=HORIZON_DAYS,
        p_up_entry=P_UP_THRESHOLD,
        p_up_exit=0.50,
        use_trend_200=True,
        atr_stop_mult=SL_ATR_MULT,
        round_trip_cost=ROUND_TRIP_COST,
    )

    equity_df, trades_df, stats = backtest_oos(oos, cfg)

    os.makedirs("reports", exist_ok=True)
    equity_df.to_csv(os.path.join("reports", "equity_curve_oos.csv"))
    trades_df.to_csv(os.path.join("reports", "trades_oos.csv"), index=False)

    print("✅ OOS Backtest complete")
    print("Saved: reports/equity_curve_oos.csv")
    print("Saved: reports/trades_oos.csv")
    print("\n📊 OOS Performance:")
    for k, v in stats.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")
