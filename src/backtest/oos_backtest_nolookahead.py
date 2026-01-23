# src/backtest/oos_backtest_nolookahead.py

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


def build_next_day_open_map(oos: pd.DataFrame):
    """
    For each (Date, Symbol), get NEXT trading day's Open/Low/Close.
    This lets us enter at next day's open using today's signal.
    """
    oos = oos.sort_values(["Symbol", "Date"]).copy()
    oos["Open_next"] = oos.groupby("Symbol")["Open"].shift(-1)
    oos["Low_next"] = oos.groupby("Symbol")["Low"].shift(-1)
    oos["Close_next"] = oos.groupby("Symbol")["Close"].shift(-1)
    oos["Date_next"] = oos.groupby("Symbol")["Date"].shift(-1)
    return oos


def backtest_single_position_nolookahead(oos: pd.DataFrame, cfg: StrategyConfig):
    """
    NO-LOOKAHEAD BACKTEST:
    - Use signal at day t
    - Enter at OPEN of day t+1
    - Exit at CLOSE of exit day (or stop)
    One position at a time, picks best signal each day.
    """
    oos = build_next_day_open_map(oos)

    # Remove rows where we don't have next day prices (can't trade)
    oos = oos.dropna(subset=["Open_next", "Date_next"]).copy()
    oos = oos.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    dates = sorted(oos["Date"].unique())
    by_date = {d: oos[oos["Date"] == d] for d in dates}

    cash = 1.0
    in_pos = False
    entry_symbol = None
    entry_price = None
    entry_date = None
    entry_atr = None
    stop_price = None
    exit_date = None

    equity = []
    trades = []

    # We'll track equity on signal dates (t), even though fills happen t+1
    for i, d in enumerate(dates):
        day = by_date[d]

        if not in_pos:
            candidates = day.copy()

            if cfg.use_trend_200 and "trend_200" in candidates.columns:
                candidates = candidates[candidates["trend_200"] == 1]

            candidates = candidates[candidates["p_up_oos"] >= cfg.p_up_entry]

            if not candidates.empty:
                pick = candidates.sort_values("p_up_oos", ascending=False).iloc[0]

                # enter next day open
                entry_symbol = pick["Symbol"]
                entry_date = pick["Date_next"]
                entry_price = float(pick["Open_next"])
                entry_atr = float(pick["atr_14"]) if "atr_14" in pick else np.nan

                cash *= (1.0 - cfg.round_trip_cost / 2.0)

                stop_price = None
                if np.isfinite(entry_atr):
                    stop_price = entry_price - cfg.atr_stop_mult * entry_atr

                # exit date is horizon trading days after ENTRY DATE (approx by counting signal dates)
                # We'll approximate by using "dates" list: find index of entry_date in dates if present
                # If not present, we still hold until horizon days from the signal index.
                # Simple approach: use signal index i and add horizon_days
                exit_i = min(i + cfg.horizon_days, len(dates) - 1)
                exit_date = dates[exit_i]

                trades.append({
                    "Symbol": entry_symbol,
                    "SignalDate": d,
                    "EntryDate": entry_date,
                    "EntryPrice": entry_price,
                    "StopPrice": stop_price,
                    "PlannedExitDate": exit_date,
                    "ExitDate": None,
                    "ExitPrice": None,
                    "Reason": None,
                    "PnL": None
                })

                in_pos = True

        else:
            # Manage open position using the current day's market data for that symbol
            row = day[day["Symbol"] == entry_symbol]
            if not row.empty:
                row = row.iloc[0]
                # We check stop using NEXT day prices relative to signal date
                # But since we are tracking daily by signal date, use current day's actual Low/Close
                low = float(row["Low"])
                close = float(row["Close"])
                p = float(row["p_up_oos"])

                exit_now = False
                reason = None
                exit_price = None

                if stop_price is not None and low <= stop_price:
                    exit_now = True
                    reason = "STOP_ATR"
                    exit_price = stop_price

                if (not exit_now) and (d >= exit_date):
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

                    trades[-1]["ExitDate"] = d
                    trades[-1]["ExitPrice"] = float(exit_price)
                    trades[-1]["Reason"] = reason
                    trades[-1]["PnL"] = float(pnl)

                    in_pos = False
                    entry_symbol = entry_price = entry_date = entry_atr = stop_price = exit_date = None

        equity.append((d, cash))

    equity_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    trades_df = pd.DataFrame(trades)
    stats = summary_stats(equity_df["Equity"])
    return equity_df, trades_df, stats


if __name__ == "__main__":
    oos = load_oos()

    # Use your best threshold from grid, but we will re-test it honestly now
    cfg = StrategyConfig(
        horizon_days=HORIZON_DAYS,
        p_up_entry=0.52,
        p_up_exit=0.50,
        use_trend_200=True,
        atr_stop_mult=SL_ATR_MULT,
        round_trip_cost=ROUND_TRIP_COST
    )

    equity_df, trades_df, stats = backtest_single_position_nolookahead(oos, cfg)

    os.makedirs("reports", exist_ok=True)
    equity_df.to_csv("reports/equity_curve_oos_nolookahead.csv")
    trades_df.to_csv("reports/trades_oos_nolookahead.csv", index=False)

    print("✅ OOS No-Lookahead backtest complete")
    print("Saved: reports/equity_curve_oos_nolookahead.csv")
    print("Saved: reports/trades_oos_nolookahead.csv")
    print("\n📊 OOS No-Lookahead Performance:")
    for k, v in stats.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")
