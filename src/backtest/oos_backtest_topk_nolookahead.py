# src/backtest/oos_backtest_topk_nolookahead.py

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


def add_next_day_fields(oos: pd.DataFrame) -> pd.DataFrame:
    oos = oos.sort_values(["Symbol", "Date"]).copy()
    oos["Open_next"] = oos.groupby("Symbol")["Open"].shift(-1)
    oos["Date_next"] = oos.groupby("Symbol")["Date"].shift(-1)
    # we keep Close for daily marking / exits
    return oos.dropna(subset=["Open_next", "Date_next"]).copy()


def backtest_topk_nolookahead(oos: pd.DataFrame, cfg: StrategyConfig, top_k: int = 3):
    """
    NO-LOOKAHEAD Top-K:
    - On signal date t: select top_k symbols
    - Enter at next day open (t+1)
    - Hold for horizon_days (based on signal-date stepping)
    - Equal weight across active holdings
    - Apply simple entry/exit cost
    """
    oos = add_next_day_fields(oos)
    oos = oos.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    dates = sorted(oos["Date"].unique())
    by_date = {d: oos[oos["Date"] == d] for d in dates}

    cash = 1.0
    equity = []

    # active positions: sym -> dict(entry_price, entry_date, exit_signal_date)
    active = {}

    for i, d in enumerate(dates):
        day = by_date[d]

        # 1) Close positions whose exit_signal_date <= today OR model flip
        to_close = []
        for sym, pos in active.items():
            if d >= pos["exit_signal_date"]:
                to_close.append(sym)
                continue
            # model flip (if symbol row exists today)
            r = day[day["Symbol"] == sym]
            if not r.empty:
                p = float(r.iloc[0]["p_up_oos"])
                if p < cfg.p_up_exit:
                    to_close.append(sym)

        if to_close:
            cash *= (1.0 - cfg.round_trip_cost / 2.0)
            for sym in to_close:
                active.pop(sym, None)

        # 2) Enter new positions based on today's signal, filled at next day open
        capacity = top_k - len(active)
        if capacity > 0:
            candidates = day.copy()

            if cfg.use_trend_200 and "trend_200" in candidates.columns:
                candidates = candidates[candidates["trend_200"] == 1]

            candidates = candidates[candidates["p_up_oos"] >= cfg.p_up_entry]
            candidates = candidates[~candidates["Symbol"].isin(active.keys())]
            candidates = candidates.sort_values("p_up_oos", ascending=False).head(capacity)

            if not candidates.empty:
                cash *= (1.0 - cfg.round_trip_cost / 2.0)
                exit_i = min(i + cfg.horizon_days, len(dates) - 1)
                exit_signal_date = dates[exit_i]

                for _, r in candidates.iterrows():
                    sym = r["Symbol"]
                    entry_price = float(r["Open_next"])
                    active[sym] = {
                        "entry_price": entry_price,
                        "entry_date": r["Date_next"],
                        "exit_signal_date": exit_signal_date,
                    }

        # 3) Mark-to-market daily using Close-to-Close for active symbols (approx)
        # (This is a reasonable portfolio approximation; can be upgraded later)
        if i > 0 and active:
            prev = by_date[dates[i - 1]]
            daily_rets = []
            for sym in list(active.keys()):
                r_today = day[day["Symbol"] == sym]
                r_prev = prev[prev["Symbol"] == sym]
                if r_today.empty or r_prev.empty:
                    continue
                c_today = float(r_today.iloc[0]["Close"])
                c_prev = float(r_prev.iloc[0]["Close"])
                if c_prev > 0:
                    daily_rets.append(c_today / c_prev - 1.0)

            if daily_rets:
                cash *= (1.0 + float(np.mean(daily_rets)))

        equity.append((d, cash))

    equity_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    stats = summary_stats(equity_df["Equity"])
    return equity_df, stats


if __name__ == "__main__":
    oos = load_oos()

    cfg = StrategyConfig(
        horizon_days=HORIZON_DAYS,
        p_up_entry=0.52,   # set after running optimize_threshold_nolookahead
        p_up_exit=0.50,
        use_trend_200=True,
        atr_stop_mult=SL_ATR_MULT,
        round_trip_cost=ROUND_TRIP_COST,
    )

    equity_df, stats = backtest_topk_nolookahead(oos, cfg, top_k=3)

    os.makedirs("reports", exist_ok=True)
    out = os.path.join("reports", "equity_curve_oos_topk_nolookahead.csv")
    equity_df.to_csv(out)

    print("✅ Top-K No-Lookahead saved:", out)
    print("\n📊 Top-K No-Lookahead Performance:")
    for k, v in stats.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")
