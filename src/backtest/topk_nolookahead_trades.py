# src/backtest/topk_nolookahead_trades.py

import numpy as np
import pandas as pd

from src.backtest.metrics import summary_stats
from src.backtest.strategy import StrategyConfig


def add_next_day_fields(oos: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Open_next and Date_next for next-day execution (no-lookahead).
    """
    oos = oos.sort_values(["Symbol", "Date"]).copy()
    oos["Open_next"] = oos.groupby("Symbol")["Open"].shift(-1)
    oos["Date_next"] = oos.groupby("Symbol")["Date"].shift(-1)
    return oos.dropna(subset=["Open_next", "Date_next"]).copy()


def backtest_topk_nolookahead_with_trades(
    oos: pd.DataFrame,
    cfg: StrategyConfig,
    top_k: int = 3,
):
    """
    NO-LOOKAHEAD Top-K portfolio backtest with TRADE + HOLDINGS logs.

    Rules:
    - Signal computed on day t (row Date = t)
    - Enter at next day open (Open_next, Date_next)
    - Hold until:
        * TIME_EXIT: signal-date index reaches horizon_days
        * MODEL_FLIP: p_up_oos < cfg.p_up_exit
    - Equal-weight across active positions.
    - Portfolio equity is mark-to-market using Close-to-Close of active holdings.
    - Costs: apply half round-trip on entry batch + half on exit batch at portfolio level,
      and also annotate per-trade costs for transparency (approx allocation).

    Returns:
      equity_df: Date index, Equity
      trades_df: one row per trade (per symbol)
      holdings_df: daily holdings snapshot (symbols + weights)
      stats: summary_stats dict
    """
    oos = oos.copy()
    oos["Date"] = pd.to_datetime(oos["Date"], errors="coerce")
    oos = oos.dropna(subset=["Date"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)

    oos = add_next_day_fields(oos)
    oos = oos.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    dates = sorted(oos["Date"].unique())
    by_date = {d: oos[oos["Date"] == d] for d in dates}

    cash = 1.0
    equity = []
    holdings_rows = []
    trades = []

    # active positions: sym -> dict
    active = {}

    def snapshot_holdings(d):
        if not active:
            holdings_rows.append({"Date": d, "n_positions": 0, "symbols": "", "weights": ""})
            return
        syms = sorted(active.keys())
        w = 1.0 / len(syms)
        weights = ",".join([f"{w:.4f}"] * len(syms))
        holdings_rows.append(
            {"Date": d, "n_positions": len(syms), "symbols": ",".join(syms), "weights": weights}
        )

    for i, d in enumerate(dates):
        day = by_date[d]

        # ---------- exits ----------
        to_close = []
        for sym, pos in active.items():
            # time exit based on signal-date index
            if d >= pos["exit_signal_date"]:
                to_close.append((sym, "TIME_EXIT"))
                continue

            # model flip exit (if symbol exists today)
            r = day[day["Symbol"] == sym]
            if not r.empty:
                p = float(r.iloc[0]["p_up_oos"])
                if p < cfg.p_up_exit:
                    to_close.append((sym, "MODEL_FLIP"))

        if to_close:
            # portfolio-level exit cost (approx)
            cash *= (1.0 - cfg.round_trip_cost / 2.0)

            for sym, reason in to_close:
                r = day[day["Symbol"] == sym]
                if r.empty:
                    # can't price it today; close using last available close from earlier dates is complex
                    # so we skip closing if missing (rare). keep it active.
                    continue

                exit_price = float(r.iloc[0]["Close"])
                pos = active.get(sym)
                if not pos:
                    continue

                entry_price = float(pos["entry_price"])
                pnl_gross = (exit_price / entry_price) - 1.0

                # approximate per-trade cost allocation: half on entry + half on exit
                pnl_net = (1.0 + pnl_gross) * (1.0 - cfg.round_trip_cost) - 1.0

                trades.append(
                    {
                        "Symbol": sym,
                        "SignalDate": pos["signal_date"],
                        "EntryDate": pos["entry_date"],
                        "EntryPrice": entry_price,
                        "ExitDate": d,
                        "ExitPrice": exit_price,
                        "Reason": reason,
                        "PnL_gross": pnl_gross,
                        "PnL_net_approx": pnl_net,
                        "HoldDays_signal": (d - pos["signal_date"]).days,
                        "TopK": top_k,
                        "EntryThreshold": cfg.p_up_entry,
                        "ExitThreshold": cfg.p_up_exit,
                        "Cost": cfg.round_trip_cost,
                    }
                )

                active.pop(sym, None)

        # ---------- entries ----------
        capacity = top_k - len(active)
        if capacity > 0:
            candidates = day.copy()

            if cfg.use_trend_200 and "trend_200" in candidates.columns:
                candidates = candidates[candidates["trend_200"] == 1]

            candidates = candidates[candidates["p_up_oos"] >= cfg.p_up_entry]
            candidates = candidates[~candidates["Symbol"].isin(active.keys())]
            candidates = candidates.sort_values("p_up_oos", ascending=False).head(capacity)

            if not candidates.empty:
                # portfolio-level entry cost (approx)
                cash *= (1.0 - cfg.round_trip_cost / 2.0)

                exit_i = min(i + cfg.horizon_days, len(dates) - 1)
                exit_signal_date = dates[exit_i]

                for _, r in candidates.iterrows():
                    sym = r["Symbol"]
                    active[sym] = {
                        "signal_date": d,
                        "entry_date": pd.to_datetime(r["Date_next"]),
                        "entry_price": float(r["Open_next"]),
                        "exit_signal_date": exit_signal_date,
                        "p_entry": float(r["p_up_oos"]),
                    }

        # ---------- mark-to-market daily ----------
        # Use Close-to-Close mean return of active holdings
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
        snapshot_holdings(d)

    equity_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    holdings_df = pd.DataFrame(holdings_rows)
    trades_df = pd.DataFrame(trades)

    stats = summary_stats(equity_df["Equity"])
    return equity_df, trades_df, holdings_df, stats
