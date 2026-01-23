# src/backtest/optimize_threshold_nolookahead.py

import os
import numpy as np
import pandas as pd

from src.backtest.oos_backtest_nolookahead import load_oos, backtest_single_position_nolookahead
from src.backtest.strategy import StrategyConfig
from src.config import HORIZON_DAYS, ROUND_TRIP_COST, SL_ATR_MULT


def run_grid():
    oos = load_oos()

    thresholds = np.round(np.arange(0.50, 0.70, 0.01), 2)
    results = []

    for th in thresholds:
        cfg = StrategyConfig(
            horizon_days=HORIZON_DAYS,
            p_up_entry=float(th),
            p_up_exit=0.50,
            use_trend_200=True,
            atr_stop_mult=SL_ATR_MULT,
            round_trip_cost=ROUND_TRIP_COST,
        )

        equity_df, trades_df, stats = backtest_single_position_nolookahead(oos, cfg)
        results.append({
            "threshold": th,
            "CAGR": stats["CAGR"],
            "Sharpe": stats["Sharpe"],
            "MaxDrawdown": stats["MaxDrawdown"],
            "TotalReturn": stats["TotalReturn"],
            "Trades": int(trades_df["ExitDate"].notna().sum()) if "ExitDate" in trades_df.columns else len(trades_df),
        })

    res = pd.DataFrame(results).sort_values("Sharpe", ascending=False).reset_index(drop=True)
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "threshold_grid_nolookahead.csv")
    res.to_csv(out_path, index=False)

    print("✅ Saved:", out_path)
    print("\nTop 10 by Sharpe (No-Lookahead):")
    print(res.head(10).to_string(index=False))

    print("\nTop 10 by CAGR (No-Lookahead):")
    print(res.sort_values("CAGR", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    run_grid()
