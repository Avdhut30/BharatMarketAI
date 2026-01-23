# src/backtest/robustness_tests.py

import os
import numpy as np
import pandas as pd

from src.backtest.oos_backtest_topk_nolookahead import load_oos, backtest_topk_nolookahead
from src.backtest.strategy import StrategyConfig
from src.config import HORIZON_DAYS, SL_ATR_MULT


def run():
    oos = load_oos()

    thresholds = [0.52, 0.55, 0.58]
    topks = [1, 3, 5]
    costs = [0.0005, 0.0010, 0.0020]  # 0.05%, 0.10%, 0.20% round trip

    results = []

    for th in thresholds:
        for k in topks:
            for c in costs:
                cfg = StrategyConfig(
                    horizon_days=HORIZON_DAYS,
                    p_up_entry=th,
                    p_up_exit=0.50,
                    use_trend_200=True,
                    atr_stop_mult=SL_ATR_MULT,
                    round_trip_cost=c,
                )

                equity_df, stats = backtest_topk_nolookahead(oos, cfg, top_k=k)
                results.append({
                    "threshold": th,
                    "top_k": k,
                    "round_trip_cost": c,
                    "CAGR": stats["CAGR"],
                    "Sharpe": stats["Sharpe"],
                    "MaxDrawdown": stats["MaxDrawdown"],
                    "TotalReturn": stats["TotalReturn"],
                })

    res = pd.DataFrame(results).sort_values(["Sharpe", "CAGR"], ascending=False).reset_index(drop=True)
    os.makedirs("reports", exist_ok=True)
    out = os.path.join("reports", "robustness_grid.csv")
    res.to_csv(out, index=False)

    print("✅ Saved:", out)
    print("\nTop 15 configs by Sharpe:")
    print(res.head(15).to_string(index=False))


if __name__ == "__main__":
    run()
