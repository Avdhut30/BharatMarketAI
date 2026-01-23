# src/backtest/robustness_grid.py
import os
import json
import numpy as np
import pandas as pd

from src.backtest.strategy import StrategyConfig
from src.backtest.metrics import summary_stats
from src.config import HORIZON_DAYS
from src.backtest.oos_backtest_topk_nolookahead_trades import load_oos, backtest_topk_nolookahead


def score_row(stats: dict) -> float:
    """
    Robust score: favor Sharpe and CAGR, penalize drawdown.
    This avoids selecting extreme high-dd configs.
    """
    sharpe = stats["Sharpe"]
    cagr = stats["CAGR"]
    mdd = abs(stats["MaxDrawdown"])
    # penalty term
    return float(sharpe + 0.3 * cagr - 0.8 * mdd)


def main():
    oos = load_oos()

    thresholds = np.round(np.arange(0.50, 0.61, 0.01), 2)   # 0.50..0.60
    topks = [1, 3, 5]
    costs = [0.0005, 0.0010, 0.0020]
    exits = [0.48, 0.50, 0.52]
    trend_flags = [True, False]

    rows = []
    total = len(thresholds) * len(topks) * len(costs) * len(exits) * len(trend_flags)
    n = 0

    for th in thresholds:
        for k in topks:
            for cost in costs:
                for exit_th in exits:
                    for trend in trend_flags:
                        n += 1
                        cfg = StrategyConfig(
                            horizon_days=HORIZON_DAYS,
                            p_up_entry=float(th),
                            p_up_exit=float(exit_th),
                            use_trend_200=bool(trend),
                            atr_stop_mult=1.5,
                            round_trip_cost=float(cost),
                        )

                        equity_df, trades_df, stats = backtest_topk_nolookahead(oos, cfg, top_k=int(k))

                        rows.append({
                            "threshold": float(th),
                            "top_k": int(k),
                            "round_trip_cost": float(cost),
                            "exit_threshold": float(exit_th),
                            "trend_filter": bool(trend),
                            "CAGR": float(stats["CAGR"]),
                            "Sharpe": float(stats["Sharpe"]),
                            "MaxDrawdown": float(stats["MaxDrawdown"]),
                            "TotalReturn": float(stats["TotalReturn"]),
                            "Days": int(stats["Days"]),
                            "Score": score_row(stats),
                        })

                        if n % 20 == 0:
                            print(f"Progress {n}/{total}")

    res = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

    os.makedirs("reports", exist_ok=True)
    grid_path = os.path.join("reports", "robustness_grid.csv")
    res.to_csv(grid_path, index=False)
    print("✅ Saved:", grid_path)

    best = res.iloc[0].to_dict()
    cfg_out = {
        "threshold": best["threshold"],
        "top_k": best["top_k"],
        "round_trip_cost": best["round_trip_cost"],
        "exit_threshold": best["exit_threshold"],
        "trend_filter": best["trend_filter"],
        "horizon_days": HORIZON_DAYS,
        "atr_stop_mult": 1.5,
    }

    json_path = os.path.join("reports", "selected_config.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)

    # keep backward compatibility with your UI threshold reader
    with open(os.path.join("reports", "selected_threshold.txt"), "w", encoding="utf-8") as f:
        f.write(f"threshold {best['threshold']}\n")

    print("✅ Best config saved:", json_path)
    print("🏆 Best config:", cfg_out)

    print("\nTop 10 configs:")
    print(res.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
