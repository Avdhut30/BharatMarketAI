# src/backtest/select_best_threshold.py

import os
import pandas as pd

# Choose this cap for realism; you can tune later
DRAWDOWN_CAP = -0.20  # e.g. -20%


def main():
    path = os.path.join("reports", "threshold_grid_nolookahead.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run: python -m src.backtest.optimize_threshold_nolookahead")

    df = pd.read_csv(path)

    # Filter by drawdown cap
    feasible = df[df["MaxDrawdown"] >= DRAWDOWN_CAP].copy()

    if feasible.empty:
        print("❌ No threshold meets drawdown cap. Relax cap and retry.")
        print("Worst 10 drawdowns:")
        print(df.sort_values("MaxDrawdown").head(10).to_string(index=False))
        return

    best = feasible.sort_values("Sharpe", ascending=False).iloc[0]

    print("✅ Best threshold under drawdown cap")
    print(f"Drawdown cap: {DRAWDOWN_CAP}")
    print(best.to_string())

    # Optional: write a small text file for your config
    out_txt = os.path.join("reports", "selected_threshold.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Selected threshold (no-lookahead, drawdown cap {DRAWDOWN_CAP}): {best['threshold']}\n")
        f.write(best.to_string())
        f.write("\n")
    print("✅ Saved:", out_txt)


if __name__ == "__main__":
    main()
