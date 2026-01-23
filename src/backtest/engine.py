# src/backtest/engine.py

import os
import joblib
import pandas as pd
import numpy as np

from src.config import DATA_DIR, HORIZON_DAYS, ROUND_TRIP_COST, SL_ATR_MULT, P_UP_THRESHOLD
from src.backtest.strategy import StrategyConfig
from src.backtest.metrics import summary_stats

MODEL_PATH = os.path.join("reports", "lgbm_classifier.pkl")


def load_features_df() -> pd.DataFrame:
    parq = os.path.join(DATA_DIR, "features.parquet")
    csv = os.path.join(DATA_DIR, "features.csv")

    if os.path.exists(parq):
        df = pd.read_parquet(parq)
    elif os.path.exists(csv):
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError("Run build_features first: python -m src.features.build_features")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return df


def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train it first: python -m src.models.train")
    return joblib.load(MODEL_PATH)


def generate_predictions(df: pd.DataFrame, feature_cols: list, model) -> pd.DataFrame:
    """
    Add p_up column for every row.
    IMPORTANT: This uses the trained model, but features are all "past/current"
    so using them for prediction at time t is valid.
    """
    X = df[feature_cols]
    df = df.copy()
    df["p_up"] = model.predict_proba(X)[:, 1]
    return df


def backtest_long_only(df: pd.DataFrame, cfg: StrategyConfig):
    """
    Portfolio backtest:
    - At each day, pick the single best signal (highest p_up) that passes entry filters.
    - Hold for cfg.horizon_days OR hit ATR stop.
    - Apply round-trip cost.
    """
    df = df.copy()
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    # We'll simulate 1 position at a time (simpler, clean for portfolio projects)
    dates = sorted(df["Date"].unique())
    equity = []
    trades = []

    cash = 1.0
    in_pos = False
    entry_date = None
    entry_price = None
    entry_symbol = None
    entry_atr = None
    planned_exit_index = None

    # For faster lookup
    by_date = {d: df[df["Date"] == d] for d in dates}

    i = 0
    while i < len(dates):
        d = dates[i]
        day_rows = by_date[d]

        if not in_pos:
            # Entry selection: choose best symbol today
            candidates = day_rows.copy()

            if cfg.use_trend_200 and "trend_200" in candidates.columns:
                candidates = candidates[candidates["trend_200"] == 1]

            candidates = candidates[candidates["p_up"] >= cfg.p_up_entry]

            if not candidates.empty:
                pick = candidates.sort_values("p_up", ascending=False).iloc[0]
                entry_symbol = pick["Symbol"]
                entry_date = d
                entry_price = float(pick["Open"]) if "Open" in pick else float(pick["Close"])
                entry_atr = float(pick.get("atr_14", np.nan))
                # cost half at entry half at exit (approx)
                cash *= (1.0 - cfg.round_trip_cost / 2.0)

                # stop price
                stop_price = None
                if np.isfinite(entry_atr):
                    stop_price = entry_price - cfg.atr_stop_mult * entry_atr

                planned_exit_index = min(i + cfg.horizon_days, len(dates) - 1)
                in_pos = True

                trades.append({
                    "Symbol": entry_symbol,
                    "EntryDate": entry_date,
                    "EntryPrice": entry_price,
                    "StopPrice": stop_price,
                    "PlannedExitDate": dates[planned_exit_index],
                    "ExitDate": None,
                    "ExitPrice": None,
                    "Reason": None,
                    "PnL": None
                })

        else:
            # We are in a position: check stop or time exit
            # Get today's row for that symbol
            sym_row = day_rows[day_rows["Symbol"] == entry_symbol]
            if sym_row.empty:
                # No data for symbol today (rare), just skip day
                equity.append((d, cash))
                i += 1
                continue

            sym_row = sym_row.iloc[0]
            low = float(sym_row["Low"])
            close = float(sym_row["Close"])
            openp = float(sym_row["Open"])

            last_trade = trades[-1]
            stop_price = last_trade["StopPrice"]

            exit_now = False
            reason = None
            exit_price = None

            # ATR stop-loss (intraday)
            if stop_price is not None and low <= stop_price:
                exit_now = True
                reason = "STOP_ATR"
                # assume worst-case fill at stop price
                exit_price = stop_price

            # time exit
            if (not exit_now) and (i >= planned_exit_index):
                exit_now = True
                reason = "TIME_EXIT"
                exit_price = close  # exit at close on planned day

            # confidence drop exit (optional)
            if (not exit_now) and ("p_up" in sym_row) and (float(sym_row["p_up"]) < cfg.p_up_exit):
                exit_now = True
                reason = "MODEL_FLIP"
                exit_price = close

            if exit_now:
                # compute pnl
                pnl = (exit_price / entry_price) - 1.0
                cash *= (1.0 + pnl)
                # apply remaining half cost
                cash *= (1.0 - cfg.round_trip_cost / 2.0)

                last_trade["ExitDate"] = d
                last_trade["ExitPrice"] = float(exit_price)
                last_trade["Reason"] = reason
                last_trade["PnL"] = float(pnl)

                # reset position
                in_pos = False
                entry_date = entry_price = entry_symbol = entry_atr = None
                planned_exit_index = None

        equity.append((d, cash))
        i += 1

    equity_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    trades_df = pd.DataFrame(trades)

    stats = summary_stats(equity_df["Equity"])
    return equity_df, trades_df, stats


def main():
    df = load_features_df()
    bundle = load_model_bundle()
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = generate_predictions(df, feature_cols, model)

    cfg = StrategyConfig(
        horizon_days=HORIZON_DAYS,
        p_up_entry=P_UP_THRESHOLD,
        p_up_exit=0.50,
        use_trend_200=True,
        atr_stop_mult=SL_ATR_MULT,
        round_trip_cost=ROUND_TRIP_COST
    )

    equity_df, trades_df, stats = backtest_long_only(df, cfg)

    os.makedirs("reports", exist_ok=True)
    equity_path = os.path.join("reports", "equity_curve.csv")
    trades_path = os.path.join("reports", "trades.csv")

    equity_df.to_csv(equity_path)
    trades_df.to_csv(trades_path, index=False)

    print("✅ Backtest complete")
    print("Saved:", equity_path)
    print("Saved:", trades_path)
    print("\n📊 Performance:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:>12}: {v:.4f}")
        else:
            print(f"{k:>12}: {v}")


if __name__ == "__main__":
    main()
