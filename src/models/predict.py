# src/models/predict.py

import os
import joblib
import pandas as pd

from src.config import DATA_DIR, HORIZON_DAYS

MODEL_DIR = "reports"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_classifier.pkl")


def load_data():
    path_parq = os.path.join(DATA_DIR, "features.parquet")
    path_csv = os.path.join(DATA_DIR, "features.csv")

    if os.path.exists(path_parq):
        df = pd.read_parquet(path_parq)
    elif os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
    else:
        raise FileNotFoundError("features.parquet or features.csv not found. Run build_features first.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return df


def predict_latest():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run: python -m src.models.train")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    target_col = bundle["target_col"]

    df = load_data()

    # Use the latest available date per symbol
    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date].copy()

    X = latest[feature_cols]
    latest["p_up"] = model.predict_proba(X)[:, 1]

    # Sort by confidence
    latest = latest.sort_values("p_up", ascending=False)

    out_path = os.path.join("reports", f"latest_predictions_{latest_date.date()}.csv")
    latest[["Date", "Symbol", "p_up", "Close", "rsi_14", "trend_200"]].to_csv(out_path, index=False)

    print("✅ Latest date:", latest_date.date())
    print("✅ Saved predictions:", out_path)
    print("\nTop 10 signals:")
    print(latest[["Symbol", "p_up", "Close", "rsi_14", "trend_200"]].head(10).to_string(index=False))


if __name__ == "__main__":
    predict_latest()
