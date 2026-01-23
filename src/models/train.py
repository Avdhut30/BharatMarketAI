# src/models/train.py

import os
import joblib
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from src.config import DATA_DIR, HORIZON_DAYS

MODEL_DIR = "reports"
TARGET_COL = f"target_up_{HORIZON_DAYS}d"


def load_features():
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


def get_feature_columns(df: pd.DataFrame):
    drop_cols = {"Date", "Symbol", TARGET_COL, f"target_ret_{HORIZON_DAYS}d"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return feature_cols


def time_split(df: pd.DataFrame, split_date: str = "2023-01-01"):
    """
    Train on data < split_date, validate on data >= split_date
    """
    split_date = pd.to_datetime(split_date)
    train_df = df[df["Date"] < split_date].copy()
    valid_df = df[df["Date"] >= split_date].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError(
            f"Bad split. Train rows: {len(train_df)}, Valid rows: {len(valid_df)}. "
            "Change split_date to something inside your dataset date range."
        )

    return train_df, valid_df


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_features()
    feature_cols = get_feature_columns(df)

    # choose a split date that gives enough training history
    train_df, valid_df = time_split(df, split_date="2023-01-01")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[TARGET_COL].astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    # Evaluate
    p_valid = model.predict_proba(X_valid)[:, 1]
    pred_valid = (p_valid >= 0.5).astype(int)

    acc = accuracy_score(y_valid, pred_valid)
    auc = roc_auc_score(y_valid, p_valid)

    print("✅ Validation Accuracy:", round(acc, 4))
    print("✅ Validation ROC-AUC:", round(auc, 4))
    print("\nClassification Report:\n", classification_report(y_valid, pred_valid))

    # Save model bundle
    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "split_date": "2023-01-01"
    }

    out_path = os.path.join(MODEL_DIR, "lgbm_classifier.pkl")
    joblib.dump(bundle, out_path)
    print(f"✅ Saved model bundle: {out_path}")


if __name__ == "__main__":
    train()
