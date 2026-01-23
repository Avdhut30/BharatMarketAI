# src/models/walk_forward.py

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.config import DATA_DIR, HORIZON_DAYS

TARGET_COL = f"target_up_{HORIZON_DAYS}d"


def load_features():
    parq = os.path.join(DATA_DIR, "features.parquet")
    csv = os.path.join(DATA_DIR, "features.csv")
    if os.path.exists(parq):
        df = pd.read_parquet(parq)
    elif os.path.exists(csv):
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError("Run: python -m src.features.build_features")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return df


def get_feature_cols(df: pd.DataFrame):
    drop = {"Date", "Symbol", TARGET_COL, f"target_ret_{HORIZON_DAYS}d"}
    return [c for c in df.columns if c not in drop]


def walk_forward_oos_predictions(
    df: pd.DataFrame,
    train_years: int = 3,
    test_months: int = 6,
    min_train_rows: int = 2000,
):
    """
    Rolling walk-forward:
    Train on last `train_years` years, test next `test_months` months.
    Outputs OOS predictions for every row in test windows.
    """
    feature_cols = get_feature_cols(df)

    start_date = df["Date"].min()
    end_date = df["Date"].max()

    # anchor starts after we have train_years of data
    anchor = start_date + pd.DateOffset(years=train_years)

    all_preds = []
    aucs = []

    while anchor < end_date:
        train_start = anchor - pd.DateOffset(years=train_years)
        train_end = anchor
        test_end = anchor + pd.DateOffset(months=test_months)

        train_df = df[(df["Date"] >= train_start) & (df["Date"] < train_end)].copy()
        test_df = df[(df["Date"] >= train_end) & (df["Date"] < test_end)].copy()

        if len(train_df) < min_train_rows or len(test_df) == 0:
            anchor = anchor + pd.DateOffset(months=test_months)
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[TARGET_COL].astype(int)

        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_COL].astype(int)

        model = lgb.LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        p = model.predict_proba(X_test)[:, 1]
        # Guard: if only one class appears in y_test, AUC is undefined
        auc = None
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, p)
            aucs.append(auc)

        out = test_df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume", "atr_14", "trend_200", TARGET_COL]].copy()
        out["p_up_oos"] = p
        out["wf_train_start"] = train_start
        out["wf_train_end"] = train_end
        out["wf_test_end"] = test_end
        all_preds.append(out)

        print(
            f"WF window train[{train_start.date()}..{(train_end - pd.Timedelta(days=1)).date()}] "
            f"test[{train_end.date()}..{(test_end - pd.Timedelta(days=1)).date()}] "
            f"rows train={len(train_df)} test={len(test_df)} "
            f"AUC={auc if auc is not None else 'NA'}"
        )

        anchor = anchor + pd.DateOffset(months=test_months)

    if not all_preds:
        raise RuntimeError("No walk-forward predictions created. Try reducing train_years or test_months.")

    oos = pd.concat(all_preds, ignore_index=True).sort_values(["Date", "Symbol"]).reset_index(drop=True)
    mean_auc = float(np.mean(aucs)) if len(aucs) else None
    return oos, mean_auc


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)

    df = load_features()
    oos, mean_auc = walk_forward_oos_predictions(df, train_years=3, test_months=6)

    out_path = os.path.join("reports", f"oos_predictions_{HORIZON_DAYS}d.csv")
    oos.to_csv(out_path, index=False)

    print("\n✅ Saved OOS predictions:", out_path)
    print("✅ Mean WF AUC:", mean_auc)
