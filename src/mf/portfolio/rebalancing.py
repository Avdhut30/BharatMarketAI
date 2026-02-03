# src/mf/portfolio/rebalancing.py

import pandas as pd
import numpy as np


def allocation_table(summary: pd.DataFrame, by_col: str) -> pd.DataFrame:
    """
    summary must include:
      - current_value
      - invested
      - gain
      - by_col (e.g. asset_class, goal)
    """
    g = (
        summary.groupby(by_col)
        .agg(
            invested=("invested", "sum"),
            current_value=("current_value", "sum"),
            gain=("gain", "sum"),
        )
        .reset_index()
    )
    total = g["current_value"].sum()
    g["weight"] = np.where(total > 0, g["current_value"] / total, 0.0)
    g = g.sort_values("current_value", ascending=False)
    return g


def compute_drift(current_alloc: pd.DataFrame, target: dict, col_name="asset_class"):
    """
    current_alloc from allocation_table(...), has columns [col_name, weight]
    target: {"Equity":0.7, "Debt":0.3}
    returns drift table
    """
    target_rows = []
    for k, v in target.items():
        target_rows.append({col_name: k, "target_weight": float(v)})

    tgt = pd.DataFrame(target_rows)

    out = tgt.merge(current_alloc[[col_name, "weight"]], on=col_name, how="left")
    out["weight"] = out["weight"].fillna(0.0)
    out["drift"] = out["weight"] - out["target_weight"]
    out = out.sort_values("drift", ascending=False)
    return out


def suggestion_add_next(drift_df: pd.DataFrame):
    """
    Suggest which asset class to add money to (most negative drift).
    """
    if drift_df.empty:
        return None
    row = drift_df.sort_values("drift").iloc[0]
    if row["drift"] < 0:
        return str(row.iloc[0])  # asset class
    return None
