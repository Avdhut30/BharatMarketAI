# src/mf/portfolio/sip.py

import pandas as pd
import numpy as np
from datetime import datetime


def detect_sips(txn_df: pd.DataFrame, min_months=3):
    """
    Detect SIPs by:
    - Same scheme_code
    - Same investment amount
    - Repeating monthly pattern
    """
    df = txn_df.copy()
    df["month"] = df["date"].dt.to_period("M")

    sip_rows = []

    for (scode, amt), g in df.groupby(["scheme_code", "amount"]):
        months = g["month"].sort_values().unique()

        if len(months) < min_months:
            continue

        # check monthly continuity
        diffs = [(months[i] - months[i - 1]).n for i in range(1, len(months))]
        if all(d == 1 for d in diffs):
            sip_rows.append({
                "scheme_code": scode,
                "amount": amt,
                "start_month": months[0],
                "end_month": months[-1],
                "months": len(months),
            })

    return pd.DataFrame(sip_rows)


def sip_calendar(txn_df: pd.DataFrame):
    df = txn_df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    cal = (
        df.groupby(["year", "month"])
        .agg(
            sip_amount=("amount", "sum"),
            sip_count=("scheme_code", "count"),
        )
        .reset_index()
        .sort_values(["year", "month"])
    )
    return cal


def sip_health(sip_df):
    """
    Simple health classification:
    - Active: SIP present in last 2 months
    - Paused: Not present in last 2 months
    """
    today = datetime.today().to_period("M")
    statuses = []

    for _, r in sip_df.iterrows():
        last = r["end_month"]
        diff = (today - last).n

        if diff <= 2:
            status = "Active"
        else:
            status = "Paused"

        statuses.append(status)

    sip_df = sip_df.copy()
    sip_df["status"] = statuses
    return sip_df
