# src/mf/data/amfi_nav.py

import os
import io
import pandas as pd
import requests

AMFI_URL = "https://portal.amfiindia.com/spages/NAVOpen.txt"
CACHE_PATH = "data_cache/amfi_nav.parquet"


def download_amfi_nav(force: bool = False) -> pd.DataFrame:
    os.makedirs("data_cache", exist_ok=True)

    if os.path.exists(CACHE_PATH) and not force:
        return pd.read_parquet(CACHE_PATH)

    r = requests.get(AMFI_URL, timeout=30)
    r.raise_for_status()

    raw = r.text

    # AMFI file is semi-colon separated; also contains category header lines.
    # We'll read it and then clean invalid rows.
    df = pd.read_csv(
        io.StringIO(raw),
        sep=";",
        skiprows=1,
        names=[
            "amfi_scheme_code",   # numeric code in AMFI
            "isin_growth",
            "isin_div",
            "scheme_name",
            "nav",
            "nav_date",
        ],
        dtype=str,
        engine="python",
    )

    # Clean dtypes
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["nav_date"] = pd.to_datetime(df["nav_date"], errors="coerce")

    # Drop garbage rows (headers / empty separators)
    df = df.dropna(subset=["nav", "nav_date", "scheme_name"]).copy()

    # ✅ KEY FIX:
    # Our app uses scheme_code from Groww which looks like INFxxxxx (ISIN).
    # So we will set scheme_code = isin_growth (fallback to isin_div).
    df["scheme_code"] = df["isin_growth"].fillna("").astype(str).str.strip()
    fallback = df["isin_div"].fillna("").astype(str).str.strip()

    df.loc[df["scheme_code"] == "", "scheme_code"] = fallback

    # Drop rows where both ISINs missing
    df = df[df["scheme_code"] != ""].copy()

    # Standardize column names to match metrics.py expectations
    df = df.rename(columns={"nav_date": "date"})  # metrics expects date
    df = df[["scheme_code", "scheme_name", "nav", "date"]].copy()

    # Save cache
    df.to_parquet(CACHE_PATH, index=False)
    return df


def get_latest_nav(force: bool = False) -> pd.DataFrame:
    """
    Returns latest NAV per scheme_code with columns:
      scheme_code, scheme_name, nav, date
    """
    df = download_amfi_nav(force=force)

    # Latest per scheme_code
    df = df.dropna(subset=["scheme_code", "nav", "date"]).copy()
    df = df.sort_values("date").groupby("scheme_code", as_index=False).tail(1)

    # Ensure expected schema
    df["scheme_code"] = df["scheme_code"].astype(str).str.strip()
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df[["scheme_code", "scheme_name", "nav", "date"]].reset_index(drop=True)
