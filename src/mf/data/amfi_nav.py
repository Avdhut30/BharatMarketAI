# src/mf/data/amfi_nav.py

import os
import io
import pandas as pd
import requests

AMFI_URL = "https://portal.amfiindia.com/spages/NAVOpen.txt"

CACHE_DIR = "data_cache"
CACHE_PATH = os.path.join(CACHE_DIR, "amfi_nav.csv.gz")  # ✅ changed from parquet


def _read_cache() -> pd.DataFrame:
    if os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH, compression="gzip", dtype=str)
    return pd.DataFrame()


def _write_cache(df: pd.DataFrame) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False, compression="gzip")


def download_amfi_nav(force: bool = False) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not force:
        cached = _read_cache()
        if not cached.empty:
            # restore types
            cached["nav"] = pd.to_numeric(cached["nav"], errors="coerce")
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached = cached.dropna(subset=["scheme_code", "nav", "date"])
            return cached

    r = requests.get(AMFI_URL, timeout=30)
    r.raise_for_status()

    raw = r.text

    df = pd.read_csv(
        io.StringIO(raw),
        sep=";",
        skiprows=1,
        names=[
            "amfi_scheme_code",
            "isin_growth",
            "isin_div",
            "scheme_name",
            "nav",
            "nav_date",
        ],
        dtype=str,
        engine="python",
    )

    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["nav_date"] = pd.to_datetime(df["nav_date"], errors="coerce")
    df = df.dropna(subset=["nav", "nav_date", "scheme_name"]).copy()

    # ✅ KEY FIX: scheme_code = ISIN
    df["scheme_code"] = df["isin_growth"].fillna("").astype(str).str.strip()
    fallback = df["isin_div"].fillna("").astype(str).str.strip()
    df.loc[df["scheme_code"] == "", "scheme_code"] = fallback
    df = df[df["scheme_code"] != ""].copy()

    df = df.rename(columns={"nav_date": "date"})
    df = df[["scheme_code", "scheme_name", "nav", "date"]].copy()

    # cache as csv.gz
    _write_cache(df)
    return df


def get_latest_nav(force: bool = False) -> pd.DataFrame:
    df = download_amfi_nav(force=force)

    df = df.dropna(subset=["scheme_code", "nav", "date"]).copy()
    df = df.sort_values("date").groupby("scheme_code", as_index=False).tail(1)

    df["scheme_code"] = df["scheme_code"].astype(str).str.strip()
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df[["scheme_code", "scheme_name", "nav", "date"]].reset_index(drop=True)
