# src/mf/nav_amfi.py
import os
import time
import requests
import pandas as pd

AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
DATA_DIR = "data_cache"
NAV_CSV_PATH = os.path.join(DATA_DIR, "amfi_nav.csv")

def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def _parse_amfi_nav_text(text: str) -> pd.DataFrame:
    """
    AMFI NAVAll.txt is semi-colon separated.
    We keep the useful fields:
      - Scheme Code
      - ISIN Div Payout/ISIN Growth (varies)
      - ISIN Div Reinvestment (varies)
      - Scheme Name
      - Net Asset Value
      - Date
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        # data rows are like: code;isin1;isin2;scheme;nav;date
        if ";" not in ln:
            continue
        parts = [p.strip() for p in ln.split(";")]
        if len(parts) < 6:
            continue

        scheme_code = parts[0]
        isin1 = parts[1] if parts[1] else None
        isin2 = parts[2] if parts[2] else None
        scheme_name = parts[3]
        nav = parts[4]
        nav_date = parts[5]

        # Skip header-ish rows
        if not scheme_code.isdigit():
            continue

        try:
            nav_val = float(nav)
        except:
            continue

        rows.append({
            "scheme_code": scheme_code,
            "isin_1": isin1,
            "isin_2": isin2,
            "scheme_name": scheme_name,
            "nav": nav_val,
            "nav_date": nav_date
        })

    df = pd.DataFrame(rows)
    return df

def fetch_and_cache_amfi_nav(force: bool = False, max_age_hours: int = 24) -> pd.DataFrame:
    """
    Downloads AMFI NAV file and caches to CSV.
    If cache is fresh (within max_age_hours), reuse it.
    """
    _ensure_dir()

    if not force and os.path.exists(NAV_CSV_PATH):
        age_sec = time.time() - os.path.getmtime(NAV_CSV_PATH)
        if age_sec < max_age_hours * 3600:
            try:
                return pd.read_csv(NAV_CSV_PATH)
            except:
                pass

    r = requests.get(AMFI_URL, timeout=30)
    r.raise_for_status()

    df = _parse_amfi_nav_text(r.text)
    df.to_csv(NAV_CSV_PATH, index=False)
    return df

def get_nav_for_isin(amfi_df: pd.DataFrame, isin: str) -> float | None:
    """
    Find NAV for a given ISIN in isin_1 or isin_2.
    """
    if amfi_df is None or amfi_df.empty or not isin:
        return None

    isin = str(isin).strip().upper()
    m = amfi_df[(amfi_df["isin_1"].fillna("").str.upper() == isin) |
                (amfi_df["isin_2"].fillna("").str.upper() == isin)]
    if m.empty:
        return None
    # take first match
    return float(m.iloc[0]["nav"])
