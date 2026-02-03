# src/mf/data/nav_history.py
import os
import io
import json
import time
import requests
import pandas as pd

AMFI_URL = "https://portal.amfiindia.com/spages/NAVOpen.txt"
MFAPI_URL = "https://api.mfapi.in/mf/{amfi_code}"

CACHE_DIR = "data_cache"
AMFI_MASTER_CACHE = os.path.join(CACHE_DIR, "amfi_master.parquet")
MFAPI_NAV_DIR = os.path.join(CACHE_DIR, "mfapi_nav")  # per-scheme history cache


def _ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MFAPI_NAV_DIR, exist_ok=True)


def _read_amfi_master(force: bool = False) -> pd.DataFrame:
    """
    Returns AMFI master table with:
      amfi_scheme_code, isin_growth, isin_div, scheme_name, nav, date
    """
    _ensure_dirs()

    if os.path.exists(AMFI_MASTER_CACHE) and not force:
        try:
            return pd.read_parquet(AMFI_MASTER_CACHE)
        except Exception:
            # corrupted cache
            try:
                os.remove(AMFI_MASTER_CACHE)
            except Exception:
                pass

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
            "date",
        ],
        dtype=str,
        engine="python",
    )

    df["amfi_scheme_code"] = df["amfi_scheme_code"].astype(str).str.strip()
    df["isin_growth"] = df["isin_growth"].fillna("").astype(str).str.strip()
    df["isin_div"] = df["isin_div"].fillna("").astype(str).str.strip()
    df["scheme_name"] = df["scheme_name"].fillna("").astype(str).str.strip()
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["amfi_scheme_code", "scheme_name", "nav", "date"]).copy()
    df = df[df["scheme_name"] != ""].copy()

    df.to_parquet(AMFI_MASTER_CACHE, index=False)
    return df


def build_isin_to_amfi_map(force: bool = False) -> dict:
    """
    Returns dict: ISIN (INFxxxx) -> AMFI numeric scheme code
    """
    master = _read_amfi_master(force=force)
    mapping = {}

    # growth isin
    g = master[master["isin_growth"] != ""][["isin_growth", "amfi_scheme_code"]].drop_duplicates()
    for _, row in g.iterrows():
        mapping[row["isin_growth"]] = row["amfi_scheme_code"]

    # dividend isin fallback
    d = master[master["isin_div"] != ""][["isin_div", "amfi_scheme_code"]].drop_duplicates()
    for _, row in d.iterrows():
        mapping.setdefault(row["isin_div"], row["amfi_scheme_code"])

    return mapping


def _fetch_mfapi_nav_history(amfi_code: str, sleep_sec: float = 0.10) -> pd.DataFrame:
    """
    Fetch NAV history from mfapi.in for AMFI numeric scheme code.

    Output columns: date, nav
    """
    url = MFAPI_URL.format(amfi_code=str(amfi_code).strip())
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()

    data = js.get("data", []) or []
    if not data:
        return pd.DataFrame(columns=["date", "nav"])

    # mfapi dates are usually "dd-mm-yyyy"
    df = pd.DataFrame(data)
    if "date" not in df.columns or "nav" not in df.columns:
        return pd.DataFrame(columns=["date", "nav"])

    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    # gentle rate-limit
    if sleep_sec:
        time.sleep(sleep_sec)

    return df[["date", "nav"]]


def _mfapi_cache_path(amfi_code: str) -> str:
    return os.path.join(MFAPI_NAV_DIR, f"{amfi_code}.parquet")


def get_nav_history_for_isins(
    isins: list[str],
    force_refresh: bool = False,
    max_years: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """
    Given a list of ISINs (INFxxxx), returns:
      nav_df: scheme_code(ISIN), date, nav, scheme_name
      meta: dict with missing_isins / missing_amfi / fetched_count

    This returns NAV HISTORY (many dates) so your invested/current/gain becomes real.
    """
    _ensure_dirs()

    isins = [str(x).strip() for x in (isins or []) if str(x).strip()]
    isins = sorted(list(dict.fromkeys(isins)))

    if not isins:
        return pd.DataFrame(columns=["scheme_code", "scheme_name", "date", "nav"]), {
            "missing_isins": [],
            "missing_amfi": [],
            "fetched_count": 0,
        }

    master = _read_amfi_master(force=False)
    isin_to_amfi = build_isin_to_amfi_map(force=False)

    # map ISIN -> AMFI code
    missing_amfi = [i for i in isins if i not in isin_to_amfi]
    usable = [i for i in isins if i in isin_to_amfi]

    # also get scheme_name for each ISIN from master
    name_map = {}
    m2 = master.copy()
    # choose scheme_name from growth match first, else div
    for isin in usable:
        row = m2[m2["isin_growth"] == isin]
        if row.empty:
            row = m2[m2["isin_div"] == isin]
        if not row.empty:
            name_map[isin] = row.iloc[0]["scheme_name"]
        else:
            name_map[isin] = isin

    all_rows = []
    fetched = 0

    # filter window
    start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=max_years * 365)

    for isin in usable:
        amfi_code = isin_to_amfi[isin]
        cache_path = _mfapi_cache_path(amfi_code)

        df_hist = None
        if (not force_refresh) and os.path.exists(cache_path):
            try:
                df_hist = pd.read_parquet(cache_path)
            except Exception:
                df_hist = None
                try:
                    os.remove(cache_path)
                except Exception:
                    pass

        if df_hist is None or df_hist.empty:
            try:
                df_hist = _fetch_mfapi_nav_history(amfi_code)
                if not df_hist.empty:
                    df_hist.to_parquet(cache_path, index=False)
                    fetched += 1
            except Exception:
                df_hist = pd.DataFrame(columns=["date", "nav"])

        if df_hist.empty:
            continue

        # trim to max_years for performance
        df_hist = df_hist[df_hist["date"] >= start_date].copy()

        # convert to app schema where scheme_code = ISIN
        df_hist["scheme_code"] = isin
        df_hist["scheme_name"] = name_map.get(isin, isin)
        all_rows.append(df_hist[["scheme_code", "scheme_name", "date", "nav"]])

    nav_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(
        columns=["scheme_code", "scheme_name", "date", "nav"]
    )

    # final cleanup
    if not nav_df.empty:
        nav_df["scheme_code"] = nav_df["scheme_code"].astype(str).str.strip()
        nav_df["scheme_name"] = nav_df["scheme_name"].astype(str).str.strip()
        nav_df["date"] = pd.to_datetime(nav_df["date"], errors="coerce")
        nav_df["nav"] = pd.to_numeric(nav_df["nav"], errors="coerce")
        nav_df = nav_df.dropna(subset=["scheme_code", "date", "nav"]).copy()
        nav_df = nav_df.sort_values(["scheme_code", "date"]).reset_index(drop=True)

    return nav_df, {
        "missing_isins": missing_amfi,   # ISINs not found in AMFI master
        "missing_amfi": missing_amfi,    # same list for clarity
        "fetched_count": fetched,
    }
