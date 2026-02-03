# src/mf/portfolio/nav_loader.py
import os
import io
import pandas as pd

DATA_CACHE_DIR = "data_cache"


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).replace("\ufeff", "").strip().lower().replace(" ", "_")
        for c in df.columns
    ]
    return df


def load_nav_csv(uploaded=None) -> pd.DataFrame:
    """
    Loads NAV csv and returns standardized columns:
      scheme_code, date, nav

    Accepts many common column variants:
      scheme_code: scheme_code, schemecode, code, schemeid, isin
      date: date, nav_date, navdate, datetime
      nav: nav, nav_value, navvalue, net_asset_value, nav_rs, value
    """
    # If not uploaded, try default path
    if uploaded is None:
        default_path = os.path.join(DATA_CACHE_DIR, "mf_nav.csv")
        if os.path.exists(default_path):
            uploaded = default_path
        else:
            raise FileNotFoundError(
                "NAV file not found. Add data_cache/mf_nav.csv or upload NAV CSV in UI."
            )

    # Read file-like or path
    if isinstance(uploaded, str):
        df = pd.read_csv(uploaded, sep=None, engine="python")
    elif hasattr(uploaded, "read"):
        raw = uploaded.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(raw), sep=None, engine="python")
    else:
        df = pd.read_csv(uploaded, sep=None, engine="python")

    df = _norm_cols(df)

    rename = {
        # scheme_code aliases
        "schemecode": "scheme_code",
        "schemeid": "scheme_code",
        "scheme_id": "scheme_code",
        "code": "scheme_code",
        "isin": "scheme_code",

        # date aliases
        "nav_date": "date",
        "navdate": "date",
        "datetime": "date",

        # nav aliases
        "navvalue": "nav",
        "nav_value": "nav",
        "net_asset_value": "nav",
        "nav_rs": "nav",
        "value": "nav",
    }

    df = df.rename(columns=rename)

    required = {"scheme_code", "date", "nav"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"NAV data missing columns: {missing}. Found: {list(df.columns)}. "
            f"Expected at least: scheme_code, date, nav"
        )

    df["scheme_code"] = df["scheme_code"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")

    df = df.dropna(subset=["scheme_code", "date", "nav"]).reset_index(drop=True)

    return df
