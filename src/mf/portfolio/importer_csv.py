# src/mf/portfolio/importer_csv.py
import os
import io
import pandas as pd

DATA_CACHE_DIR = "data_cache"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).replace("\ufeff", "").strip().lower().replace(" ", "_")
        for c in df.columns
    ]
    return df


def _read_csv_any(uploaded_or_path):
    """
    Reads CSV from:
    - Streamlit UploadedFile (has .read())
    - file path string
    - file-like
    """
    if uploaded_or_path is None:
        return None

    # Path
    if isinstance(uploaded_or_path, str):
        if not os.path.exists(uploaded_or_path):
            raise FileNotFoundError(f"File not found: {uploaded_or_path}")
        return pd.read_csv(uploaded_or_path, sep=None, engine="python")

    # Streamlit UploadedFile / file-like
    if hasattr(uploaded_or_path, "read"):
        data = uploaded_or_path.read()
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        else:
            text = str(data)
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")

    return pd.read_csv(uploaded_or_path, sep=None, engine="python")


def load_mf_csv(uploaded=None) -> pd.DataFrame:
    """
    Auto-detect & load MF CSV.

    Supports:
    A) Holdings CSV:
       scheme_code, scheme_name, units

    B) Transactions CSV:
       scheme_code, scheme_name, date, amount (+ optional type)

    Returns df + df.attrs["mf_kind"] = "holdings" | "transactions"
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # If user didn't upload, try defaults
    if uploaded is None:
        p_txn = os.path.join(DATA_CACHE_DIR, "mf_transactions.csv")
        p_hold = os.path.join(DATA_CACHE_DIR, "mf_portfolio.csv")
        if os.path.exists(p_txn):
            uploaded = p_txn
        elif os.path.exists(p_hold):
            uploaded = p_hold

    df = _read_csv_any(uploaded)
    if df is None:
        raise ValueError(
            "No MF CSV provided and no default file found. "
            "Place `data_cache/mf_transactions.csv` or `data_cache/mf_portfolio.csv`."
        )

    df = _normalize_columns(df)

    # Aliases -> canonical
    rename_map = {
        # scheme code
        "schemecode": "scheme_code",
        "code": "scheme_code",
        "fund_code": "scheme_code",
        "isin": "scheme_code",

        # scheme name
        "schemename": "scheme_name",
        "fundname": "scheme_name",
        "fund_name": "scheme_name",
        "name": "scheme_name",

        # units (holdings)
        "unit": "units",
        "units_held": "units",
        "qty": "units",
        "quantity": "units",

        # transactions
        "txn_date": "date",
        "transaction_date": "date",
        "datetime": "date",

        "amt": "amount",
        "cashflow": "amount",
        "transaction_amount": "amount",
        "value": "amount",
    }
    df = df.rename(columns=rename_map)

    has_holdings = {"scheme_code", "scheme_name", "units"}.issubset(df.columns)
    has_txn = {"scheme_code", "scheme_name", "date", "amount"}.issubset(df.columns)

    # Transactions
    if has_txn and not has_holdings:
        df["scheme_code"] = df["scheme_code"].astype(str).str.strip()
        df["scheme_name"] = df["scheme_name"].astype(str).str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        df = df.dropna(subset=["scheme_code", "scheme_name", "date", "amount"]).reset_index(drop=True)
        df = df.sort_values("date")
        df.attrs["mf_kind"] = "transactions"
        return df

    # Holdings
    if has_holdings and not has_txn:
        df["scheme_code"] = df["scheme_code"].astype(str).str.strip()
        df["scheme_name"] = df["scheme_name"].astype(str).str.strip()
        df["units"] = pd.to_numeric(df["units"], errors="coerce")

        df = df.dropna(subset=["scheme_code", "scheme_name", "units"])
        df = df[df["units"] > 0].reset_index(drop=True)
        df.attrs["mf_kind"] = "holdings"
        return df

    # If both present, treat as transactions (richer)
    if has_txn and has_holdings:
        df["scheme_code"] = df["scheme_code"].astype(str).str.strip()
        df["scheme_name"] = df["scheme_name"].astype(str).str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df.attrs["mf_kind"] = "transactions"
        return df

    raise ValueError(
        f"CSV format not recognized. Found columns: {list(df.columns)}. "
        f"Need either holdings: scheme_code, scheme_name, units "
        f"OR transactions: scheme_code, scheme_name, date, amount."
    )


# Backward compatibility
def load_portfolio_csv(uploaded=None) -> pd.DataFrame:
    return load_mf_csv(uploaded)
