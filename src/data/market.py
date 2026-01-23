# src/data/market.py

import os
import pandas as pd
import yfinance as yf

from src.config import START_DATE, END_DATE, DATA_DIR

NUMERIC_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _debug_where_am_i():
    # Helps confirm you're running the file you edited
    print(f"🧭 market.py path: {os.path.abspath(__file__)}")


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns MultiIndex columns:
      (Price, Ticker) e.g. ('Close','RELIANCE.NS')
    We want single level: Open/High/Low/Close/Volume.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # take level 0 => ['Close','High','Low','Open','Volume']
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure dataframe index is a DatetimeIndex.
    Never rely on a 'Date' column existing.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # If not, try common column names
    for c in ["Date", "Datetime", "index"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.dropna(subset=[c]).set_index(c)
            return df

    # Fallback: try first column
    if len(df.columns) > 0:
        c = df.columns[0]
        df[c] = pd.to_datetime(df[c], errors="coerce")
        df = df.dropna(subset=[c]).set_index(c)
        return df

    raise ValueError("Could not create a DatetimeIndex.")


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _flatten_yf_columns(df)
    df = _ensure_datetime_index(df)

    missing = [c for c in NUMERIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}. Found: {list(df.columns)}")

    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df["Volume"] = df["Volume"].fillna(0)
    df = df.sort_index()
    df.index.name = "Date"  # normalize index name
    return df


def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    # Load cache
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)

        # Set index from Date if present, else from first column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")
        else:
            df = _ensure_datetime_index(df)

        df = _clean_ohlcv(df)
        return df

    # Download fresh
    print(f"📥 Downloading {symbol}...")
    df = yf.download(
        tickers=symbol,
        start=START_DATE,
        end=END_DATE,
        period=None,
        progress=False,
        auto_adjust=True,
        group_by="column",
        actions=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data found for {symbol}")

    # IMPORTANT: clean using index directly (no reset_index needed)
    df = _clean_ohlcv(df)

    # Save cache with an explicit Date column
    out = df.reset_index()  # now it will be named 'Date' because df.index.name="Date"
    out.to_csv(cache_path, index=False)

    return df


def load_universe(symbols: list) -> dict:
    data = {}
    for sym in symbols:
        try:
            data[sym] = fetch_ohlcv(sym)
        except Exception as e:
            print(f"❌ Failed {sym}: {e}")
    return data


if __name__ == "__main__":
    _debug_where_am_i()
    from src.config import NIFTY_50

    market_data = load_universe(NIFTY_50)
    print(f"✅ Loaded {len(market_data)} stocks")
