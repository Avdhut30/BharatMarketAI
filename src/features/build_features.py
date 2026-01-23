# src/features/build_features.py

import os
import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange

from src.config import NIFTY_50, HORIZON_DAYS, DATA_DIR
from src.data.market import load_universe, fetch_ohlcv

# Optional: market index proxy for regime features
MARKET_PROXY = "^NSEI"  # NIFTY 50 index on Yahoo


def _log_ret(s: pd.Series) -> pd.Series:
    return np.log(s).diff()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Technical features (NO future leakage).
    Input df: index=Date, cols Open High Low Close Volume
    """
    df = df.copy()
    c = df["Close"]

    # returns (log)
    df["logret_1d"] = _log_ret(c)
    df["logret_5d"] = np.log(c).diff(5)
    df["logret_10d"] = np.log(c).diff(10)
    df["logret_20d"] = np.log(c).diff(20)

    # simple returns too (some models like both)
    df["ret_1d"] = c.pct_change(1)
    df["ret_5d"] = c.pct_change(5)
    df["ret_10d"] = c.pct_change(10)

    # volatility
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["vol_60d"] = df["ret_1d"].rolling(60).std()

    # moving averages + regime
    df["sma_20"] = SMAIndicator(c, window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(c, window=50).sma_indicator()
    df["sma_200"] = SMAIndicator(c, window=200).sma_indicator()

    df["close_over_sma20"] = c / df["sma_20"] - 1.0
    df["close_over_sma50"] = c / df["sma_50"] - 1.0
    df["close_over_sma200"] = c / df["sma_200"] - 1.0

    df["trend_20_50"] = (df["sma_20"] > df["sma_50"]).astype(int)
    df["trend_50_200"] = (df["sma_50"] > df["sma_200"]).astype(int)
    df["trend_200"] = (c > df["sma_200"]).astype(int)

    # RSI
    df["rsi_14"] = RSIIndicator(c, window=14).rsi()

    # ATR + range
    atr = AverageTrueRange(df["High"], df["Low"], c, window=14)
    df["atr_14"] = atr.average_true_range()
    df["atr_pct"] = df["atr_14"] / c

    df["range_pct"] = (df["High"] - df["Low"]) / c
    df["close_to_high"] = (df["High"] - c) / c
    df["close_to_low"] = (c - df["Low"]) / c

    # volume features
    v = df["Volume"].replace(0, np.nan)
    df["vol_z_20"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    df["vol_chg_5d"] = v.pct_change(5)
    df["vol_chg_20d"] = v.pct_change(20)

    # cleanup
    return df


def add_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df[f"target_ret_{horizon}d"] = df["Close"].shift(-horizon) / df["Close"] - 1.0
    df[f"target_up_{horizon}d"] = (df[f"target_ret_{horizon}d"] > 0).astype(int)
    return df


def build_market_regime_features() -> pd.DataFrame:
    """
    Market regime using NIFTY index proxy. Joins later by Date for all symbols.
    """
    try:
        mkt = fetch_ohlcv(MARKET_PROXY)
        mkt = mkt.copy()
        mkt["mkt_ret_1d"] = mkt["Close"].pct_change(1)
        mkt["mkt_ret_5d"] = mkt["Close"].pct_change(5)
        mkt["mkt_vol_20d"] = mkt["mkt_ret_1d"].rolling(20).std()
        mkt["mkt_sma_50"] = SMAIndicator(mkt["Close"], window=50).sma_indicator()
        mkt["mkt_trend_50"] = (mkt["Close"] > mkt["mkt_sma_50"]).astype(int)
        mkt = mkt[["mkt_ret_1d", "mkt_ret_5d", "mkt_vol_20d", "mkt_trend_50"]].dropna()
        mkt.index.name = "Date"
        return mkt
    except Exception:
        # If Yahoo blocks index or fails, return empty (still works without it)
        return pd.DataFrame()


def build_dataset(symbols: list) -> pd.DataFrame:
    universe = load_universe(symbols)
    frames = []

    mkt = build_market_regime_features()

    for sym, df in universe.items():
        if df is None or df.empty:
            continue

        feat = make_features(df)
        feat = add_targets(feat, HORIZON_DAYS)

        # join market regime by date (optional)
        if not mkt.empty:
            feat = feat.join(mkt, how="left")

        feat["Symbol"] = sym
        feat = feat.reset_index()  # Date column

        frames.append(feat)

    full = pd.concat(frames, ignore_index=True)

    # drop rows with NaNs (rolling windows + label shift)
    full = full.dropna().reset_index(drop=True)

    full["Date"] = pd.to_datetime(full["Date"], errors="coerce")
    full = full.dropna(subset=["Date"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)

    return full


def save_dataset(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        print(f"✅ Saved dataset: {path}")
    except Exception:
        csv_path = path.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved dataset: {csv_path} (parquet unavailable)")


if __name__ == "__main__":
    out_path = os.path.join(DATA_DIR, "features.parquet")
    ds = build_dataset(NIFTY_50)
    print("Dataset shape:", ds.shape)
    print(ds.head(3))
    save_dataset(ds, out_path)
