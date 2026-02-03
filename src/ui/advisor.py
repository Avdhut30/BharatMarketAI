# src/ui/advisor.py

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange


REPORTS_DIR = "reports"
DATA_DIR = "data_cache"
MODEL_PATH = os.path.join(REPORTS_DIR, "lgbm_classifier.pkl")
NEWS_DAILY_PATH = os.path.join(DATA_DIR, "news_daily.csv")
SELECTED_CONFIG_PATH = os.path.join(REPORTS_DIR, "selected_config.json")

NUMERIC_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Cache config (helps on Streamlit Cloud where Yahoo blocks often)
CACHE_TTL_HOURS = 6  # use cached price data for 6 hours
YF_MAX_RETRIES = 3


# ----------------------------
# Helpers
# ----------------------------
def _ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def read_best_threshold(default=0.52):
    """
    Reads the threshold from selected_config.json if it exists.
    Otherwise returns the provided default.
    """
    if not os.path.exists(SELECTED_CONFIG_PATH):
        return float(default)
    try:
        with open(SELECTED_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return float(cfg.get("threshold", default))
    except Exception:
        return float(default)


def _flatten_yf_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    yfinance may return MultiIndex columns (Price, Ticker).
    Convert to simple columns: Open, High, Low, Close, Volume.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        else:
            df = df.xs(df.columns.get_level_values(-1)[0], axis=1, level=-1)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in NUMERIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}. Found: {list(df.columns)}")

    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df["Volume"] = df["Volume"].fillna(0)
    return df


def _cache_path(symbol: str, years: int) -> str:
    safe = symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
    return os.path.join(DATA_DIR, f"yf_{safe}_{years}y.parquet")


def _is_cache_fresh(path: str, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    if not os.path.exists(path):
        return False
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return (datetime.now() - mtime) <= timedelta(hours=ttl_hours)
    except Exception:
        return False


def _load_cache(path: str) -> pd.DataFrame | None:
    try:
        if os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        return None
    return None


def _save_cache(df: pd.DataFrame, path: str):
    try:
        df.to_parquet(path)
    except Exception:
        # fallback to csv if parquet fails
        try:
            df.to_csv(path.replace(".parquet", ".csv"))
        except Exception:
            pass


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same technical features as training pipeline (no leakage).
    """
    df = df.copy()

    df["ret_1d"] = df["Close"].pct_change(1)
    df["ret_5d"] = df["Close"].pct_change(5)

    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    df["sma_20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()
    df["sma_200"] = SMAIndicator(df["Close"], window=200).sma_indicator()

    df["close_over_sma20"] = df["Close"] / df["sma_20"] - 1.0
    df["close_over_sma50"] = df["Close"] / df["sma_50"] - 1.0
    df["trend_200"] = (df["Close"] > df["sma_200"]).astype(int)

    df["rsi_14"] = RSIIndicator(df["Close"], window=14).rsi()

    atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
    df["atr_14"] = atr.average_true_range()
    df["atr_pct"] = df["atr_14"] / df["Close"]

    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["vol_z_20"] = (df["Volume"] - vol_mean) / vol_std

    return df


def load_news_daily() -> pd.DataFrame | None:
    """
    Loads daily news features if file exists.
    Returns dataframe indexed by Date.
    """
    if not os.path.exists(NEWS_DAILY_PATH):
        return None

    try:
        nd = pd.read_csv(NEWS_DAILY_PATH)
        if "Date" not in nd.columns:
            return None
        nd["Date"] = pd.to_datetime(nd["Date"], errors="coerce")
        nd = nd.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        nd = nd.select_dtypes(include=[np.number]).copy()
        return nd
    except Exception:
        return None


def merge_news_features(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily news features by Date index. Missing -> fill 0.
    """
    nd = load_news_daily()
    if nd is None or nd.empty:
        return feat

    out = feat.join(nd, how="left")
    for c in nd.columns:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    return out


def load_model_bundle():
    _ensure_dirs()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model bundle not found: {MODEL_PATH}. Run: python -m src.models.train"
        )

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        model = bundle.get("model") or bundle.get("clf") or bundle.get("lgbm") or bundle.get("estimator")
        feature_cols = bundle.get("feature_cols") or bundle.get("features")
        if model is None:
            raise ValueError("Model bundle is a dict but model object not found.")
        return model, feature_cols, bundle

    return bundle, None, {"model": bundle}


def download_history(symbol: str, years: int = 3) -> pd.DataFrame:
    """
    Download enough history to compute SMA200 etc.
    Adds: disk cache + retries for Streamlit Cloud / Yahoo blocking.
    """
    _ensure_dirs()

    cache_file = _cache_path(symbol, years)

    # 1) Use fresh cache immediately
    if _is_cache_fresh(cache_file):
        cached = _load_cache(cache_file)
        if cached is not None and not cached.empty:
            cached.index = pd.to_datetime(cached.index, errors="coerce")
            cached = cached.dropna(axis=0, how="any")
            cached.index.name = "Date"
            cached = _clean_ohlcv(cached)
            return cached

    last_err = None

    # 2) Retry yfinance a few times
    for attempt in range(1, YF_MAX_RETRIES + 1):
        try:
            df = yf.download(
                symbol,
                period=f"{years}y",
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
            )

            if df is None or df.empty:
                raise ValueError("Empty response from Yahoo/yfinance")

            df = _flatten_yf_columns(df, symbol)

            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                date_col = "Date" if "Date" in df.columns else df.columns[0]
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col]).set_index(date_col)

            df.index.name = "Date"
            df = _clean_ohlcv(df)

            # save cache
            _save_cache(df, cache_file)

            return df

        except Exception as e:
            last_err = e
            # exponential backoff: 1s, 2s, 4s
            time.sleep(2 ** (attempt - 1))

    # 3) If yfinance failed, try using stale cache (better than nothing)
    cached = _load_cache(cache_file)
    if cached is not None and not cached.empty:
        try:
            cached.index = pd.to_datetime(cached.index, errors="coerce")
            cached = cached.dropna(axis=0, how="any")
            cached.index.name = "Date"
            cached = _clean_ohlcv(cached)
            return cached
        except Exception:
            pass

    raise ValueError(
        f"Failed to fetch data for {symbol} from Yahoo/yfinance after {YF_MAX_RETRIES} attempts. "
        f"Reason: {repr(last_err)}"
    )


# ----------------------------
# Main Scoring
# ----------------------------
def score_single_symbol(
    symbol: str,
    buy_threshold: float = 0.52,
    sell_threshold: float = 0.45,
    require_trend200_for_buy: bool = True,
):
    """
    Returns:
      feat (full features dataframe),
      latest_row (series),
      p_up (float),
      decision (BUY/HOLD/SELL)
    """
    model, feature_cols, _bundle = load_model_bundle()

    df = download_history(symbol, years=3)

    feat = make_features(df).dropna().copy()
    feat = merge_news_features(feat)

    if feat.empty:
        raise ValueError("Not enough history after indicators (need ~200+ rows).")

    latest = feat.iloc[-1:].copy()

    if feature_cols is None:
        exclude = set(["Open", "High", "Low", "Close", "Volume"])
        feature_cols = [
            c for c in latest.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(latest[c])
        ]

    for c in feature_cols:
        if c not in latest.columns:
            latest[c] = 0.0

    X = latest[feature_cols].copy()

    if hasattr(model, "predict_proba"):
        p_up = float(model.predict_proba(X)[0, 1])
    else:
        raw = float(model.predict(X)[0])
        p_up = float(1.0 / (1.0 + np.exp(-raw)))

    trend_200 = int(latest["trend_200"].iloc[0]) if "trend_200" in latest.columns else 1

    if p_up >= buy_threshold and (trend_200 == 1 or not require_trend200_for_buy):
        decision = "BUY"
    elif p_up <= sell_threshold:
        decision = "SELL"
    else:
        decision = "HOLD"

    return feat, latest.iloc[0], p_up, decision
