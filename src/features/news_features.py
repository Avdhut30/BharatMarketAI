# src/features/news_features.py

import os
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import DATA_DIR

analyzer = SentimentIntensityAnalyzer()

# Keyword buckets (simple but effective baseline)
KEYWORDS = {
    "geo_risk": [
        "war", "conflict", "attack", "missile", "military", "terror",
        "sanction", "embargo", "border", "nuclear", "geopolit"
    ],
    "oil_energy": ["oil", "crude", "opec", "gas", "energy", "brent", "wti"],
    "rates_inflation": ["inflation", "rate", "rates", "fed", "rbi", "yield", "bond", "cpi"],
    "india": ["india", "nifty", "sensex", "rupee", "rbi", "sebi"],
}


def _count_keywords(text: str, words: list) -> int:
    t = text.lower()
    count = 0
    for w in words:
        # substring match is fine for baseline
        if w in t:
            count += 1
    return count


def build_daily_news_features(news_raw_path: str = None) -> pd.DataFrame:
    """
    Input: data_cache/news_raw.csv
    Output: daily aggregated features by Date
    """
    if news_raw_path is None:
        news_raw_path = os.path.join(DATA_DIR, "news_raw.csv")

    if not os.path.exists(news_raw_path):
        raise FileNotFoundError("news_raw.csv not found. Run: python -m src.data.news")

    df = pd.read_csv(news_raw_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # Sentiment score per headline (compound in [-1,1])
    def sent(x):
        x = "" if pd.isna(x) else str(x)
        return analyzer.polarity_scores(x)["compound"]

    df["sent_compound"] = df["title"].apply(sent)

    # Keyword counts
    for k, words in KEYWORDS.items():
        df[k] = df["title"].astype(str).apply(lambda x: _count_keywords(x, words))

    # Aggregate daily
    agg = {
        "sent_compound": ["mean", "min", "max", "std"],
        "title": "count",
    }
    for k in KEYWORDS.keys():
        agg[k] = "sum"

    daily = df.groupby("Date").agg(agg)
    daily.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in daily.columns]
    daily = daily.reset_index()

    # rename a couple
    daily = daily.rename(columns={"title_count": "news_count"})
    daily["sent_compound_std"] = daily["sent_compound_std"].fillna(0.0)

    return daily


def save_daily_features(df: pd.DataFrame, out_path: str = None):
    if out_path is None:
        out_path = os.path.join(DATA_DIR, "news_daily.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved daily news features: {out_path}")


if __name__ == "__main__":
    daily = build_daily_news_features()
    print(daily.head(5))
    save_daily_features(daily)
