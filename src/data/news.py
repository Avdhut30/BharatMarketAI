# src/data/news.py

import os
import time
import hashlib
import feedparser
import pandas as pd


RSS_FEEDS = [
    # Business / Markets
    "https://www.reuters.com/rssFeed/businessNews",
    "https://www.reuters.com/rssFeed/marketsNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.livemint.com/rss/markets",
    "https://www.moneycontrol.com/rss/business.xml",

    # Geopolitics / World
    "https://www.reuters.com/rssFeed/worldNews",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
]


def _safe_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def fetch_rss(feeds=None, max_items_per_feed=200, sleep_s=0.3) -> pd.DataFrame:
    """
    Fetch RSS feeds and return a dataframe:
    Date, title, source, link, id
    """
    feeds = feeds or RSS_FEEDS
    rows = []

    for url in feeds:
        try:
            d = feedparser.parse(url)
            source = d.feed.get("title", url)

            items = d.entries[:max_items_per_feed]
            for it in items:
                title = (it.get("title") or "").strip()
                link = (it.get("link") or "").strip()

                # Prefer published, fallback updated
                dt = it.get("published") or it.get("updated") or None
                if dt is None:
                    continue

                # feedparser gives structured time sometimes, but string is ok -> parse later
                uid = _safe_id(f"{title}|{link}|{dt}")

                rows.append({
                    "date_raw": dt,
                    "title": title,
                    "source": source,
                    "link": link,
                    "id": uid,
                })

            time.sleep(sleep_s)
        except Exception as e:
            print(f"❌ RSS failed {url}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Parse date
    df["DateTime"] = pd.to_datetime(df["date_raw"], errors="coerce", utc=True)
    df = df.dropna(subset=["DateTime"])
    # Convert to date (UTC date). For India you can later shift timezone if you want.
    df["Date"] = df["DateTime"].dt.date
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    df = df[["Date", "DateTime", "title", "source", "link", "id"]].sort_values("DateTime").reset_index(drop=True)
    return df


def save_news(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved news: {path}")


if __name__ == "__main__":
    out = "data_cache/news_raw.csv"
    df = fetch_rss()
    print("Rows:", len(df))
    print(df.head(5))
    if len(df):
        save_news(df, out)
