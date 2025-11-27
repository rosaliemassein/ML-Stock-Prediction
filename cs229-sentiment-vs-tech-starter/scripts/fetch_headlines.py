#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fetch headlines for a list of tickers using:
  (1) yfinance Ticker.news
  (2) Yahoo Finance RSS per ticker
  (3) Google News RSS with ticker + aliases

Outputs a CSV with columns: date,ticker,text,source,url,published_at

Usage (module):
  python -m scripts.fetch_headlines --config configs/default.yaml \
    --start 2025-08-06 --end 2025-11- \
    --tickers AAPL,TSLA,MSFT,SPY,NVDA,GOOG,AMZN,META,NFLX,AMD \
    --out data/raw/news/headlines.csv
"""

import argparse
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from src.utils.io import load_config, ensure_dir

# --- Configs ---
TZ_EXCHANGE = "US/Eastern"   # local exchange timezone for dating news
REQ_COLS = ["date", "ticker", "text", "source", "url", "published_at"]

# Minimum number of news items we try to get per ticker in [start, end]
MIN_NEWS_PER_TICKER = 20

# Aliases: also search by full names / variants in Google News
ALIASES = {
    # Big tech
    "AAPL": ["Apple", "Apple Inc", "iPhone", "MacBook"],
    "TSLA": ["Tesla", "Tesla Motors", "Elon Musk", "Cybertruck", "Model 3", "Model Y"],
    "MSFT": ["Microsoft", "Microsoft Corporation", "Windows", "Azure"],
    "GOOG": ["Google", "Alphabet", "Alphabet Inc", "YouTube"],
    "GOOGL": ["Google", "Alphabet", "Alphabet Inc", "YouTube"],
    "AMZN": ["Amazon", "Amazon.com", "AWS", "Prime Video"],
    "META": ["Meta", "Meta Platforms", "Facebook", "Instagram", "WhatsApp", "Mark Zuckerberg"],
    "NFLX": ["Netflix", "streaming platform"],
    "NVDA": ["Nvidia", "NVIDIA", "NVIDIA Corp", "GPU"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "INTC": ["Intel", "Intel Corp"],
    "QCOM": ["Qualcomm"],
    "SHOP": ["Shopify"],
    "CRM": ["Salesforce", "Salesforce.com"],
    "ORCL": ["Oracle"],

    # Banks & payments
    "JPM": ["JPMorgan", "JPMorgan Chase"],
    "BAC": ["Bank of America"],
    "GS": ["Goldman Sachs"],
    "MS": ["Morgan Stanley"],
    "MA": ["Mastercard"],
    "V": ["Visa"],

    # Energy
    "XOM": ["Exxon", "Exxon Mobil"],
    "CVX": ["Chevron"],
    "BP": ["BP plc"],

    # Healthcare
    "PFE": ["Pfizer"],
    "MRK": ["Merck"],
    "UNH": ["UnitedHealth", "UnitedHealth Group"],

    # Consumer / industrial names
    "DIS": ["Disney", "Walt Disney"],
    "NKE": ["Nike"],
    "BA": ["Boeing"],
    "CAT": ["Caterpillar"],
    "COST": ["Costco"],
    "WMT": ["Walmart"],
    "TGT": ["Target"],
    "KO": ["Coca-Cola"],
    "PEP": ["Pepsi", "PepsiCo"],
    "IBM": ["IBM"],
    "UPS": ["UPS", "United Parcel Service"],

    # ETFs & indices (broad news)
    "SPY": ["S&P 500", "SPDR S&P 500"],
    "QQQ": ["Nasdaq 100", "Invesco QQQ"],
    "DIA": ["Dow Jones", "Dow Jones Industrial Average"],
}
  

# --- Helpers -----------------------------------------------------------------

def _first_nonempty(d: Dict[str, Any], keys: Iterable[str], default: Optional[str] = None) -> Optional[str]:
    """Return the first present & non-empty value for any of the keys."""
    for k in keys:
        if k in d and d[k] not in (None, ""):
            v = d[k]
            # yfinance sometimes returns nested dicts for provider/publisher
            if isinstance(v, dict):
                # try common fields a nested provider might carry
                for kk in ("displayName", "name", "title", "publisher"):
                    if kk in v and v[kk]:
                        return str(v[kk])
                return str(v)  # fallback: stringified dict
            return str(v)
    return default


def _parse_publish_ts(item: Dict[str, Any]) -> Optional[pd.Timestamp]:
    """
    Handle yfinance variants:
      - 'providerPublishTime' (int seconds since epoch)
      - 'published_at'/'providerPublishDate' (ISO string)
    Always return UTC-aware timestamp.
    """
    if "providerPublishTime" in item and item["providerPublishTime"] not in (None, ""):
        try:
            return pd.to_datetime(int(item["providerPublishTime"]), unit="s", utc=True)
        except Exception:
            pass

    for k in ("published_at", "providerPublishDate"):
        if k in item and item[k]:
            try:
                return pd.to_datetime(item[k], utc=True)
            except Exception:
                continue

    # yfinance newer: sometimes under "content": {"pubDate": "..."}
    if "content" in item and isinstance(item["content"], dict):
        for k in ("pubDate", "publishedAt"):
            if k in item["content"] and item["content"][k]:
                try:
                    return pd.to_datetime(item["content"][k], utc=True)
                except Exception:
                    continue

    return None


def _source_str(item: Dict[str, Any]) -> str:
    # Try common fields to build a compact source string
    s = _first_nonempty(item, ("provider", "publisher", "source", "content", "author"), default="")
    return str(s)


def _title_str(item: Dict[str, Any]) -> str:
    # yfinance: "title" or sometimes "content": {"title": ...}
    t = _first_nonempty(item, ("title",), default=None)
    if t:
        return t
    if "content" in item and isinstance(item["content"], dict):
        if item["content"].get("title"):
            return str(item["content"]["title"])
    return ""


def _url_str(item: Dict[str, Any]) -> str:
    u = _first_nonempty(item, ("link", "url"), default=None)
    if u:
        return u
    if "content" in item and isinstance(item["content"], dict):
        for k in ("canonicalUrl", "url", "shareUrl"):
            if item["content"].get(k):
                return str(item["content"][k])
    return ""


# --- Fetchers ----------------------------------------------------------------

def fetch_yf_news_for_ticker(ticker: str) -> List[Dict[str, Any]]:
    """Use yfinance Ticker.news."""
    import yfinance as yf  # local import to avoid hard dependency if unused
    try:
        news = yf.Ticker(ticker).news or []
        if not isinstance(news, list):
            news = []
    except Exception:
        news = []
    return news


def _ensure_feedparser():
    try:
        import feedparser  # noqa: F401
        return True
    except Exception:
        print("[warn] feedparser not installed -> RSS fallbacks disabled")
        return False


def fetch_yahoo_rss(ticker: str) -> List[Dict[str, Any]]:
    """
    Yahoo Finance RSS per ticker.
    Example URL:
      https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US
    """
    if not _ensure_feedparser():
        return []
    import feedparser

    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    parsed = feedparser.parse(url)

    out = []
    for e in parsed.entries:
        title = getattr(e, "title", "") or ""
        link  = getattr(e, "link", "") or ""
        published = None
        if hasattr(e, "published"):
            try:
                published = pd.to_datetime(e.published, utc=True)
            except Exception:
                published = None
        if published is None and hasattr(e, "updated"):
            try:
                published = pd.to_datetime(e.updated, utc=True)
            except Exception:
                published = None
        out.append({
            "title": title,
            "link": link,
            "published_at": published.isoformat() if isinstance(published, pd.Timestamp) else None,
            "provider": "Yahoo Finance",
        })
    return out


def fetch_google_rss(query: str) -> List[Dict[str, Any]]:
    """
    Google News RSS search.

    NOTE: We removed the 'when:90d' restriction so that the feed can
    return older articles; we then filter by [start, end] using timestamps.
    """
    if not _ensure_feedparser():
        return []
    import feedparser
    from urllib.parse import quote

    q = quote(query, safe='"')
    # NO 'when:90d' here → let RSS return as much as possible, then filter
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    parsed = feedparser.parse(url)

    out = []
    for e in parsed.entries:
        title = getattr(e, "title", "") or ""
        link  = getattr(e, "link", "") or ""
        published = None
        if hasattr(e, "published"):
            try:
                published = pd.to_datetime(e.published, utc=True)
            except Exception:
                published = None
        if published is None and hasattr(e, "updated"):
            try:
                published = pd.to_datetime(e.updated, utc=True)
            except Exception:
                published = None
        out.append({
            "title": title,
            "link": link,
            "published_at": published.isoformat() if isinstance(published, pd.Timestamp) else None,
            "provider": getattr(e, "source", {}).get("title", "") if hasattr(e, "source") else "Google News",
        })
    return out


# --- Normalization ------------------------------------------------------------

def normalize_items(ticker: str,
                    items: List[Dict[str, Any]],
                    start: pd.Timestamp,
                    end: pd.Timestamp) -> List[Dict[str, Any]]:
    """
    Normalize raw items (yf or RSS) into rows with REQ_COLS.
    Filter by [start, end] in UTC, then date in TZ_EXCHANGE.
    """
    rows = []
    for it in items:
        ts = _parse_publish_ts(it)
        if ts is None:
            # try RSS 'published_at' if we set it explicitly
            if "published_at" in it and it["published_at"]:
                try:
                    ts = pd.to_datetime(it["published_at"], utc=True)
                except Exception:
                    ts = None
        if ts is None:
            continue

        if not (start <= ts <= end):
            continue

        # Build fields
        title  = _title_str(it).strip()
        if not title:
            continue

        url    = _url_str(it)
        source = _source_str(it)

        # align 'date' to exchange timezone
        dt_local = ts.tz_convert(TZ_EXCHANGE)
        date_str = dt_local.date().isoformat()

        rows.append({
            "date": date_str,
            "ticker": ticker,
            "text": title,
            "source": source if source else "",
            "url": url if url else "",
            "published_at": ts.isoformat()
        })
    return rows


# --- Main --------------------------------------------------------------------

def main(cfg_path: str,
         out_path: str,
         start: Optional[str],
         end: Optional[str],
         tickers_csv: Optional[str]) -> None:
    cfg = load_config(cfg_path)

    # Tickers: CLI override OR YAML
    if tickers_csv:
        tickers = [t.strip() for t in tickers_csv.split(",") if t.strip()]
    else:
        tickers = cfg["universe"]["tickers"]

    # Date window: CLI override OR YAML
    start_ts = pd.Timestamp(start or cfg["universe"]["start"], tz="UTC")
    end_ts   = pd.Timestamp(end   or cfg["universe"]["end"],   tz="UTC")

    ensure_dir("data/raw/news")

    all_rows: List[Dict[str, Any]] = []

    for t in tickers:
        kept_total = 0

        # 1) yfinance news (by ticker)
        yf_items = fetch_yf_news_for_ticker(t)
        rows = normalize_items(t, yf_items, start_ts, end_ts)
        kept_total += len(rows)
        all_rows.extend(rows)

        # 2) Yahoo Finance RSS per ticker
        if kept_total < MIN_NEWS_PER_TICKER:
            rss_yahoo = fetch_yahoo_rss(t)
            rows2 = normalize_items(t, rss_yahoo, start_ts, end_ts)
            kept_total += len(rows2)
            all_rows.extend(rows2)

        # 3) Google News RSS with aliases
        if kept_total < MIN_NEWS_PER_TICKER:
            toks = [t] + ALIASES.get(t, [])
            query = " OR ".join(toks)
            rss_google = fetch_google_rss(query)
            rows3 = normalize_items(t, rss_google, start_ts, end_ts)
            kept_total += len(rows3)
            all_rows.extend(rows3)

        print(f"[info] {t}: kept {kept_total} items in [{start_ts.date()} → {end_ts.date()}]")
        time.sleep(0.2)  # be gentle

    # Build DataFrame with required columns even if no rows
    df = pd.DataFrame(all_rows, columns=REQ_COLS)

    # Dedup (same (ticker,date,text)) + sort
    if len(df):
        df = df.drop_duplicates(subset=["ticker", "date", "text"]).sort_values(["ticker", "date"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path} | rows={len(df)}")


# --- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (default.yaml)")
    ap.add_argument("--out", default="data/raw/news/headlines.csv", help="Output CSV path")
    ap.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None,   help="Override end date (YYYY-MM-DD)")
    ap.add_argument("--tickers", default=None, help="Comma-separated tickers override")
    args = ap.parse_args()
    main(args.config, args.out, args.start, args.end, args.tickers)
