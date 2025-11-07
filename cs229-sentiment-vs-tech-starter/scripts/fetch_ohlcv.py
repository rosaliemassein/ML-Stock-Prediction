#!/usr/bin/env python
import argparse
import pandas as pd
import yfinance as yf
from src.utils.io import load_config, ensure_dir

def fetch_ticker_df(ticker, start, end, use_adjusted=False):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} in range {start}..{end}")
    df = df.reset_index().rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low":  "low",
        "Close":"close",
        "Adj Close":"adj_close",
        "Volume":"volume",
    })
    df["ticker"] = ticker
    if use_adjusted and "adj_close" in df:
        df["close"] = df["adj_close"]  # account for splits/dividends
    return df[["date","ticker","open","high","low","close","volume"]]

def main(cfg_path):
    cfg = load_config(cfg_path)
    ensure_dir("data/raw")
    tickers = cfg["universe"]["tickers"]
    start   = cfg["universe"]["start"]
    end     = cfg["universe"]["end"]
    use_adj = cfg.get("universe", {}).get("use_adjusted_close", False)

    for t in tickers:
        try:
            df = fetch_ticker_df(t, start, end, use_adjusted=use_adj)
            out = f"data/raw/ohlcv_{t}.csv"
            df.to_csv(out, index=False)
            print(f"Saved {out} ({len(df)} rows)")
        except Exception as e:
            print(f"[WARN] {t}: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)