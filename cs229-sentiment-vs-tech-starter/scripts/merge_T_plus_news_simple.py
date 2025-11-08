#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", required=True, help="data/processed/technical_only_h5.csv or merge_T_only_h5.csv")
    ap.add_argument("--news", required=True, help="data/processed/news_features_simple.csv")
    ap.add_argument("--out",  required=True)
    args = ap.parse_args()

    tech = pd.read_csv(args.tech, parse_dates=["date"])
    news = pd.read_csv(args.news, parse_dates=["date"])
    tech["ticker"] = tech["ticker"].astype(str)
    news["ticker"] = news["ticker"].astype(str)

    merged = tech.merge(news, on=["date","ticker"], how="left")
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out} | rows={len(merged)} | cols={len(merged.columns)}")

if __name__ == "__main__":
    main()
