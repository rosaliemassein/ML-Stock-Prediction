#!/usr/bin/env python
import argparse, glob, pandas as pd
from src.features.technical import compute_technical_features
from src.utils.io import load_config, ensure_dir

NUMERIC_COLS = ["open", "high", "low", "close", "volume"]

def coerce_numeric(df):
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main(cfg_path):
    cfg = load_config(cfg_path)
    ensure_dir('data/processed')
    dfs = []
    for path in glob.glob('data/raw/ohlcv_*.csv'):
        df = pd.read_csv(path, parse_dates=['date']).sort_values('date')
        # force numeric types in case CSV had strings
        df = coerce_numeric(df)
        df = compute_technical_features(
            df,
            tuple(cfg['technical']['windows']),
            cfg['technical']['rsi_window'],
            cfg['technical']['vol_window'],
        )
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)

    # label: next-day return > threshold
    out['y'] = (
        out['close'].shift(-cfg['labels']['horizon_days']) / out['close'] - 1.0
        > cfg['labels']['threshold']
    ).astype(int)

    out.to_csv('data/processed/technical_only.csv', index=False)
    print("Saved data/processed/technical_only.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)