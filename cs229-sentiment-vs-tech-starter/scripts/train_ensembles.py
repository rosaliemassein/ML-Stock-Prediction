#!/usr/bin/env python
import argparse, pandas as pd, numpy as np, json
from src.utils.io import load_config, ensure_dir
from src.utils.cv import time_series_splits
from src.models.ensembles import train_eval_ensembles

def main(cfg_path):
    cfg = load_config(cfg_path)
    df = pd.read_csv('data/processed/with_sentiment.csv', parse_dates=['date']).sort_values('date')
    tech_cols = [c for c in df.columns if c.startswith(('sma_','mom_','vol_','rsi')) or c=='ret_1d']
    sent_cols = ['sentiment']
    all_cols = sorted(set(tech_cols) | set(sent_cols))
    label = 'y'
    results = []
    for tr_idx, te_idx in time_series_splits(df['date'], **cfg['cv']):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        Xtr, Xte = tr[all_cols], te[all_cols]
        ytr, yte = tr[label], te[label]
        res = train_eval_ensembles(Xtr, ytr, Xte, yte)
        results.append(res)
    ensure_dir('reports')
    with open('reports/ensembles.json','w') as f:
        json.dump(results, f, indent=2)
    print("Wrote reports/ensembles.json")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
