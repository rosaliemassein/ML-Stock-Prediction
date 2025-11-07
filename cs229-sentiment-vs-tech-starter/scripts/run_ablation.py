#!/usr/bin/env python
import argparse, pandas as pd, numpy as np, json
from src.utils.io import load_config, ensure_dir
from src.utils.cv import time_series_splits
from src.models.baselines import train_eval_baselines
from src.models.ensembles import train_eval_ensembles

def main(cfg_path):
    cfg = load_config(cfg_path)
    df = pd.read_csv('data/processed/with_sentiment.csv', parse_dates=['date']).sort_values('date')
    tech_cols = [c for c in df.columns if c.startswith(('sma_','mom_','vol_','rsi')) or c=='ret_1d']
    sent_cols = ['sentiment']
    label = 'y'
    out = dict(tech_only=[], sent_only=[], combined_baselines=[], combined_ensembles=[])
    for tr_idx, te_idx in time_series_splits(df['date'], **cfg['cv']):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        # Tech only
        res_t = train_eval_baselines(tr[tech_cols], tr[label], te[tech_cols], te[label])
        out['tech_only'].append(res_t)
        # Sent only
        res_s = train_eval_baselines(tr[sent_cols], tr[label], te[sent_cols], te[label])
        out['sent_only'].append(res_s)
        # Combined
        res_c_b = train_eval_baselines(tr[tech_cols+sent_cols], tr[label], te[tech_cols+sent_cols], te[label])
        out['combined_baselines'].append(res_c_b)
        res_c_e = train_eval_ensembles(tr[tech_cols+sent_cols], tr[label], te[tech_cols+sent_cols], te[label])
        out['combined_ensembles'].append(res_c_e)
    ensure_dir('reports')
    with open('reports/ablation.json','w') as f:
        json.dump(out, f, indent=2)
    print("Wrote reports/ablation.json")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
