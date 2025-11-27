#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
import argparse, pandas as pd, os

def make_label(df, horizon_days, threshold):
    df = df.sort_values(["ticker","date"]).copy()
    # forward return over H days: (close_{t+H} / close_t) - 1
    df[f"ret_fwd_{horizon_days}d"] = (
        df.groupby("ticker")["close"].shift(-horizon_days) / df["close"] - 1.0
    )
    # label: 1 if forward return > threshold, else 0
    df[f"y_h{horizon_days}"] = (df[f"ret_fwd_{horizon_days}d"] > threshold).astype("Int64")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/technical_only.csv")
    ap.add_argument("--horizon_days", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["date"])
    out_df = make_label(df, args.horizon_days, args.threshold)

    # drop last H rows per ticker where label is NA (no future data)
    label_col = f"y_h{args.horizon_days}"
    out_df = (out_df
              .sort_values(["ticker","date"])
              .groupby("ticker", group_keys=False)
              .apply(lambda g: g.iloc[:-args.horizon_days] if len(g) > args.horizon_days else g.dropna(subset=[label_col]))
             )
    out_df = out_df.reset_index(drop=True)

    # pick output path
    if args.out is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.out = f"data/processed/{base}_h{args.horizon_days}.csv"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved {args.out} with label column '{label_col}'")

if __name__ == "__main__":
    main()

