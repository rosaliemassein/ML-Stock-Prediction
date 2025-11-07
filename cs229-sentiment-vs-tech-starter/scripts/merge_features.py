#!/usr/bin/env python
import argparse, os
import pandas as pd

def main(tech_path, out_path, sent_path=None, label_col="y_h5"):
    # Load technical features (+ labels)
    tech = pd.read_csv(tech_path, parse_dates=["date"]).sort_values(["ticker","date"])

    # Basic sanity
    if "ticker" not in tech.columns or "date" not in tech.columns:
        raise ValueError(f"{tech_path} must contain 'date' and 'ticker' columns")
    if label_col not in tech.columns:
        raise ValueError(f"Label column '{label_col}' not found in {tech_path}")

    merged = tech.copy()

    # Optionally bring in sentiment features
    if sent_path:
        sent = pd.read_csv(sent_path, parse_dates=["date"]).sort_values(["ticker","date"])
        if "ticker" not in sent.columns or "date" not in sent.columns:
            raise ValueError(f"{sent_path} must contain 'date' and 'ticker' columns")
        # Determine sentiment feature columns (everything except keys)
        sent_feat_cols = [c for c in sent.columns if c not in ("date","ticker")]
        if len(sent_feat_cols) == 0:
            print(f"[warn] No sentiment columns found in {sent_path}; performing a no-op merge.")
        merged = pd.merge(
            merged,
            sent[["date","ticker"] + sent_feat_cols],
            on=["date","ticker"],
            how="left",
            validate="m:1"
        )

    # Write
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path} | rows={len(merged)} | cols={len(merged.columns)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", required=True, help="CSV with technical features and labels (must include date,ticker,label)")
    ap.add_argument("--sent", default=None, help="CSV with sentiment features (must include date,ticker)")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--label_col", default="y_h5", help="Name of label column to keep/check (default: y_h5)")
    args = ap.parse_args()
    main(args.tech, args.out, args.sent, args.label_col)
