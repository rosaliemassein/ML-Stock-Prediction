#!/usr/bin/env python3
# scripts/build_news_simple.py
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

EVENT_PATTERNS = {
    "earnings":  r"\b(earnings|results|eps|quarter|q[1-4]\b|fiscal)\b",
    "rating":    r"\b(upgrade|downgrade|initiates|price target|overweight|underweight|neutral|buy|sell|hold)\b",
    "mna":       r"\b(merger|acquire|acquisition|buyout|takeover|stake|deal)\b",
    "guidance":  r"\b(guidance|outlook|forecast|raises|cuts)\b",
    "layoff":    r"\b(layoff|job cuts|restructure|restructuring|redundanc|workforce)\b",
}
EVENT_WEIGHTS = {"earnings":1.0, "rating":0.7, "mna":1.0, "guidance":0.8, "layoff":0.9}

def safe_read(p):
    return pd.read_csv(p, parse_dates=["date"]) if p else None

def agg_mean(df, cols, prefix):
    g = df.groupby(["date","ticker"], as_index=False)[cols].mean()
    return g.rename(columns={c: f"{prefix}{c}" for c in cols})

def exp_decay_per_ticker(df, col, span=3):
    df = df.sort_values(["ticker","date"]).copy()
    alpha = 2/(span+1.0)
    df[col + f"_decay{span}"] = (
        df.groupby("ticker")[col]
          .transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    )
    return df

def event_weight_from_headlines(headlines_csv):
    h = pd.read_csv(headlines_csv, parse_dates=["date"])
    h["ticker"] = h["ticker"].astype(str).str.upper()
    text = h["text"].fillna("").str.lower()

    hit = {k: text.str.contains(pat, regex=True) for k, pat in EVENT_PATTERNS.items()}
    df = pd.DataFrame({"date": h["date"], "ticker": h["ticker"]})
    for k, m in hit.items():
        df[k] = m.astype(int)
    daily = df.groupby(["date","ticker"], as_index=False)[list(hit.keys())].sum()

    def pick_weight(row):
        best, cnt = None, 0
        for k in EVENT_PATTERNS.keys():
            if row[k] > cnt:
                best, cnt = k, int(row[k])
        return EVENT_WEIGHTS.get(best, 0.0) if cnt > 0 else 0.0

    daily["event_weight"] = daily.apply(pick_weight, axis=1)
    return daily[["date","ticker","event_weight"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headlines", required=True)
    ap.add_argument("--sent_score", default="")
    ap.add_argument("--sent_triplet", default="")
    ap.add_argument("--sent_embed", default="")
    ap.add_argument("--span", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = None

    # Score
    sc = safe_read(args.sent_score)
    if sc is not None and "sent_score" in sc.columns:
        sc["ticker"] = sc["ticker"].astype(str).str.upper()
        s = agg_mean(sc, ["sent_score"], "s_")
        s = exp_decay_per_ticker(s, "s_sent_score", span=args.span)
        out = s if out is None else out.merge(s, on=["date","ticker"], how="outer")

    # Triplet
    tr = safe_read(args.sent_triplet)
    if tr is not None and {"p_pos","p_neu","p_neg"}.issubset(tr.columns):
        tr["ticker"] = tr["ticker"].astype(str).str.upper()
        t = agg_mean(tr, ["p_pos","p_neu","p_neg"], "t_")
        for c in ["t_p_pos","t_p_neu","t_p_neg"]:
            t = exp_decay_per_ticker(t, c, span=args.span)
        out = t if out is None else out.merge(t, on=["date","ticker"], how="outer")

    # Embed PCA16
    em = safe_read(args.sent_embed)
    if em is not None:
        em["ticker"] = em["ticker"].astype(str).str.upper()
        emb_cols = [c for c in em.columns if c.startswith(("emb_", "embed_pca16_"))]
        if emb_cols:
            e = agg_mean(em, emb_cols, "e_")
            out = e if out is None else out.merge(e, on=["date","ticker"], how="outer")

    if out is None:
        raise ValueError("No sentiment input found.")

    # Event weight
    ew = event_weight_from_headlines(args.headlines)
    out = out.merge(ew, on=["date","ticker"], how="left")
    out["event_weight"] = out["event_weight"].fillna(0.0)

    out = out.sort_values(["ticker","date"]).reset_index(drop=True)
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out} | rows={len(out)} | cols={len(out.columns)}")

if __name__ == "__main__":
    main()
