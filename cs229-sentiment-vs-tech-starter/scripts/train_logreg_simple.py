#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

def chrono_split(df, test_ratio=0.3):
    dates = np.sort(df["date"].unique())
    cut = int((1 - test_ratio)*len(dates))
    split_date = dates[cut]
    return df[df["date"] < split_date], df[df["date"] >= split_date], split_date

def eval_lr(df, feats, label_col):
    d = df.dropna(subset=feats + [label_col])
    tr, te, split = chrono_split(d)
    Xtr, Xte = tr[feats].values, te[feats].values
    ytr, yte = tr[label_col].astype(int).values, te[label_col].astype(int).values
    scaler = StandardScaler()
    Xtr, Xte = scaler.fit_transform(Xtr), scaler.transform(Xte)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    y_prob = clf.predict_proba(Xte)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    return accuracy_score(yte, y_pred), roc_auc_score(yte, y_prob), len(tr), len(te), split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="data/processed/merge_T_plus_news_simple_h5.csv")
    ap.add_argument("--label", default="", help="defaults to y_h5 if present else y")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"]).sort_values(["ticker","date"])
    label = args.label or ("y_h5" if "y_h5" in df.columns else "y")
    assert label in df.columns, "Label column not found."

    # technical features (keep it identical to your baseline)
    tech_feats = [c for c in ["ret_1d","sma_5","mom_5","sma_10","mom_10","sma_20","mom_20","vol_20","rsi","rsi_14"] if c in df.columns]

    # sentiment means & decays (as produced in build_news_simple.py)
    score_cols   = [c for c in df.columns if c in ("s_sent_score","s_sent_score_decay3")]
    trip_cols    = [c for c in df.columns if c in ("t_p_pos","t_p_neu","t_p_neg","t_p_pos_decay3","t_p_neu_decay3","t_p_neg_decay3")]
    embed_cols   = [c for c in df.columns if c.startswith("e_embed_pca16_")]
    evt_cols     = [c for c in df.columns if c == "event_weight"]

    configs = {
        "Tech only": tech_feats,
        "Tech + Score(decay) + EventW": tech_feats + score_cols + evt_cols,
        "Tech + Triplet(decay) + EventW": tech_feats + trip_cols + evt_cols,
        "Tech + PCA16 + EventW": tech_feats + embed_cols + evt_cols,
    }

    rows = []
    for name, feats in configs.items():
        if not feats: 
            rows.append((name, np.nan, np.nan, 0, 0, None)); continue
        acc, auc, ntr, nte, split = eval_lr(df, feats, label)
        rows.append((name, acc, auc, ntr, nte, pd.Timestamp(split).date() if split is not None else None))
    res = pd.DataFrame(rows, columns=["Config","Accuracy","AUC","Train N","Test N","Split Date"]).round(3)
    print("\n=== Logistic Regression (simple: decay + event weight) ===")
    print(res.to_string(index=False))

if __name__ == "__main__":
    main()
