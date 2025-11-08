#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(df, feature_cols, label_col="y", split="random", test_size=0.2, seed=42):
    df = df.dropna(subset=feature_cols + [label_col]).copy()

    if split == "chrono":
        # time-based split: train on earlier dates, test on later ones
        df = df.sort_values("date")
        dates = np.sort(df["date"].unique())
        cut = int((1 - test_size) * len(dates))
        split_date = dates[cut]
        train = df[df["date"] < split_date]
        test  = df[df["date"] >= split_date]
        X_train, y_train = train[feature_cols], train[label_col]
        X_test,  y_test  = test[feature_cols],  test[label_col]
    else:
        # random split (can cause temporal leakage!)
        X = df[feature_cols]
        y = df[label_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    return acc, auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random","chrono"], default="random",
                    help="random (replicates earlier results) or chrono (leak-free)")
    args = ap.parse_args()

    tech = pd.read_csv("data/processed/merge_T_only_h5.csv", parse_dates=["date"])
    score = pd.read_csv("data/processed/sent_headlines_score.csv", parse_dates=["date"])
    triplet = pd.read_csv("data/processed/sent_headlines_triplet.csv", parse_dates=["date"])
    embed = pd.read_csv("data/processed/sent_headlines_embed_pca16.csv", parse_dates=["date"])

    tech_features = ["ret_1d","sma_5","mom_5","sma_10","mom_10","sma_20","mom_20","vol_20","rsi"]

    print(f"=== Running Logistic Regression Comparisons (split={args.split}) ===")

    acc_tech, auc_tech = evaluate_model(tech, tech_features, split=args.split)

    merged_score = tech.merge(score, on=["ticker","date"], how="left").dropna(subset=["sent_score"])
    acc_score, auc_score = evaluate_model(merged_score, tech_features+["sent_score"], split=args.split)

    merged_triplet = tech.merge(triplet, on=["ticker","date"], how="left").dropna(subset=["p_pos","p_neu","p_neg"])
    acc_triplet, auc_triplet = evaluate_model(merged_triplet, tech_features+["p_pos","p_neu","p_neg"], split=args.split)

    merged_embed = tech.merge(embed, on=["ticker","date"], how="left")
    emb_cols = [c for c in merged_embed.columns if c.startswith("emb_")]
    merged_embed = merged_embed.dropna(subset=emb_cols)
    acc_embed, auc_embed = evaluate_model(merged_embed, tech_features+emb_cols, split=args.split)

    results = pd.DataFrame({
        "Model": ["Technical only","Technical + Score","Technical + Triplet","Technical + PCA16"],
        "Accuracy": [acc_tech, acc_score, acc_triplet, acc_embed],
        "AUC": [auc_tech, auc_score, auc_triplet, auc_embed]
    })
    print("\n=== Logistic Regression Comparison ===")
    print(results.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
