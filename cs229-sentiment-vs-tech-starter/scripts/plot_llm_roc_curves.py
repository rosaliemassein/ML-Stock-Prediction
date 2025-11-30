#!/usr/bin/env python3
"""
Generate ROC curves for all LLM sentiment model configurations.
Compares: Technical only, Simple Sentiment, LLM Sentiment, Enhanced LLM, Enhanced FinBERT
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb


def split_data(df, feature_cols, label_col="y", split="chrono", val_size=0.15, test_size=0.15, seed=42):
    """Split data chronologically."""
    df = df.dropna(subset=feature_cols + [label_col]).copy()
    
    if split == "chrono":
        df = df.sort_values("date")
        dates = np.sort(df["date"].unique())
        
        train_cut = int((1 - val_size - test_size) * len(dates))
        val_cut = int((1 - test_size) * len(dates))
        
        train_date = dates[train_cut]
        val_date = dates[val_cut]
        
        train = df[df["date"] < train_date]
        val = df[(df["date"] >= train_date) & (df["date"] < val_date)]
        test = df[df["date"] >= val_date]
        
        X_train, y_train = train[feature_cols], train[label_col]
        X_val, y_val = val[feature_cols], val[label_col]
        X_test, y_test = test[feature_cols], test[label_col]
    else:
        n = len(df)
        np.random.seed(seed)
        indices = np.random.permutation(n)
        
        test_n = int(test_size * n)
        val_n = int(val_size * n)
        
        test_indices = indices[:test_n]
        val_indices = indices[test_n:test_n + val_n]
        train_indices = indices[test_n + val_n:]
        
        train = df.iloc[train_indices]
        val = df.iloc[val_indices]
        test = df.iloc[test_indices]
        
        X_train, y_train = train[feature_cols], train[label_col]
        X_val, y_val = val[feature_cols], val[label_col]
        X_test, y_test = test[feature_cols], test[label_col]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, seed=42):
    """Train XGBoost with default parameters."""
    model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=300,
        random_state=seed,
        objective='binary:logistic',
        eval_metric='auc'
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def plot_roc_curves(models_data, output_path, title="ROC Curves - Model Comparison"):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(12, 9))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for i, (model_name, data) in enumerate(models_data.items()):
        y_true = data['y_true']
        y_prob = data['y_prob']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.plot(fpr, tpr, 
                color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)], 
                linewidth=2.5,
                marker=markers[i % len(markers)],
                markersize=6,
                markevery=0.1,
                label=f'{model_name} (AUC = {auc_score:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random (AUC = 0.5000)')
    
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[info] Saved ROC curves to {output_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Generate ROC curves for all LLM sentiment models")
    ap.add_argument("--split", choices=["random","chrono"], default="chrono",
                    help="Data split strategy")
    ap.add_argument("--val-size", type=float, default=0.15, help="Validation size")
    ap.add_argument("--test-size", type=float, default=0.15, help="Test size")
    ap.add_argument("--output-dir", default="reports", help="Output directory")
    args = ap.parse_args()
    
    print(f"{'='*70}")
    print(f"Generating ROC Curves for All Models")
    print(f"{'='*70}\n")
    
    # Load data
    print("[1/4] Loading datasets...")
    tech = pd.read_csv("data/processed/merge_T_only_h5.csv", parse_dates=["date"])
    score = pd.read_csv("data/processed/sent_headlines_score.csv", parse_dates=["date"])
    triplet = pd.read_csv("data/processed/sent_headlines_triplet.csv", parse_dates=["date"])
    sentiment_llm = pd.read_csv("data/processed/sentiment_features.csv", parse_dates=["date"])
    sentiment_llm_enhanced = pd.read_csv("data/processed/sentiment_llm_enhanced.csv", parse_dates=["date"])
    sent_enhanced = pd.read_csv("data/processed/sentiment_features_enhanced.csv", parse_dates=["date"])
    
    # Aggregate duplicates
    for name, df_var in [("score", score), ("triplet", triplet), ("sentiment_llm", sentiment_llm)]:
        dupes = df_var.groupby(['ticker','date']).size().max()
        if dupes > 1:
            print(f"  [warn] {name}: aggregating {dupes} rows per ticker-date...")
            cols = [c for c in df_var.columns if c not in ['date', 'ticker']]
            aggregated = df_var.groupby(['ticker','date'], as_index=False)[cols].mean()
            if name == "score":
                score = aggregated
            elif name == "triplet":
                triplet = aggregated
            elif name == "sentiment_llm":
                sentiment_llm = aggregated
    
    # Handle NaN in sentiment_llm
    sentiment_llm['sentiment_raw_weight'] = sentiment_llm['sentiment_raw_weight'].fillna(0)
    sentiment_llm['sentiment_weighted_sum'] = sentiment_llm['sentiment_weighted_sum'].fillna(0)
    if 'sentiment_weighted_avg' in sentiment_llm.columns:
        sentiment_llm['sentiment_weighted_avg'] = sentiment_llm['sentiment_weighted_avg'].fillna(0)
    sentiment_llm['sentiment_combined'] = sentiment_llm['sentiment_raw_weight'] + sentiment_llm['sentiment_weighted_sum']
    
    # Filter to sentiment coverage
    sentiment_dates = set(sent_enhanced['date'].unique())
    tech = tech[tech['date'].isin(sentiment_dates)]
    score = score[score['date'].isin(sentiment_dates)]
    triplet = triplet[triplet['date'].isin(sentiment_dates)]
    sentiment_llm = sentiment_llm[sentiment_llm['date'].isin(sentiment_dates)]
    sentiment_llm_enhanced = sentiment_llm_enhanced[sentiment_llm_enhanced['date'].isin(sentiment_dates)]
    
    print(f"  Loaded data with {len(tech['date'].unique())} dates\n")
    
    # Define features
    tech_features = [
        'ret_1d', 'rsi', 'sma_20', 'price_to_sma_20', 'mom_20', 'vol_20',
        'bb_width_20', 'bb_pct_20', 'macd', 'macd_signal', 'vol_ratio', 'close_position'
    ]
    tech_features = [f for f in tech_features if f in tech.columns]
    
    sentiment_llm_cols = [c for c in sentiment_llm.columns if c not in ['date', 'ticker']]
    sentiment_llm_enhanced_cols = [c for c in sentiment_llm_enhanced.columns if c not in ['date', 'ticker']]
    enhanced_sent_cols = [c for c in sent_enhanced.columns if c not in ['date', 'ticker']]
    
    # Define model configurations
    print("[2/4] Preparing model configurations...")
    configs = [
        ("Technical Only", tech, tech_features),
        ("Technical + Score", tech.merge(score, on=["ticker","date"], how="inner"), 
         tech_features+["sent_score"]),
        ("Technical + Triplet", tech.merge(triplet, on=["ticker","date"], how="inner"), 
         tech_features+["p_pos","p_neu","p_neg"]),
        ("Technical + LLM", tech.merge(sentiment_llm, on=["ticker","date"], how="inner"),
         tech_features+sentiment_llm_cols),
        ("Technical + LLM Enhanced", tech.merge(sentiment_llm_enhanced, on=["ticker","date"], how="inner"),
         tech_features+sentiment_llm_enhanced_cols),
        ("Technical + FinBERT Enhanced", tech.merge(sent_enhanced, on=["ticker","date"], how="inner"),
         tech_features+enhanced_sent_cols),
    ]
    
    print(f"  Configured {len(configs)} models\n")
    
    # Train and evaluate models
    print("[3/4] Training models and collecting predictions...")
    models_data = {}
    
    for i, (config_name, df_model, features) in enumerate(configs, 1):
        print(f"  [{i}/{len(configs)}] {config_name}...")
        
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                df_model, features, split=args.split,
                val_size=args.val_size, test_size=args.test_size
            )
            
            if len(X_test) == 0:
                print(f"       SKIP: No test data after split")
                continue
            
            model = train_model(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            models_data[config_name] = {
                'y_true': y_test.values if hasattr(y_test, 'values') else y_test,
                'y_prob': y_prob,
                'n_samples': len(X_test)
            }
            
            auc = roc_auc_score(y_test, y_prob)
            print(f"       Test AUC: {auc:.4f} (n={len(X_test)})")
            
        except Exception as e:
            print(f"       ERROR: {e}")
    
    print()
    
    # Generate ROC curves
    print("[4/4] Generating ROC curve plots...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if len(models_data) > 0:
        # All models
        plot_roc_curves(models_data, 
                       f"{args.output_dir}/roc_curves_all_models.png",
                       f"ROC Curves - All LLM Sentiment Models ({args.split.capitalize()} Split)")
        
        # Comparison: Simple vs Enhanced
        llm_models = {k: v for k, v in models_data.items() if 'LLM' in k or 'Technical Only' in k}
        if len(llm_models) > 0:
            plot_roc_curves(llm_models,
                           f"{args.output_dir}/roc_curves_llm_comparison.png",
                           "ROC Curves - LLM Sentiment Comparison")
        
        # Save metrics table
        metrics_data = []
        for model_name, data in models_data.items():
            auc = roc_auc_score(data['y_true'], data['y_prob'])
            metrics_data.append({
                'Model': model_name,
                'Test AUC': auc,
                'N Samples': data['n_samples']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('Test AUC', ascending=False)
        metrics_df.to_csv(f"{args.output_dir}/roc_metrics.csv", index=False)
        
        print(f"\n{'='*70}")
        print("ROC Curve Generation Complete")
        print(f"{'='*70}\n")
        print("📊 Test AUC Results:")
        print(metrics_df.round(4).to_string(index=False))
        print(f"\n📁 Generated Files:")
        print(f"  - {args.output_dir}/roc_curves_all_models.png")
        print(f"  - {args.output_dir}/roc_curves_llm_comparison.png")
        print(f"  - {args.output_dir}/roc_metrics.csv")
        print(f"\n{'='*70}\n")
    else:
        print("  ERROR: No models successfully trained!")


if __name__ == "__main__":
    main()



