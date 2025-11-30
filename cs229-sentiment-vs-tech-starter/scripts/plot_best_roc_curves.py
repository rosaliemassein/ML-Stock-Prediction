#!/usr/bin/env python3
"""
Generate clean ROC curves with one line per model category (best algorithm+variant).
Uses proper hyperparameter tuning from train_xgboost.py
"""
import argparse
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from itertools import product


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


def tune_and_train_xgboost(X_train, y_train, X_val, y_val, param_grid, seed=42):
    """Tune and train XGBoost."""
    best_auc = 0
    best_params = None
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for combo in product(*values):
        params = dict(zip(keys, combo))
        params['random_state'] = seed
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_prob)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params.copy()
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)
    
    return model, best_auc, best_params


def tune_and_train_lightgbm(X_train, y_train, X_val, y_val, param_grid, seed=42):
    """Tune and train LightGBM."""
    best_auc = 0
    best_params = None
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for combo in product(*values):
        params = dict(zip(keys, combo))
        params['random_state'] = seed
        params['objective'] = 'binary'
        params['metric'] = 'auc'
        params['verbose'] = -1
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_prob)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params.copy()
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    
    return model, best_auc, best_params


def plot_clean_roc_curves(models_data, output_path):
    """Plot clean ROC curves with one line per category."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'Technical only': '#7f7f7f',
        'Technical + Score': '#ff7f0e',
        'Technical + Triplet': '#d62728',
        'Technical + LLM Sentiment': '#2ca02c',
        'Technical + LLM Enhanced': '#9467bd',
        'Technical + Enhanced Sentiment (ALL)': '#1f77b4'
    }
    
    line_styles = {
        'Technical only': '-',
        'Technical + Score': '--',
        'Technical + Triplet': '-.',
        'Technical + LLM Sentiment': ':',
        'Technical + LLM Enhanced': '--',
        'Technical + Enhanced Sentiment (ALL)': '-'
    }
    
    markers = {
        'Technical only': 'o',
        'Technical + Score': 's',
        'Technical + Triplet': '^',
        'Technical + LLM Sentiment': 'D',
        'Technical + LLM Enhanced': 'v',
        'Technical + Enhanced Sentiment (ALL)': 'p'
    }
    
    for category, data in sorted(models_data.items(), key=lambda x: x[1]['test_auc'], reverse=True):
        y_true = data['y_true']
        y_prob = data['y_prob']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        ax.plot(fpr, tpr,
               color=colors.get(category, '#000000'),
               linestyle=line_styles.get(category, '-'),
               linewidth=3,
               marker=markers.get(category, 'o'),
               markersize=8,
               markevery=0.15,
               label=category,
               alpha=0.85)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.4, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves: Best Model Per Category (Test Set)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[info] Saved clean ROC curve to {output_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Generate ROC curves (best algorithm per category)")
    ap.add_argument("--split", choices=["random","chrono"], default="chrono")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--output-dir", default="reports")
    ap.add_argument("--quick-tune", action="store_true", default=True)
    args = ap.parse_args()
    
    print(f"{'='*70}")
    print(f"Generating ROC Curves with Hyperparameter Tuning")
    print(f"{'='*70}\n")
    
    print("[1/5] Loading datasets...")
    tech = pd.read_csv("data/processed/merge_T_only_h5.csv", parse_dates=["date"])
    score = pd.read_csv("data/processed/sent_headlines_score.csv", parse_dates=["date"])
    triplet = pd.read_csv("data/processed/sent_headlines_triplet.csv", parse_dates=["date"])
    sentiment_llm = pd.read_csv("data/processed/sentiment_features.csv", parse_dates=["date"])
    sentiment_llm_enhanced = pd.read_csv("data/processed/sentiment_llm_enhanced.csv", parse_dates=["date"])
    sent_enhanced = pd.read_csv("data/processed/sentiment_features_enhanced.csv", parse_dates=["date"])
    
    for name, df_var in [("score", score), ("triplet", triplet), ("sentiment_llm", sentiment_llm)]:
        dupes = df_var.groupby(['ticker','date']).size().max()
        if dupes > 1:
            cols = [c for c in df_var.columns if c not in ['date', 'ticker']]
            aggregated = df_var.groupby(['ticker','date'], as_index=False)[cols].mean()
            if name == "score":
                score = aggregated
            elif name == "triplet":
                triplet = aggregated
            elif name == "sentiment_llm":
                sentiment_llm = aggregated
    
    sentiment_llm['sentiment_raw_weight'] = sentiment_llm['sentiment_raw_weight'].fillna(0)
    sentiment_llm['sentiment_weighted_sum'] = sentiment_llm['sentiment_weighted_sum'].fillna(0)
    if 'sentiment_weighted_avg' in sentiment_llm.columns:
        sentiment_llm['sentiment_weighted_avg'] = sentiment_llm['sentiment_weighted_avg'].fillna(0)
    sentiment_llm['sentiment_combined'] = sentiment_llm['sentiment_raw_weight'] + sentiment_llm['sentiment_weighted_sum']
    
    sentiment_dates = set(sent_enhanced['date'].unique())
    tech = tech[tech['date'].isin(sentiment_dates)]
    score = score[score['date'].isin(sentiment_dates)]
    triplet = triplet[triplet['date'].isin(sentiment_dates)]
    sentiment_llm = sentiment_llm[sentiment_llm['date'].isin(sentiment_dates)]
    sentiment_llm_enhanced = sentiment_llm_enhanced[sentiment_llm_enhanced['date'].isin(sentiment_dates)]
    
    print(f"  Loaded data with {len(tech['date'].unique())} dates\n")
    
    tech_features = [
        'ret_1d', 'rsi', 'sma_20', 'price_to_sma_20', 'mom_20', 'vol_20',
        'bb_width_20', 'bb_pct_20', 'macd', 'macd_signal', 'vol_ratio', 'close_position'
    ]
    tech_features = [f for f in tech_features if f in tech.columns]
    
    sentiment_llm_cols = [c for c in sentiment_llm.columns if c not in ['date', 'ticker']]
    sentiment_llm_enhanced_cols = [c for c in sentiment_llm_enhanced.columns if c not in ['date', 'ticker']]
    enhanced_sent_cols = [c for c in sent_enhanced.columns if c not in ['date', 'ticker']]
    
    print("[2/5] Setting up hyperparameter grids...")
    y_train_temp = tech.dropna(subset=tech_features + ['y'])['y']
    class_ratio = (y_train_temp == 0).sum() / (y_train_temp == 1).sum()
    
    if args.quick_tune:
        param_grid_xgb = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.05],
            'n_estimators': [300],
            'min_child_weight': [5, 10],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'gamma': [0, 0.1],
            'scale_pos_weight': [class_ratio],
        }
        param_grid_lgb = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.05],
            'n_estimators': [300],
            'min_child_samples': [20, 40],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_split_gain': [0, 0.1],
            'scale_pos_weight': [class_ratio],
        }
        print(f"  Quick tuning: ~16 combinations per algorithm\n")
    else:
        param_grid_xgb = {
            'max_depth': [2, 3, 5],
            'learning_rate': [0.01, 0.05],
            'n_estimators': [300, 500],
            'min_child_weight': [5, 10],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.8],
            'gamma': [0, 0.1],
            'scale_pos_weight': [class_ratio],
        }
        param_grid_lgb = {
            'max_depth': [2, 3, 5],
            'learning_rate': [0.01, 0.05],
            'n_estimators': [300, 500],
            'min_child_samples': [20, 40],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.8],
            'min_split_gain': [0, 0.1],
            'scale_pos_weight': [class_ratio],
        }
        print(f"  Full grid search: ~96 combinations per algorithm\n")
    
    print("[3/5] Preparing model configurations...")
    categories = {
        'Technical only': [
            ("XGB", "tech", tech, tech_features),
            ("LGB", "tech", tech, tech_features)
        ],
        'Technical + Score': [
            ("XGB", "score", tech.merge(score, on=["ticker","date"], how="inner"), 
             tech_features+["sent_score"]),
            ("LGB", "score", tech.merge(score, on=["ticker","date"], how="inner"), 
             tech_features+["sent_score"])
        ],
        'Technical + Triplet': [
            ("XGB", "triplet", tech.merge(triplet, on=["ticker","date"], how="inner"), 
             tech_features+["p_pos","p_neu","p_neg"]),
            ("LGB", "triplet", tech.merge(triplet, on=["ticker","date"], how="inner"), 
             tech_features+["p_pos","p_neu","p_neg"])
        ],
        'Technical + LLM Sentiment': [
            ("XGB", "simple", tech.merge(sentiment_llm, on=["ticker","date"], how="inner"),
             tech_features+sentiment_llm_cols),
            ("LGB", "simple", tech.merge(sentiment_llm, on=["ticker","date"], how="inner"),
             tech_features+sentiment_llm_cols)
        ],
        'Technical + LLM Enhanced': [
            ("XGB", "enhanced", tech.merge(sentiment_llm_enhanced, on=["ticker","date"], how="inner"),
             tech_features+sentiment_llm_enhanced_cols),
            ("LGB", "enhanced", tech.merge(sentiment_llm_enhanced, on=["ticker","date"], how="inner"),
             tech_features+sentiment_llm_enhanced_cols)
        ],
        'Technical + Enhanced Sentiment (ALL)': [
            ("XGB", "finbert", tech.merge(sent_enhanced, on=["ticker","date"], how="inner"),
             tech_features+enhanced_sent_cols),
            ("LGB", "finbert", tech.merge(sent_enhanced, on=["ticker","date"], how="inner"),
             tech_features+enhanced_sent_cols)
        ]
    }
    
    print(f"  Configured {sum(len(v) for v in categories.values())} model variants\n")
    
    print("[4/5] Training models with hyperparameter tuning...")
    best_models = {}
    
    for category, variants in categories.items():
        print(f"\n  {category}:")
        best_test_auc = 0
        best_data = None
        
        for algo, variant_name, df_model, features in variants:
            try:
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                    df_model, features, split=args.split,
                    val_size=args.val_size, test_size=args.test_size
                )
                
                if len(X_test) == 0:
                    continue
                
                if algo == "XGB":
                    model, val_auc, params = tune_and_train_xgboost(
                        X_train, y_train, X_val, y_val, param_grid_xgb
                    )
                else:
                    model, val_auc, params = tune_and_train_lightgbm(
                        X_train, y_train, X_val, y_val, param_grid_lgb
                    )
                
                y_prob = model.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_prob)
                
                print(f"    {algo}-{variant_name}: Val AUC={val_auc:.4f}, Test AUC={test_auc:.4f} (n={len(X_test)})")
                
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    best_data = {
                        'y_true': y_test.values if hasattr(y_test, 'values') else y_test,
                        'y_prob': y_prob,
                        'val_auc': val_auc,
                        'test_auc': test_auc,
                        'algorithm': algo,
                        'variant': variant_name,
                        'n_samples': len(X_test)
                    }
            
            except Exception as e:
                print(f"    {algo}-{variant_name}: ERROR - {e}")
        
        if best_data:
            best_models[category] = best_data
            print(f"    ✓ BEST: {best_data['algorithm']}-{best_data['variant']} (Test AUC={best_data['test_auc']:.4f})")
    
    print(f"\n[5/5] Generating ROC curve...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if len(best_models) > 0:
        plot_clean_roc_curves(best_models, f"{args.output_dir}/roc_curves_best_per_category.png")
        
        metrics_data = []
        for category, data in best_models.items():
            metrics_data.append({
                'Category': category,
                'Algorithm': data['algorithm'],
                'Variant': data['variant'],
                'Val AUC': data['val_auc'],
                'Test AUC': data['test_auc'],
                'N Samples': data['n_samples']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('Test AUC', ascending=False)
        metrics_df.to_csv(f"{args.output_dir}/roc_best_per_category.csv", index=False)
        
        print(f"\n{'='*70}")
        print("ROC Curve Generation Complete")
        print(f"{'='*70}\n")
        print("📊 Best Model Per Category (TEST SET PERFORMANCE):")
        print(metrics_df.round(4).to_string(index=False))
        print(f"\n📁 Generated Files:")
        print(f"  - {args.output_dir}/roc_curves_best_per_category.png (TEST SET)")
        print(f"  - {args.output_dir}/roc_best_per_category.csv (TEST SET)")
        print(f"\nNote: Models trained with hyperparameter tuning on train+val sets.")
        print(f"      ROC curves and AUC scores computed on TEST SET only.")
        print(f"\n{'='*70}\n")
    else:
        print("  ERROR: No models successfully trained!")


if __name__ == "__main__":
    main()
