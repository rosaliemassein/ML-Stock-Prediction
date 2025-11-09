#!/usr/bin/env python3
"""
Feature selection using top-K features from the BEST model in train_xgboost.py.

This script:
1. Loads the best model configuration (e.g., "Technical + Enhanced Sentiment")
2. Reconstructs the exact same dataset used by that best model
3. Uses feature importance rankings from that specific best model
4. Trains XGBoost, LightGBM, and Random Forest with top-K features (10, 20, 30, 50)
5. Compares performance to see if fewer features maintain accuracy

This validates whether a simpler model with fewer features can match the full model's performance.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from itertools import product


def split_data(df, feature_cols, label_col="y", split="chrono", val_size=0.2, test_size=0.1, seed=42):
    """Split data into train/validation/test sets."""
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


def tune_xgboost(X_train, y_train, X_val, y_val, seed=42):
    """Tune XGBoost hyperparameters for the given feature set."""
    param_grid = {
        'max_depth': [2, 3, 5],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [200, 300],
        'min_child_weight': [5, 10],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'gamma': [0, 0.1],
    }
    
    best_auc = 0
    best_params = None
    
    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        params.update({
            'random_state': seed,
            'objective': 'binary:logistic',
            'eval_metric': 'auc'
        })
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        
        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_prob)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params.copy()
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)
    return model, best_params


def tune_lightgbm(X_train, y_train, X_val, y_val, seed=42):
    """Tune LightGBM hyperparameters for the given feature set."""
    param_grid = {
        'max_depth': [2, 3, 5],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [200, 300],
        'min_child_samples': [20, 40],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_split_gain': [0, 0.1],
    }
    
    best_auc = 0
    best_params = None
    
    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        params.update({
            'random_state': seed,
            'objective': 'binary',
            'metric': 'auc',
            'verbose': -1
        })
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_prob)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params.copy()
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, best_params


def tune_random_forest(X_train, y_train, X_val, y_val, seed=42):
    """Tune Random Forest hyperparameters for the given feature set."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt'],
        'class_weight': ['balanced', None],
    }
    
    best_auc = 0
    best_params = None
    
    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        params.update({
            'random_state': seed,
            'n_jobs': -1
        })
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_prob)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params.copy()
    
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, best_params


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate model on train/val/test sets."""
    results = {}
    
    for split_name, X, y in [('train', X_train, y_train), 
                              ('val', X_val, y_val), 
                              ('test', X_test, y_test)]:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        
        results[f'{split_name}_acc'] = acc
        results[f'{split_name}_auc'] = auc
    
    return results


def main():
    ap = argparse.ArgumentParser(description='Train models with top-K features from validation set')
    ap.add_argument("--split", choices=["random","chrono"], default="chrono",
                    help="random or chrono (leak-free)")
    ap.add_argument("--top-k", default="10,20,30,50",
                    help="Comma-separated list of K values for top-K features")
    ap.add_argument("--val-size", type=float, default=0.2,
                    help="Validation set size (default: 0.2)")
    ap.add_argument("--test-size", type=float, default=0.1,
                    help="Test set size (default: 0.1)")
    ap.add_argument("--tune", action="store_true", default=True,
                    help="Tune hyperparameters for each K (default: True)")
    ap.add_argument("--no-tune", dest="tune", action="store_false",
                    help="Skip hyperparameter tuning")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    args = ap.parse_args()
    
    top_k_values = [int(k) for k in args.top_k.split(',')]
    
    print(f"\n{'='*70}")
    print(f"=== Top-K Feature Selection from Best Model ===")
    print(f"Split: {args.split}, K values: {top_k_values}, Tune: {args.tune}")
    print(f"{'='*70}\n")
    
    metadata_path = "reports/best_model_metadata.csv"
    if not Path(metadata_path).exists():
        print(f"[ERROR] Best model metadata not found: {metadata_path}")
        print(f"        Please run train_xgboost.py first to generate best model metadata.")
        return
    
    best_meta = pd.read_csv(metadata_path)
    best_config = best_meta['config'].iloc[0]
    best_val_auc = best_meta['val_auc'].iloc[0]
    best_test_auc = best_meta['test_auc'].iloc[0]
    
    print(f"[info] Best model from train_xgboost.py:")
    print(f"       Configuration: {best_config}")
    print(f"       Val AUC: {best_val_auc:.4f}")
    print(f"       Test AUC: {best_test_auc:.4f}")
    print(f"       Features: {best_meta['n_features'].iloc[0]}")
    
    importance_path = "reports/best_model_feature_importance.csv"
    if not Path(importance_path).exists():
        print(f"[ERROR] Feature importance file not found: {importance_path}")
        return
    
    feature_importance = pd.read_csv(importance_path)
    print(f"\n[info] Loaded {len(feature_importance)} features from best model")
    print(f"[info] Top 10 features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"       {i+1}. {row['feature']}: {row['importance']:.6f}")
    
    tech = pd.read_csv("data/processed/merge_T_only_h5.csv", parse_dates=["date"])
    
    if "Enhanced Sentiment" in best_config:
        sent_enhanced = pd.read_csv("data/processed/sentiment_features_enhanced.csv", parse_dates=["date"])
        sent_enhanced["ticker"] = sent_enhanced["ticker"].astype(str).str.upper()
        df_merged = tech.merge(sent_enhanced, on=["ticker","date"], how="inner")
        print(f"\n[info] Using Enhanced Sentiment features (matching best model)")
    elif "Triplet" in best_config:
        triplet = pd.read_csv("data/processed/sent_headlines_triplet.csv", parse_dates=["date"])
        triplet["ticker"] = triplet["ticker"].astype(str).str.upper()
        df_merged = tech.merge(triplet, on=["ticker","date"], how="inner")
        print(f"\n[info] Using Triplet features (matching best model)")
    elif "Score" in best_config:
        score = pd.read_csv("data/processed/sent_headlines_score.csv", parse_dates=["date"])
        score["ticker"] = score["ticker"].astype(str).str.upper()
        df_merged = tech.merge(score, on=["ticker","date"], how="inner")
        print(f"\n[info] Using Score features (matching best model)")
    else:
        df_merged = tech.copy()
        print(f"\n[info] Using Technical features only (matching best model)")
    
    print(f"[info] Dataset: {len(df_merged)} rows")
    print(f"[info] Date range: {df_merged['date'].min().date()} to {df_merged['date'].max().date()}")
    
    all_features = [f for f in feature_importance['feature'].tolist() if f in df_merged.columns]
    print(f"[info] Available features: {len(all_features)}/{len(feature_importance)}")
    
    all_results = []
    
    for k in top_k_values:
        print(f"\n{'='*70}")
        print(f"=== Training with Top-{k} Features ===")
        print(f"{'='*70}")
        
        top_features = all_features[:k]
        print(f"[info] Selected top-{k} features:")
        for i, feat in enumerate(top_features[:5], 1):
            print(f"       {i}. {feat}")
        if k > 5:
            print(f"       ... and {k-5} more")
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df_merged, top_features, label_col="y", 
            split=args.split, val_size=args.val_size, 
            test_size=args.test_size, seed=args.seed
        )
        
        print(f"\n[info] Data splits:")
        print(f"       Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
        val_dist = pd.Series(y_val).value_counts(normalize=True).sort_index()
        test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
        print(f"       Train class dist: {train_dist.values}")
        print(f"       Val class dist: {val_dist.values}")
        print(f"       Test class dist: {test_dist.values}")
        
        models = {
            'XGBoost': tune_xgboost,
            'LightGBM': tune_lightgbm,
            'RandomForest': tune_random_forest
        }
        
        n_combos_xgb = 3 * 2 * 2 * 2 * 2  # 48 combinations
        n_combos_lgb = 3 * 2 * 2 * 2 * 2  # 48 combinations
        n_combos_rf = 3 * 3 * 2 * 2 * 2   # 72 combinations
        
        if args.tune:
            print(f"\n[info] Tuning hyperparameters:")
            print(f"       XGBoost: {n_combos_xgb} combinations")
            print(f"       LightGBM: {n_combos_lgb} combinations")
            print(f"       RandomForest: {n_combos_rf} combinations")
        
        for model_name, tune_func in models.items():
            print(f"\n--- {model_name} with Top-{k} ---")
            
            if args.tune:
                model, best_params = tune_func(X_train, y_train, X_val, y_val, seed=args.seed)
                print(f"  Best params: {', '.join(f'{k}={v}' for k, v in best_params.items() if k not in ['random_state', 'objective', 'eval_metric', 'metric', 'verbose', 'n_jobs'])}")
            else:
                if model_name == 'XGBoost':
                    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=200, random_state=args.seed)
                    model.fit(X_train, y_train, verbose=False)
                elif model_name == 'LightGBM':
                    model = lgb.LGBMClassifier(max_depth=3, learning_rate=0.05, n_estimators=200, random_state=args.seed, verbose=-1)
                    model.fit(X_train, y_train)
                else:
                    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=args.seed, n_jobs=-1)
                    model.fit(X_train, y_train)
                best_params = {}
            
            results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
            
            print(f"  Train: Acc={results['train_acc']:.4f}, AUC={results['train_auc']:.4f}")
            print(f"  Val:   Acc={results['val_acc']:.4f}, AUC={results['val_auc']:.4f}")
            print(f"  Test:  Acc={results['test_acc']:.4f}, AUC={results['test_auc']:.4f}")
            
            all_results.append({
                'model': model_name,
                'top_k': k,
                'n_features': k,
                'train_acc': results['train_acc'],
                'train_auc': results['train_auc'],
                'val_acc': results['val_acc'],
                'val_auc': results['val_auc'],
                'test_acc': results['test_acc'],
                'test_auc': results['test_auc'],
                'tuned': args.tune
            })
    
    results_df = pd.DataFrame(all_results)
    Path("reports").mkdir(parents=True, exist_ok=True)
    results_df.to_csv("reports/top_k_feature_results.csv", index=False)
    
    print(f"\n{'='*70}")
    print("=== Summary: Top-K Feature Selection Results ===")
    print(f"{'='*70}\n")
    
    print(f"Original Best Model ({best_config}):")
    print(f"  All {best_meta['n_features'].iloc[0]} features")
    print(f"  Val AUC: {best_val_auc:.4f}, Test AUC: {best_test_auc:.4f}")
    
    print("\n--- Validation AUC by Model and K ---")
    pivot_val = results_df.pivot(index='model', columns='top_k', values='val_auc')
    print(pivot_val.round(4))
    
    print("\n--- Test AUC by Model and K ---")
    pivot_test = results_df.pivot(index='model', columns='top_k', values='test_auc')
    print(pivot_test.round(4))
    
    print(f"\n[info] Full results saved to: reports/top_k_feature_results.csv")
    
    best_idx = results_df['val_auc'].idxmax()
    best = results_df.iloc[best_idx]
    print(f"\n{'='*70}")
    print(f"Best Reduced Model (by Validation AUC):")
    print(f"  Model: {best['model']}")
    print(f"  Top-K: {int(best['top_k'])} features (reduced from {best_meta['n_features'].iloc[0]})")
    print(f"  Val AUC: {best['val_auc']:.4f} (original: {best_val_auc:.4f})")
    print(f"  Test AUC: {best['test_auc']:.4f} (original: {best_test_auc:.4f})")
    print(f"  Performance delta: Val {best['val_auc']-best_val_auc:+.4f}, Test {best['test_auc']-best_test_auc:+.4f}")
    print(f"  Feature reduction: {(1 - best['top_k']/best_meta['n_features'].iloc[0])*100:.1f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

