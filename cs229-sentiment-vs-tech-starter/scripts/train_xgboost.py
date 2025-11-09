#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from itertools import product

def split_data(df, feature_cols, label_col="y", split="random", val_size=0.2, test_size=0.1, seed=42):
    """
    Split data into train/validation/test sets.
    
    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        label_col: Name of label column
        split: 'random' or 'chrono' (chronological)
        val_size: Fraction of data for validation
        test_size: Fraction of data for test
        seed: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = df.dropna(subset=feature_cols + [label_col]).copy()

    if split == "chrono":
        # time-based split: train on earlier dates, val on middle dates, test on later ones
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
        # random split (can cause temporal leakage!)
        n = len(df)
        np.random.seed(seed)
        indices = np.random.permutation(n)
        
        test_n = int(test_size * n)
        val_n = int(val_size * n)
        
        test_indices = indices[:test_n]
        val_indices = indices[test_n:test_n + val_n]
        train_indices = indices[test_n + val_n:]
        
        # Keep as DataFrames to preserve feature names for LightGBM/XGBoost
        train = df.iloc[train_indices]
        val = df.iloc[val_indices]
        test = df.iloc[test_indices]
        
        X_train, y_train = train[feature_cols], train[label_col]
        X_val, y_val = val[feature_cols], val[label_col]
        X_test, y_test = test[feature_cols], test[label_col]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid=None, seed=42):
    """
    Tune XGBoost hyperparameters using validation set.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        param_grid: Dictionary of hyperparameters to search
        seed: Random seed
    
    Returns:
        best_params: Dictionary of best hyperparameters
        best_auc: Best validation AUC achieved
    """
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
    
    print(f"\n=== Hyperparameter Tuning ===")
    print(f"Searching through {np.prod([len(v) for v in param_grid.values()])} combinations...")
    
    best_auc = 0
    best_params = None
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for i, combo in enumerate(product(*values)):
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
            print(f"  New best AUC: {val_auc:.4f} with params: {combo}")
    
    print(f"\n=== Best Hyperparameters ===")
    for key, value in best_params.items():
        if key not in ['random_state', 'objective', 'eval_metric']:
            print(f"  {key}: {value}")
    print(f"  Validation AUC: {best_auc:.4f}")
    
    return best_params, best_auc


def evaluate_model_with_tuning(df, feature_cols, label_col="y", split="random", 
                                val_size=0.2, test_size=0.1, seed=42, 
                                param_grid=None, tune=True, return_model=False):
    """
    Evaluate XGBoost model with optional hyperparameter tuning.
    
    Returns:
        val_acc, val_auc, test_acc, test_auc, best_params, [model if return_model=True]
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, feature_cols, label_col, split, val_size, test_size, seed
    )
    
    print(f"  Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Print class distributions (convert to pandas Series for value_counts)
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    val_dist = pd.Series(y_val).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    print(f"  Train class distribution: {train_dist.values}")
    print(f"  Val class distribution: {val_dist.values}")
    print(f"  Test class distribution: {test_dist.values}")
    print(f"  Majority baseline (test): {test_dist.max():.4f}")
    
    if tune:
        best_params, val_auc = tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid, seed)
    else:
        best_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': seed,
            'objective': 'binary:logistic',
            'eval_metric': 'auc'
        }
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)
    
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_prob)
    
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob)
    
    if return_model:
        return val_acc, val_auc, test_acc, test_auc, best_params, model
    return val_acc, val_auc, test_acc, test_auc, best_params


def tune_hyperparameters_lgb(X_train, y_train, X_val, y_val, param_grid=None, seed=42):
    """
    Tune LightGBM hyperparameters using validation set.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        param_grid: Dictionary of hyperparameters to search
        seed: Random seed
    
    Returns:
        best_params: Dictionary of best hyperparameters
        best_auc: Best validation AUC achieved
    """
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'min_child_samples': [20, 30, 50],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
    
    print(f"\n=== Hyperparameter Tuning (LightGBM) ===")
    print(f"Searching through {np.prod([len(v) for v in param_grid.values()])} combinations...")
    
    best_auc = 0
    best_params = None
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for i, combo in enumerate(product(*values)):
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
            print(f"  New best AUC: {val_auc:.4f} with params: {combo}")
    
    print(f"\n=== Best Hyperparameters (LightGBM) ===")
    for key, value in best_params.items():
        if key not in ['random_state', 'objective', 'metric', 'verbose']:
            print(f"  {key}: {value}")
    print(f"  Validation AUC: {best_auc:.4f}")
    
    return best_params, best_auc


def evaluate_model_with_tuning_lgb(df, feature_cols, label_col="y", split="random", 
                                     val_size=0.2, test_size=0.2, seed=42, 
                                     param_grid=None, tune=True):
    """
    Evaluate LightGBM model with optional hyperparameter tuning.
    
    Returns:
        val_acc, val_auc, test_acc, test_auc, best_params
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, feature_cols, label_col, split, val_size, test_size, seed
    )
    
    print(f"  Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Print class distributions (convert to pandas Series for value_counts)
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    val_dist = pd.Series(y_val).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    print(f"  Train class distribution: {train_dist.values}")
    print(f"  Val class distribution: {val_dist.values}")
    print(f"  Test class distribution: {test_dist.values}")
    print(f"  Majority baseline (test): {test_dist.max():.4f}")
    
    if tune:
        best_params, val_auc = tune_hyperparameters_lgb(X_train, y_train, X_val, y_val, param_grid, seed)
    else:
        best_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': seed,
            'objective': 'binary',
            'metric': 'auc',
            'verbose': -1
        }
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_prob)
    
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob)
    
    return val_acc, val_auc, test_acc, test_auc, best_params


def tune_hyperparameters_rf(X_train, y_train, X_val, y_val, param_grid=None, seed=42):
    """
    Tune Random Forest hyperparameters using validation set.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        param_grid: Dictionary of hyperparameters to search
        seed: Random seed
    
    Returns:
        best_params: Dictionary of best hyperparameters
        best_auc: Best validation AUC achieved
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        }
    
    print(f"\n=== Hyperparameter Tuning (Random Forest) ===")
    print(f"Searching through {np.prod([len(v) for v in param_grid.values()])} combinations...")
    
    best_auc = 0
    best_params = None
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for i, combo in enumerate(product(*values)):
        params = dict(zip(keys, combo))
        params['random_state'] = seed
        params['n_jobs'] = -1
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_prob)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params.copy()
            print(f"  New best AUC: {val_auc:.4f} with params: {combo}")
    
    print(f"\n=== Best Hyperparameters (Random Forest) ===")
    for key, value in best_params.items():
        if key not in ['random_state', 'n_jobs']:
            print(f"  {key}: {value}")
    print(f"  Validation AUC: {best_auc:.4f}")
    
    return best_params, best_auc


def evaluate_model_with_tuning_rf(df, feature_cols, label_col="y", split="random", 
                                    val_size=0.2, test_size=0.2, seed=42, 
                                    param_grid=None, tune=True):
    """
    Evaluate Random Forest model with optional hyperparameter tuning.
    
    Returns:
        val_acc, val_auc, test_acc, test_auc, best_params
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, feature_cols, label_col, split, val_size, test_size, seed
    )
    
    print(f"  Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Print class distributions (convert to pandas Series for value_counts)
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    val_dist = pd.Series(y_val).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    print(f"  Train class distribution: {train_dist.values}")
    print(f"  Val class distribution: {val_dist.values}")
    print(f"  Test class distribution: {test_dist.values}")
    print(f"  Majority baseline (test): {test_dist.max():.4f}")
    
    if tune:
        best_params, val_auc = tune_hyperparameters_rf(X_train, y_train, X_val, y_val, param_grid, seed)
    else:
        best_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': seed,
            'n_jobs': -1
        }
    
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_prob)
    
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob)
    
    return val_acc, val_auc, test_acc, test_auc, best_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random","chrono"], default="chrono",
                    help="random (replicates earlier results) or chrono (leak-free)")
    ap.add_argument("--tune", action="store_true", default=True,
                    help="Perform hyperparameter tuning on validation set")
    ap.add_argument("--no-tune", dest="tune", action="store_false",
                    help="Skip hyperparameter tuning (use defaults)")
    ap.add_argument("--quick-tune", action="store_true",
                    help="Use smaller parameter grid for faster tuning")
    ap.add_argument("--val-size", type=float, default=0.15,
                    help="Validation set size (default: 0.15)")
    ap.add_argument("--test-size", type=float, default=0.15,
                    help="Test set size (default: 0.15)")
    ap.add_argument("--model", choices=["xgboost", "lightgbm", "randomforest", "all"], default="all",
                    help="Which model to train: xgboost, lightgbm, randomforest, or all (default: all)")
    args = ap.parse_args()

    tech = pd.read_csv("data/processed/merge_T_only_h5.csv", parse_dates=["date"])
    
    # Load all sentiment datasets
    score = pd.read_csv("data/processed/sent_headlines_score.csv", parse_dates=["date"])
    triplet = pd.read_csv("data/processed/sent_headlines_triplet.csv", parse_dates=["date"])
    embed = pd.read_csv("data/processed/sent_headlines_embed_pca16.csv", parse_dates=["date"])
    
    for name, df_var in [("score", score), ("triplet", triplet), ("embed", embed)]:
        dupes = df_var.groupby(['ticker','date']).size().max()
        if dupes > 1:
            print(f"[warn] {name}: Found {dupes} rows per ticker-date, aggregating...")
            cols = [c for c in df_var.columns if c not in ['date', 'ticker']]
            aggregated = df_var.groupby(['ticker','date'], as_index=False)[cols].mean()
            if name == "score":
                score = aggregated
            elif name == "triplet":
                triplet = aggregated
            elif name == "embed":
                embed = aggregated

    # Load enhanced sentiment features (140+ features)
    sent_enhanced = pd.read_csv("data/processed/sentiment_features_enhanced.csv", parse_dates=["date"])
    sent_enhanced["ticker"] = sent_enhanced["ticker"].astype(str).str.upper()
    print(f"[info] Loaded enhanced sentiment features: {len(sent_enhanced.columns) - 2} features")
    
    print(f"[info] Original sent_enhanced shape: {sent_enhanced.shape}")
    dupes_before = sent_enhanced.groupby(['ticker','date']).size().max()
    if dupes_before > 1:
        print(f"[warn] Found up to {dupes_before} rows per ticker-date, aggregating by mean...")
        sent_cols = [c for c in sent_enhanced.columns if c not in ['date', 'ticker']]
        sent_enhanced = sent_enhanced.groupby(['ticker','date'], as_index=False)[sent_cols].mean()
        print(f"[info] After aggregation: {sent_enhanced.shape}")
    
    sentiment_dates = set(sent_enhanced['date'].unique())
    tech_dates_before = len(tech['date'].unique())
    tech = tech[tech['date'].isin(sentiment_dates)]
    tech_dates_after = len(tech['date'].unique())
    
    print(f"\n{'='*70}")
    print(f"FILTERING TO SENTIMENT COVERAGE PERIOD")
    print(f"{'='*70}")
    print(f"  Original tech data: {tech_dates_before} dates")
    print(f"  Sentiment coverage: {len(sentiment_dates)} dates")
    print(f"  Filtered tech data: {tech_dates_after} dates")
    print(f"  Date range: {tech['date'].min().date()} to {tech['date'].max().date()}")
    print(f"{'='*70}\n")
    
    if tech_dates_after < 30:
        print(f"[ERROR] Insufficient data after filtering to sentiment dates!")
        print(f"        You need to collect more historical sentiment data.")
        print(f"        Run: python scripts/fetch_headlines.py --start-date 2024-01-01")
        return
    
    # Also filter other sentiment datasets to match
    score = score[score['date'].isin(sentiment_dates)]
    triplet = triplet[triplet['date'].isin(sentiment_dates)]
    embed = embed[embed['date'].isin(sentiment_dates)]

    # Expanded technical features - automatically detect all available technical indicators
    base_features = ["ret_1d", "rsi"]
    
    ma_features = [c for c in tech.columns if c.startswith(('sma_', 'ema_', 'mom_', 'price_to_sma_', 'price_to_ema_'))]
    
    # Volatility features (exclude volume-related features that start with vol_)
    vol_features = [c for c in tech.columns if c.startswith(('atr_', 'bb_')) or (c.startswith('vol_') and not c.startswith('vol_sma') and not c == 'vol_ratio')]
    
    momentum_features = [c for c in tech.columns if c.startswith(('stoch_', 'willr_', 'roc_', 'cci_', 'macd'))]
    
    trend_features = [c for c in tech.columns if c.startswith('adx_')]
    
    volume_features = [c for c in tech.columns if c.startswith(('vol_sma_', 'obv', 'vwap_', 'price_to_vwap')) or c == 'vol_ratio']
    
    pattern_features = [c for c in tech.columns if c in ['high_low_range', 'close_position', 'upper_shadow', 'lower_shadow', 'body_size', 'body_size_ratio']]
    
    # Combine all technical features and remove duplicates
    tech_features = base_features + ma_features + vol_features + momentum_features + trend_features + volume_features + pattern_features
    
    # Remove duplicates while preserving order
    seen = set()
    tech_features = [x for x in tech_features if not (x in seen or seen.add(x))]
    
    # Filter to only include features that actually exist in the dataframe
    tech_features = [f for f in tech_features if f in tech.columns]
    
    print(f"\n{'='*70}")
    print(f"Using {len(tech_features)} technical features:")
    print(f"  - Basic: {len(base_features)}")
    print(f"  - Moving Averages: {len(ma_features)}")
    print(f"  - Volatility: {len(vol_features)}")
    print(f"  - Momentum: {len(momentum_features)}")
    print(f"  - Trend: {len(trend_features)}")
    print(f"  - Volume: {len(volume_features)}")
    print(f"  - Price Patterns: {len(pattern_features)}")
    print(f"{'='*70}\n")

    print(f"\n{'='*70}")
    print(f"=== Ensemble Model Evaluation ===")
    print(f"Model(s): {args.model}, Split: {args.split}, Tune: {args.tune}")
    print(f"{'='*70}")

    results = []
    
    # IMPORTANT: Select based on VALIDATION AUC to prevent data leakage
    best_models = {
        'xgboost': {'model': None, 'val_auc': 0, 'test_auc': 0, 'config': None, 'features': None, 'data': None},
        'lightgbm': {'model': None, 'val_auc': 0, 'test_auc': 0, 'config': None, 'features': None, 'data': None},
        'randomforest': {'model': None, 'val_auc': 0, 'test_auc': 0, 'config': None, 'features': None, 'data': None}
    }
    
    y_train_temp = tech.dropna(subset=tech_features + ['y'])['y']
    class_ratio = (y_train_temp == 0).sum() / (y_train_temp == 1).sum()
    print(f"Class ratio (neg/pos): {class_ratio:.3f}")
    
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
        param_grid_rf = {
            'n_estimators': [100, 300],
            'max_depth': [5, 10],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt'],
            'class_weight': ['balanced', None],
        }
        print(f"Quick tuning: ~16 combinations per model")
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
        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', None],
        }
        print(f"Full grid search: ~96 combinations per model")
    
    # === Define Enhanced Sentiment Features ===
    enhanced_sent_cols = [c for c in sent_enhanced.columns if c not in ['date', 'ticker']]
    
    enhanced_ma_features = [c for c in enhanced_sent_cols if '_sma_' in c or '_ema_' in c or '_ma_' in c or '_dist_' in c]
    print(f"[info] Enhanced sentiment MA features: {len(enhanced_ma_features)}")
    
    enhanced_mom_features = [c for c in enhanced_sent_cols if '_mom_' in c or '_roc_' in c or '_trend_' in c]
    print(f"[info] Enhanced sentiment momentum features: {len(enhanced_mom_features)}")
    
    enhanced_vol_features = [c for c in enhanced_sent_cols if '_std_' in c or '_cv_' in c]
    print(f"[info] Enhanced sentiment volatility features: {len(enhanced_vol_features)}")
    
    enhanced_range_features = [c for c in enhanced_sent_cols if '_min_' in c or '_max_' in c or '_position_' in c]
    print(f"[info] Enhanced sentiment range features: {len(enhanced_range_features)}")
    
    enhanced_cumulative_features = [c for c in enhanced_sent_cols if '_cumsum' in c or '_ewm_cumsum' in c]
    print(f"[info] Enhanced sentiment cumulative features: {len(enhanced_cumulative_features)}")
    
    cross_sentiment_features = [c for c in enhanced_sent_cols if c.startswith('sent_') and c not in ['sent_score']]
    print(f"[info] Cross-sentiment features: {len(cross_sentiment_features)}")
    
    news_features = [c for c in enhanced_sent_cols if 'news_' in c]
    print(f"[info] News count features: {len(news_features)}")
    
    df_merged = tech.merge(sent_enhanced, on=["ticker","date"], how="left")
    
    def dedupe_features(feature_list):
        seen = set()
        return [x for x in feature_list if not (x in seen or seen.add(x))]
    
    # Create interaction features: MA * news_count
    df_for_interactions = tech.merge(sent_enhanced, on=["ticker","date"], how="inner")
    ma_news_interactions = []
    if 'news_count' in df_for_interactions.columns:
        for ma_col in enhanced_ma_features:
            interaction_name = f"{ma_col}*news_count"
            df_for_interactions[interaction_name] = df_for_interactions[ma_col] * df_for_interactions['news_count']
            ma_news_interactions.append(interaction_name)
        print(f"[info] Created {len(ma_news_interactions)} MA*news_count interaction features")
    else:
        print(f"[warn] 'news_count' column not found, skipping interaction features")
    
    model_configs = [
        ("Technical only", tech, tech_features),
        
        ("Technical + Score", 
         tech.merge(score, on=["ticker","date"], how="inner"), 
         dedupe_features(tech_features+["sent_score"])),
        
        ("Technical + Triplet", 
         tech.merge(triplet, on=["ticker","date"], how="inner"), 
         dedupe_features(tech_features+["p_pos","p_neu","p_neg"])),
        
        ("Technical + Enhanced Sentiment (ALL)", 
         tech.merge(sent_enhanced, on=["ticker","date"], how="inner"),
         dedupe_features(tech_features+enhanced_sent_cols)),
    ]
    
    print(f"\n[info] Model configurations: {len(model_configs)}")
    print(f"       1. Technical only (baseline)")
    print(f"       2. Technical + Score (simple sentiment)")
    print(f"       3. Technical + Triplet (FinBERT probabilities)")
    print(f"       4. Technical + Enhanced Sentiment ALL ({len(enhanced_sent_cols)} features)")
    
    # Run models based on selection
    for i, (model_name, df_model, features) in enumerate(model_configs, 1):
        print(f"\n{'='*70}")
        print(f"Model {i}: {model_name}")
        print(f"{'='*70}")
        
        if args.model in ["xgboost", "all"]:
            print(f"\n--- XGBoost ---")
            return_vals = evaluate_model_with_tuning(
                df_model, features, split=args.split, 
                val_size=args.val_size, test_size=args.test_size,
                param_grid=param_grid_xgb, tune=args.tune, return_model=True
            )
            val_acc, val_auc, test_acc, test_auc, params, model = return_vals
            results.append([f"XGB: {model_name}", val_acc, val_auc, test_acc, test_auc])
            
            # Track best based on VALIDATION AUC to prevent data leakage
            if val_auc > best_models['xgboost']['val_auc']:
                best_models['xgboost'] = {
                    'model': model,
                    'val_auc': val_auc,
                    'test_auc': test_auc,
                    'config': model_name,
                    'features': features,
                    'data': df_model
                }
                print(f"  [New best XGBoost based on Val AUC: {val_auc:.4f}]")
        
        if args.model in ["lightgbm", "all"]:
            print(f"\n--- LightGBM ---")
            val_acc, val_auc, test_acc, test_auc, params = evaluate_model_with_tuning_lgb(
                df_model, features, split=args.split,
                val_size=args.val_size, test_size=args.test_size,
                param_grid=param_grid_lgb, tune=args.tune
            )
            results.append([f"LGB: {model_name}", val_acc, val_auc, test_acc, test_auc])
        
        if args.model in ["randomforest", "all"]:
            print(f"\n--- Random Forest ---")
            val_acc, val_auc, test_acc, test_auc, params = evaluate_model_with_tuning_rf(
                df_model, features, split=args.split,
                val_size=args.val_size, test_size=args.test_size,
                param_grid=param_grid_rf, tune=args.tune
            )
            results.append([f"RF: {model_name}", val_acc, val_auc, test_acc, test_auc])
    
    if best_models['xgboost']['model'] is not None:
        best_model_info = best_models['xgboost']
        print(f"\n{'='*70}")
        print(f"Best model (by Val AUC): {best_model_info['config']}")
        print(f"  Val AUC: {best_model_info['val_auc']:.4f}, Test AUC: {best_model_info['test_auc']:.4f}")
        print(f"{'='*70}")
        
        feature_importance = best_model_info['model'].feature_importances_
        feature_names = best_model_info['features']
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        Path("reports").mkdir(parents=True, exist_ok=True)
        importance_df.to_csv("reports/best_model_feature_importance.csv", index=False)
        print(f"[info] Saved feature importance to reports/best_model_feature_importance.csv")
        
        # Save best model metadata
        best_model_meta = pd.DataFrame([{
            'config': best_model_info['config'],
            'val_auc': best_model_info['val_auc'],
            'test_auc': best_model_info['test_auc'],
            'n_features': len(feature_names),
            'features': ','.join(feature_names)
        }])
        best_model_meta.to_csv("reports/best_model_metadata.csv", index=False)
        print(f"[info] Saved best model metadata to reports/best_model_metadata.csv")
    
    results_df = pd.DataFrame(results, columns=["Model", "Val Accuracy", "Val AUC", "Test Accuracy", "Test AUC"])
    print(f"\n{'='*70}")
    print("=== Final Results Summary ===")
    print(f"{'='*70}")
    print(results_df.round(4).to_string(index=False))
    print(f"{'='*70}\n")
    
    Path("reports").mkdir(parents=True, exist_ok=True)
    results_df.to_csv("reports/model_comparison_results.csv", index=False)
    print(f"[info] Saved results to reports/model_comparison_results.csv")
    
    config_info = []
    for i, (model_name, df_model, features) in enumerate(model_configs):
        config_info.append({
            'config_id': i,
            'config_name': model_name,
            'n_features': len(features),
            'features': ','.join(features)
        })
    config_df = pd.DataFrame(config_info)
    config_df.to_csv("reports/model_configurations.csv", index=False)
    print(f"[info] Saved configurations to reports/model_configurations.csv")
    print(f"\n[info] To analyze best models, run: uv run scripts/analyze_models.py\n")

if __name__ == "__main__":
    main()

