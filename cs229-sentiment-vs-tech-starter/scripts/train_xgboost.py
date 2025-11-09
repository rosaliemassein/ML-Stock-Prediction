#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
from itertools import product

def split_data(df, feature_cols, label_col="y", split="random", val_size=0.1, test_size=0.1, seed=42):
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
        X = df[feature_cols].values
        y = df[label_col].values
        
        # First split: separate test set
        n = len(X)
        np.random.seed(seed)
        indices = np.random.permutation(n)
        
        test_n = int(test_size * n)
        val_n = int(val_size * n)
        
        test_indices = indices[:test_n]
        val_indices = indices[test_n:test_n + val_n]
        train_indices = indices[test_n + val_n:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]
    
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
        # Default parameter grid
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
    
    # Grid search
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
                                val_size=0.2, test_size=0.2, seed=42, 
                                param_grid=None, tune=True):
    """
    Evaluate XGBoost model with optional hyperparameter tuning.
    
    Returns:
        val_acc, val_auc, test_acc, test_auc, best_params
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, feature_cols, label_col, split, val_size, test_size, seed
    )
    
    print(f"  Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Print class distributions
    train_dist = y_train.value_counts(normalize=True).sort_index()
    val_dist = y_val.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    print(f"  Train class distribution: {train_dist.values}")
    print(f"  Val class distribution: {val_dist.values}")
    print(f"  Test class distribution: {test_dist.values}")
    print(f"  Majority baseline (test): {test_dist.max():.4f}")
    
    if tune:
        # Tune hyperparameters on validation set
        best_params, val_auc = tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid, seed)
    else:
        # Use default parameters
        best_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': seed,
            'objective': 'binary:logistic',
            'eval_metric': 'auc'
        }
    
    # Train final model with best parameters
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_prob)
    
    # Evaluate on test set
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
    args = ap.parse_args()

    tech = pd.read_csv("data/processed/merge_T_only_h5.csv", parse_dates=["date"])
    score = pd.read_csv("data/processed/sent_headlines_score.csv", parse_dates=["date"])
    triplet = pd.read_csv("data/processed/sent_headlines_triplet.csv", parse_dates=["date"])
    embed = pd.read_csv("data/processed/sent_headlines_embed_pca16.csv", parse_dates=["date"])

    tech_features = ["ret_1d","sma_5","mom_5","sma_10","mom_10","sma_20","mom_20","vol_20","rsi"]

    print(f"\n{'='*70}")
    print(f"=== XGBoost Model Evaluation (split={args.split}, tune={args.tune}) ===")
    print(f"{'='*70}")

    results = []
    
    # Calculate class weight for handling slight imbalance
    y_train_temp = tech.dropna(subset=tech_features + ['y'])['y']
    class_ratio = (y_train_temp == 0).sum() / (y_train_temp == 1).sum()
    print(f"Class ratio (neg/pos): {class_ratio:.3f}")
    
    # Define parameter grid based on quick_tune flag
    if args.quick_tune:
        # Smaller grid for quick experiments (~16 combinations)
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.05],
            'n_estimators': [300],
            'min_child_weight': [5, 10],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'gamma': [0, 0.1],
            'scale_pos_weight': [class_ratio],
        }
        print(f"Quick tuning: {2*2*1*2*1*1*2*1} = 16 combinations per model")
    else:
        # Full grid optimized for weak signal detection (~96 combinations)
        param_grid = {
            'max_depth': [2, 3, 5],  # Shallower trees to prevent overfitting
            'learning_rate': [0.01, 0.05],  # Lower learning rates
            'n_estimators': [300, 500],  # More trees for weak patterns
            'min_child_weight': [5, 10],  # Higher to prevent overfitting noise
            'subsample': [0.7, 0.8],  # Subsampling
            'colsample_bytree': [0.8],  # Feature sampling
            'gamma': [0, 0.1],  # Regularization
            'scale_pos_weight': [class_ratio],  # Handle class imbalance
        }
        print(f"Full grid search: {3*2*2*2*2*1*2*1} = 96 combinations per model")
    
    # 1. Technical only
    print(f"\n{'='*70}")
    print("Model 1: Technical Features Only")
    print(f"{'='*70}")
    val_acc, val_auc, test_acc, test_auc, params = evaluate_model_with_tuning(
        tech, tech_features, split=args.split, 
        val_size=args.val_size, test_size=args.test_size,
        param_grid=param_grid, tune=args.tune
    )
    results.append(["Technical only", val_acc, val_auc, test_acc, test_auc])
    
    # 2. Technical + Score
    print(f"\n{'='*70}")
    print("Model 2: Technical + Sentiment Score")
    print(f"{'='*70}")
    merged_score = tech.merge(score, on=["ticker","date"], how="left").dropna(subset=["sent_score"])
    val_acc, val_auc, test_acc, test_auc, params = evaluate_model_with_tuning(
        merged_score, tech_features+["sent_score"], split=args.split,
        val_size=args.val_size, test_size=args.test_size,
        param_grid=param_grid, tune=args.tune
    )
    results.append(["Technical + Score", val_acc, val_auc, test_acc, test_auc])
    
    # 3. Technical + Triplet
    print(f"\n{'='*70}")
    print("Model 3: Technical + Sentiment Triplet")
    print(f"{'='*70}")
    merged_triplet = tech.merge(triplet, on=["ticker","date"], how="left").dropna(subset=["p_pos","p_neu","p_neg"])
    val_acc, val_auc, test_acc, test_auc, params = evaluate_model_with_tuning(
        merged_triplet, tech_features+["p_pos","p_neu","p_neg"], split=args.split,
        val_size=args.val_size, test_size=args.test_size,
        param_grid=param_grid, tune=args.tune
    )
    results.append(["Technical + Triplet", val_acc, val_auc, test_acc, test_auc])
    
    # 4. Technical + PCA16
    print(f"\n{'='*70}")
    print("Model 4: Technical + PCA16 Embeddings")
    print(f"{'='*70}")
    merged_embed = tech.merge(embed, on=["ticker","date"], how="left")
    emb_cols = [c for c in merged_embed.columns if c.startswith("emb_")]
    merged_embed = merged_embed.dropna(subset=emb_cols)
    val_acc, val_auc, test_acc, test_auc, params = evaluate_model_with_tuning(
        merged_embed, tech_features+emb_cols, split=args.split,
        val_size=args.val_size, test_size=args.test_size,
        param_grid=param_grid, tune=args.tune
    )
    results.append(["Technical + PCA16", val_acc, val_auc, test_acc, test_auc])
    
    # Print final results
    results_df = pd.DataFrame(results, columns=["Model", "Val Accuracy", "Val AUC", "Test Accuracy", "Test AUC"])
    print(f"\n{'='*70}")
    print("=== Final Results Summary ===")
    print(f"{'='*70}")
    print(results_df.round(4).to_string(index=False))
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

