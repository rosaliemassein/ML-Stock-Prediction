#!/usr/bin/env python3
"""
Analyze and compare the best models:
- Tree models (XGBoost, LightGBM, Random Forest), trained here
- LSTM sequence model (trained separately, loaded from saved results)

Generates:
1. Table of accuracy and AUC metrics (trees + LSTM, if available)
2. Feature importance plots for each tree model
3. ROC curves comparing all models (trees + LSTM, if available)
4. Training history/loss curves for XGBoost and LightGBM
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from pathlib import Path
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, 
                             confusion_matrix, classification_report)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from itertools import product


def split_data_chrono(df, feature_cols, label_col="y_h5", val_size=0.15, test_size=0.15):
    """Chronological split for time series data."""
    df = df.dropna(subset=feature_cols + [label_col]).copy()
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
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train, val, test


def train_xgboost(X_train, y_train, X_val, y_val, seed=42):
    """Train XGBoost with best parameters (from prior tuning)."""
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    best_params = {
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'scale_pos_weight': class_ratio,
        'random_state': seed,
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }
    
    model = xgb.XGBClassifier(**best_params)
    
    # Track training history
    eval_result = {}
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    eval_result = model.evals_result()
    
    return model, eval_result


def train_lightgbm(X_train, y_train, X_val, y_val, seed=42):
    """Train LightGBM with best parameters (from prior tuning)."""
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    best_params = {
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_split_gain': 0,
        'scale_pos_weight': class_ratio,
        'random_state': seed,
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**best_params)
    
    # Track training history
    callbacks = [lgb.record_evaluation(eval_result := {})]
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        callbacks=callbacks
    )
    
    return model, eval_result


def train_random_forest(X_train, y_train, seed=42):
    """Train Random Forest with best parameters (from prior tuning)."""
    best_params = {
        'n_estimators': 300,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': seed,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    
    return model, None  # RF doesn't have training history


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Comprehensive model evaluation for tree models."""
    results = {}
    
    for split_name, X, y in [("train", X_train, y_train), 
                              ("val",   X_val,   y_val), 
                              ("test",  X_test,  y_test)]:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        
        results[split_name] = {
            'accuracy': acc,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_prob,
            'true_labels': y.values if hasattr(y, 'values') else y
        }
    
    return results


def plot_roc_curves(models_results, output_path):
    """Plot ROC curves for all tree models PLUS LSTM if available."""
    import pandas as pd

    plt.figure(figsize=(10, 8))

    colors = {
        "XGBoost": "#1f77b4",
        "LightGBM": "#ff7f0e",
        "RandomForest": "#2ca02c",
        "LSTM": "#d62728",
    }
    line_styles = {
        "XGBoost": "-",
        "LightGBM": "--",
        "RandomForest": "-.",
        "LSTM": "--"
    }

    # 1) Tree models
    for model_name, results in models_results.items():
        y_true = results["test"]["true_labels"]
        y_prob = results["test"]["probabilities"]
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = results["test"]["auc"]

        plt.plot(
            fpr, tpr,
            color=colors.get(model_name, "#444444"),
            linestyle=line_styles.get(model_name, "-"),
            linewidth=2,
            label=f"{model_name} (AUC = {auc_score:.4f})"
        )

    # 2) LSTM (loaded from saved predictions)
    lstm_pred_file = Path("reports/lstm_test_predictions.csv")
    if lstm_pred_file.exists():
        lstm_df = pd.read_csv(lstm_pred_file)
        y_true_lstm = lstm_df["y_true"].values
        y_prob_lstm = lstm_df["y_prob"].values

        fpr_lstm, tpr_lstm, _ = roc_curve(y_true_lstm, y_prob_lstm)
        auc_lstm = roc_auc_score(y_true_lstm, y_prob_lstm)

        plt.plot(
            fpr_lstm, tpr_lstm,
            color=colors["LSTM"],
            linestyle=line_styles["LSTM"],
            linewidth=2,
            label=f"LSTM (AUC = {auc_lstm:.4f})"
        )
        print(f"[info] Added LSTM ROC curve (AUC={auc_lstm:.4f})")
    else:
        print("[warn] LSTM predictions file not found → skipping LSTM ROC")

    # 3) Random baseline
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5000)')

    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Model Comparison (Trees + LSTM)', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[info] Saved ROC curves to {output_path}")
    plt.close()


def plot_training_history(train_histories, output_path):
    """Plot training history for XGBoost and LightGBM."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # XGBoost
    if 'XGBoost' in train_histories and train_histories['XGBoost'] is not None:
        eval_result = train_histories['XGBoost']
        epochs = len(eval_result['validation_0']['auc'])
        
        ax1.plot(range(epochs), eval_result['validation_0']['auc'], 
                label='Train', linewidth=2, color='#1f77b4')
        ax1.plot(range(epochs), eval_result['validation_1']['auc'], 
                label='Validation', linewidth=2, color='#ff7f0e')
        ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('AUC', fontsize=12, fontweight='bold')
        ax1.set_title('XGBoost Training History', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # LightGBM
    if 'LightGBM' in train_histories and train_histories['LightGBM'] is not None:
        eval_result = train_histories['LightGBM']
        epochs = len(eval_result['training']['auc'])
        
        ax2.plot(range(epochs), eval_result['training']['auc'], 
                label='Train', linewidth=2, color='#1f77b4')
        ax2.plot(range(epochs), eval_result['valid_1']['auc'], 
                label='Validation', linewidth=2, color='#ff7f0e')
        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('AUC', fontsize=12, fontweight='bold')
        ax2.set_title('LightGBM Training History', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[info] Saved training history to {output_path}")
    plt.close()


def plot_feature_importance(model, feature_names, model_name, output_path, top_n=30):
    """Plot feature importance for a single tree model."""
    importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
    colors = cm.viridis(np.linspace(0, 1, top_n))
    
    ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importance - {model_name}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    for i, v in enumerate(top_features['importance']):
        ax.text(v, i, f' {v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[info] Saved feature importance to {output_path}")
    plt.close()
    
    return importance_df


def main():
    ap = argparse.ArgumentParser(description='Analyze best models and generate visualizations')
    ap.add_argument("--results-file", default="reports/model_comparison_results.csv",
                    help="Results file from train_xgboost.py (tree model comparison)")
    ap.add_argument("--output-dir", default="reports/model_analysis",
                    help="Directory to save analysis outputs")
    args = ap.parse_args()
    
    print("="*70)
    print("ANALYZING BEST MODELS (Trees + LSTM)")
    print("="*70)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load results from tree comparison (optional, just for logging)
    print(f"\n[1/7] Loading tree comparison results from {args.results_file}...")
    try:
        results_df = pd.read_csv(args.results_file)
        best_xgb = results_df[results_df['Model'].str.contains('XGB')].nlargest(1, 'Test AUC').iloc[0]
        best_lgb = results_df[results_df['Model'].str.contains('LGB')].nlargest(1, 'Test AUC').iloc[0]
        best_rf  = results_df[results_df['Model'].str.contains('RF')].nlargest(1, 'Test AUC').iloc[0]
    
        print(f"  Best XGBoost: {best_xgb['Model']} (Test AUC: {best_xgb['Test AUC']:.4f})")
        print(f"  Best LightGBM: {best_lgb['Model']} (Test AUC: {best_lgb['Test AUC']:.4f})")
        print(f"  Best Random Forest: {best_rf['Model']} (Test AUC: {best_rf['Test AUC']:.4f})")
    except FileNotFoundError:
        print(f"[warn] results_file not found. Continuing with fresh training only.")
    except Exception as e:
        print(f"[warn] Could not parse results_file: {e}")
    
    # Load data
    print(f"\n[2/7] Loading data...")
    tech = pd.read_csv("data/processed/merge_T_only_h5.csv", parse_dates=["date"])
    tech["ticker"] = tech["ticker"].astype(str).str.upper()

    # ---- NEW: rebuild ret_fwd_5d and y_h5 if missing ----
    if "y_h5" not in tech.columns:
        tech = tech.sort_values(["ticker", "date"])
        tech["close_fwd_5d"] = tech.groupby("ticker")["close"].shift(-5)
        tech["ret_fwd_5d"] = (tech["close_fwd_5d"] - tech["close"]) / tech["close"]
        tech["y_h5"] = (tech["ret_fwd_5d"] > 0).astype(int)
        tech = tech.drop(columns=["close_fwd_5d"])
        # (optional but nice: persist for future scripts)
        tech.to_csv("data/processed/merge_T_only_h5.csv", index=False)
        print("[info] Recomputed ret_fwd_5d and y_h5 and rewrote merge_T_only_h5.csv")

    sent_enhanced = pd.read_csv("data/processed/sentiment_features_enhanced.csv", parse_dates=["date"])
    sent_enhanced["ticker"] = sent_enhanced["ticker"].astype(str).str.upper()
    df_merged = tech.merge(sent_enhanced, on=["ticker","date"], how="left")
    
    # Get features (technical + sentiment moving averages)
    tech_features = [c for c in tech.columns 
                     if c not in ['date', 'ticker', 'open', 'high', 'low', 'close',
                                  'volume', 'y', 'ret_fwd_5d', 'y_h5']]
    enhanced_sent_cols = [c for c in sent_enhanced.columns if c not in ['date', 'ticker']]
    enhanced_ma_features = [c for c in enhanced_sent_cols 
                            if '_sma_' in c or '_ema_' in c or '_ma_' in c]
    all_features = tech_features + enhanced_ma_features
    all_features = [f for f in all_features if f in df_merged.columns]
    
    print(f"  Using {len(all_features)} features ({len(tech_features)} technical + {len(enhanced_ma_features)} sentiment MA)")
    
    # Split data
    print(f"\n[3/7] Splitting data (chronological)...")
    X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = split_data_chrono(
        df_merged, all_features, label_col="y_h5"
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train tree models
    print(f"\n[4/7] Training tree models with best hyperparameters...")
    models = {}
    train_histories = {}
    
    print("  Training XGBoost...")
    models['XGBoost'], train_histories['XGBoost'] = train_xgboost(X_train, y_train, X_val, y_val)
    
    print("  Training LightGBM...")
    models['LightGBM'], train_histories['LightGBM'] = train_lightgbm(X_train, y_train, X_val, y_val)
    
    print("  Training RandomForest...")
    models['RandomForest'], train_histories['RandomForest'] = train_random_forest(X_train, y_train)
    
    # Evaluate tree models
    print(f"\n[5/7] Evaluating tree models...")
    models_results = {}
    for model_name, model in models.items():
        results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
        models_results[model_name] = results
        print(f"  {model_name}: Test AUC = {results['test']['auc']:.4f}, Acc = {results['test']['accuracy']:.4f}")
    
    # Save tree models
    Path("models").mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        model_path = f"models/best_{model_name.lower()}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved {model_name} to {model_path}")
    
    # Generate outputs
    print(f"\n[6/7] Generating analysis outputs...")
    
    # 1. Metrics table (trees + LSTM, if available)
    metrics_data = []
    for model_name, results in models_results.items():
        metrics_data.append({
            'Model': model_name,
            'Train Accuracy': results['train']['accuracy'],
            'Train AUC': results['train']['auc'],
            'Val Accuracy': results['val']['accuracy'],
            'Val AUC': results['val']['auc'],
            'Test Accuracy': results['test']['accuracy'],
            'Test AUC': results['test']['auc']
        })
    
    metrics_df = pd.DataFrame(metrics_data)

    # Try to append LSTM metrics if file exists
    lstm_metrics_file = Path("reports/lstm_results.csv")
    if lstm_metrics_file.exists():
        try:
            lstm_df = pd.read_csv(lstm_metrics_file)
            metrics_df = pd.concat([metrics_df, lstm_df], ignore_index=True)
            print(f"  [info] Appended LSTM metrics from {lstm_metrics_file}")
        except Exception as e:
            print(f"  [warn] Could not load LSTM metrics: {e}")
    else:
        print("  [warn] LSTM metrics file not found → LSTM not in metrics_table")

    metrics_df.to_csv(f"{args.output_dir}/metrics_table.csv", index=False)
    print(f"  [1/4] Saved metrics table (trees + LSTM if available)")
    
    # 2. ROC curves (trees + optional LSTM)
    plot_roc_curves(models_results, f"{args.output_dir}/roc_curves.png")
    print(f"  [2/4] Saved ROC curves")
    
    # 3. Training history (trees only)
    plot_training_history(train_histories, f"{args.output_dir}/training_history.png")
    print(f"  [3/4] Saved training history")
    
    # 4. Feature importance for each tree model
    for model_name, model in models.items():
        importance_df = plot_feature_importance(
            model, all_features, model_name,
            f"{args.output_dir}/feature_importance_{model_name.lower()}.png"
        )
        importance_df.to_csv(f"{args.output_dir}/feature_importance_{model_name.lower()}.csv", index=False)
    print(f"  [4/4] Saved feature importance plots")
    
    # Print summary
    print(f"\n[7/7] Generating summary report...")
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY (Trees + LSTM)")
    print("="*70)
    print("\n📊 Metrics Table:")
    print(metrics_df.round(4).to_string(index=False))
    
    print(f"\n\n📁 Generated Files:")
    print(f"  Models (trees):")
    print(f"    - models/best_xgboost.pkl")
    print(f"    - models/best_lightgbm.pkl")
    print(f"    - models/best_randomforest.pkl")
    print(f"\n  Analysis Outputs (in {args.output_dir}/):")
    print(f"    - metrics_table.csv          (trees + LSTM row if available)")
    print(f"    - roc_curves.png             (trees + LSTM curve if available)")
    print(f"    - training_history.png       (trees only)")
    print(f"    - feature_importance_xgboost.png + .csv")
    print(f"    - feature_importance_lightgbm.png + .csv")
    print(f"    - feature_importance_randomforest.png + .csv")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
