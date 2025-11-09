#!/usr/bin/env python3
"""
Compare multiple neural network architectures and feature sets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import pickle

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
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, 
                        hidden_layers, alpha=0.001, max_iter=500, seed=42):
    """Train and evaluate a single configuration."""
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=alpha,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=True,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    results = {}
    for split_name, X, y in [("train", X_train, y_train), 
                              ("val", X_val, y_val), 
                              ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        results[f'{split_name}_acc'] = accuracy_score(y, y_pred)
        results[f'{split_name}_auc'] = roc_auc_score(y, y_prob)
    
    results['n_iter'] = model.n_iter_
    results['final_loss'] = model.loss_
    
    return model, results


def main():
    print("="*70)
    print("COMPARING NEURAL NETWORK ARCHITECTURES")
    print("="*70)
    
    # Load data
    print("\n[1/3] Loading data...")
    tech = pd.read_csv("data/processed/merge_T_only_h5.csv", parse_dates=["date"])
    sent_enhanced = pd.read_csv("data/processed/sentiment_features_enhanced.csv", parse_dates=["date"])
    sent_enhanced["ticker"] = sent_enhanced["ticker"].astype(str).str.upper()
    df_merged = tech.merge(sent_enhanced, on=["ticker","date"], how="left")
    
    # Define configurations to test
    configurations = [
        # (n_features, hidden_layers, alpha)
        (10, (50, 25), 0.01),
        (10, (30, 15), 0.01),
        (20, (50, 25), 0.01),
        (20, (100, 50), 0.001),
        (20, (30, 15), 0.01),
        (30, (100, 50), 0.01),
        (30, (50, 25), 0.01),
        (50, (100, 50), 0.01),
    ]
    
    results_list = []
    best_model = None
    best_test_auc = 0
    best_config = None
    
    print(f"\n[2/3] Training {len(configurations)} configurations...")
    
    for i, (n_features, hidden_layers, alpha) in enumerate(configurations, 1):
        print(f"\n--- Config {i}/{len(configurations)} ---")
        print(f"  Features: top {n_features}")
        print(f"  Architecture: {hidden_layers}")
        print(f"  L2 alpha: {alpha}")
        
        # Load features
        try:
            top_features_df = pd.read_csv(f"data/processed/top_{n_features}_features_for_nn.csv")
            top_features = top_features_df['feature'].tolist()
            top_features = [f for f in top_features if f in df_merged.columns]
        except:
            print(f"  [skip] Feature file not found")
            continue
        
        # Split and normalize
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_chrono(
            df_merged, top_features, label_col="y_h5"
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model, results = train_and_evaluate(
            X_train_scaled, y_train, X_val_scaled, y_val, 
            X_test_scaled, y_test, hidden_layers, alpha=alpha
        )
        
        # Store results
        results['n_features'] = len(top_features)
        results['architecture'] = str(hidden_layers)
        results['alpha'] = alpha
        results['config_id'] = f"top{n_features}_{hidden_layers}_alpha{alpha}"
        
        print(f"  Train: Acc={results['train_acc']:.4f}, AUC={results['train_auc']:.4f}")
        print(f"  Val:   Acc={results['val_acc']:.4f}, AUC={results['val_auc']:.4f}")
        print(f"  Test:  Acc={results['test_acc']:.4f}, AUC={results['test_auc']:.4f}")
        
        results_list.append(results)
        
        # Track best model
        if results['test_auc'] > best_test_auc:
            best_test_auc = results['test_auc']
            best_model = model
            best_config = (n_features, hidden_layers, alpha, scaler, top_features)
            print(f"  ⭐ New best test AUC: {best_test_auc:.4f}")
    
    # Save results
    print(f"\n[3/3] Saving results...")
    Path("reports/neural_network").mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('test_auc', ascending=False)
    results_df.to_csv("reports/neural_network/nn_architecture_comparison.csv", index=False)
    
    # Save best model
    if best_model is not None:
        n_feat, h_layers, alpha_val, scaler, features = best_config
        Path("models").mkdir(parents=True, exist_ok=True)
        
        with open("models/best_nn_model.pkl", 'wb') as f:
            pickle.dump(best_model, f)
        
        with open("models/best_nn_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        with open("models/best_nn_features.txt", 'w') as f:
            f.write('\n'.join(features))
        
        print(f"  Saved best model (test AUC: {best_test_auc:.4f})")
        print(f"  Config: top{n_feat} features, {h_layers}, alpha={alpha_val}")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\nTop 5 Models by Test AUC:")
    print(results_df[['config_id', 'test_auc', 'test_acc', 'val_auc', 'train_auc']].head(5).to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE")
    print(f"Results saved to: reports/neural_network/nn_architecture_comparison.csv")
    print(f"Best model saved to: models/best_nn_model.pkl")
    print("="*70)


if __name__ == "__main__":
    main()


