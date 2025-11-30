#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class StockSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 2)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_sequences(df, feature_cols, label_col="y", sequence_length=5):
    """
    Create sequences for GRU training.
    Each sequence contains the last N days of features for each ticker.
    
    Returns:
        sequences: numpy array of shape (n_samples, sequence_length, n_features)
        labels: numpy array of shape (n_samples,)
        dates: numpy array of dates corresponding to each sequence
    """
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    sequences = []
    labels = []
    dates = []
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].reset_index(drop=True)
        
        for i in range(sequence_length, len(ticker_df)):
            sequence = ticker_df.iloc[i-sequence_length:i][feature_cols].values
            
            if not np.isnan(sequence).any() and not np.isnan(ticker_df.iloc[i][label_col]):
                sequences.append(sequence)
                labels.append(ticker_df.iloc[i][label_col])
                dates.append(ticker_df.iloc[i]['date'])
    
    return np.array(sequences), np.array(labels), np.array(dates)


def split_data_gru(df, feature_cols, label_col="y", split="random", 
                   val_size=0.2, test_size=0.1, sequence_length=5, seed=42):
    """
    Split data into train/validation/test sets for GRU.
    """
    df = df.copy()
    
    sequences, labels, dates = create_sequences(
        df, feature_cols, label_col, sequence_length
    )
    
    print(f"  Created {len(sequences)} sequences with length {sequence_length}")
    
    if split == "chrono":
        sorted_idx = np.argsort(dates)
        sequences = sequences[sorted_idx]
        labels = labels[sorted_idx]
        dates = dates[sorted_idx]
        
        unique_dates = np.sort(np.unique(dates))
        train_cut = int((1 - val_size - test_size) * len(unique_dates))
        val_cut = int((1 - test_size) * len(unique_dates))
        
        train_date = unique_dates[train_cut]
        val_date = unique_dates[val_cut]
        
        train_mask = dates < train_date
        val_mask = (dates >= train_date) & (dates < val_date)
        test_mask = dates >= val_date
        
        X_train, y_train = sequences[train_mask], labels[train_mask]
        X_val, y_val = sequences[val_mask], labels[val_mask]
        X_test, y_test = sequences[test_mask], labels[test_mask]
    else:
        n = len(sequences)
        np.random.seed(seed)
        indices = np.random.permutation(n)
        
        test_n = int(test_size * n)
        val_n = int(val_size * n)
        
        test_indices = indices[:test_n]
        val_indices = indices[test_n:test_n + val_n]
        train_indices = indices[test_n + val_n:]
        
        X_train, y_train = sequences[train_indices], labels[train_indices]
        X_val, y_val = sequences[val_indices], labels[val_indices]
        X_test, y_test = sequences[test_indices], labels[test_indices]
    
    scaler = StandardScaler()
    n_samples, seq_len, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(n_samples, seq_len, n_features)
    
    n_samples_val = X_val.shape[0]
    X_val_reshaped = X_val.reshape(-1, n_features)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(n_samples_val, seq_len, n_features)
    
    n_samples_test = X_test.shape[0]
    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(n_samples_test, seq_len, n_features)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def train_gru_model(X_train, y_train, X_val, y_val, 
                   hidden_size=64, num_layers=2, dropout=0.3,
                   learning_rate=0.001, batch_size=32, epochs=50,
                   patience=10, seed=42):
    """
    Train GRU model with early stopping.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    input_size = X_train.shape[2]
    
    train_dataset = StockSequenceDataset(X_train, y_train)
    val_dataset = StockSequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = GRUModel(input_size, hidden_size, num_layers, dropout).to(device)
    
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                outputs = model(sequences)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                val_preds.extend(preds)
                val_probs.extend(probs)
                val_labels.extend(labels.numpy())
        
        val_auc = roc_auc_score(val_labels, val_probs)
        val_acc = accuracy_score(val_labels, val_preds)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, device


def evaluate_gru_model(model, X, y, device, batch_size=32):
    """
    Evaluate GRU model on a dataset.
    """
    dataset = StockSequenceDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    preds = []
    probs = []
    labels = []
    
    with torch.no_grad():
        for sequences, batch_labels in loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            batch_probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            preds.extend(batch_preds)
            probs.extend(batch_probs)
            labels.extend(batch_labels.numpy())
    
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    
    return acc, auc


def evaluate_model_with_gru(df, feature_cols, label_col="y", split="random",
                            val_size=0.2, test_size=0.1, sequence_length=5,
                            hidden_size=64, num_layers=2, dropout=0.3,
                            learning_rate=0.001, batch_size=32, epochs=50,
                            patience=10, seed=42):
    """
    Evaluate GRU model.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_data_gru(
        df, feature_cols, label_col, split, val_size, test_size, sequence_length, seed
    )
    
    print(f"  Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    val_dist = pd.Series(y_val).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    print(f"  Train class distribution: {train_dist.values}")
    print(f"  Val class distribution: {val_dist.values}")
    print(f"  Test class distribution: {test_dist.values}")
    print(f"  Majority baseline (test): {test_dist.max():.4f}")
    
    model, device = train_gru_model(
        X_train, y_train, X_val, y_val,
        hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
        learning_rate=learning_rate, batch_size=batch_size, epochs=epochs,
        patience=patience, seed=seed
    )
    
    val_acc, val_auc = evaluate_gru_model(model, X_val, y_val, device, batch_size)
    test_acc, test_auc = evaluate_gru_model(model, X_test, y_test, device, batch_size)
    
    return val_acc, val_auc, test_acc, test_auc, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random","chrono"], default="chrono",
                    help="random or chrono (chronological, leak-free)")
    ap.add_argument("--sequence-length", type=int, default=5,
                    help="Number of days in each sequence (default: 5)")
    ap.add_argument("--hidden-size", type=int, default=64,
                    help="GRU hidden size (default: 64)")
    ap.add_argument("--num-layers", type=int, default=2,
                    help="Number of GRU layers (default: 2)")
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="Dropout rate (default: 0.3)")
    ap.add_argument("--learning-rate", type=float, default=0.001,
                    help="Learning rate (default: 0.001)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Batch size (default: 32)")
    ap.add_argument("--epochs", type=int, default=50,
                    help="Maximum number of epochs (default: 50)")
    ap.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience (default: 10)")
    ap.add_argument("--val-size", type=float, default=0.15,
                    help="Validation set size (default: 0.15)")
    ap.add_argument("--test-size", type=float, default=0.15,
                    help="Test set size (default: 0.15)")
    args = ap.parse_args()

    tech = pd.read_csv("data/processed/merge_T_only_h5.csv", parse_dates=["date"])
    
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
    
    score = score[score['date'].isin(sentiment_dates)]
    triplet = triplet[triplet['date'].isin(sentiment_dates)]
    embed = embed[embed['date'].isin(sentiment_dates)]

    tech_features = [
        'ret_1d',
        'rsi',
        'sma_20',
        'price_to_sma_20',
        'mom_20',
        'vol_20',
        'bb_width_20',
        'bb_pct_20',
        'macd',
        'macd_signal',
        'vol_ratio',
        'close_position',
    ]
    
    tech_features = [f for f in tech_features if f in tech.columns]
    
    print(f"\n{'='*70}")
    print(f"Using {len(tech_features)} technical features:")
    print(f"  Features: {', '.join(tech_features)}")
    print(f"{'='*70}\n")

    print(f"\n{'='*70}")
    print(f"=== GRU Model Evaluation ===")
    print(f"Split: {args.split}, Sequence Length: {args.sequence_length}")
    print(f"Hidden Size: {args.hidden_size}, Layers: {args.num_layers}, Dropout: {args.dropout}")
    print(f"Learning Rate: {args.learning_rate}, Batch Size: {args.batch_size}")
    print(f"{'='*70}")

    results = []
    
    enhanced_sent_cols = [c for c in sent_enhanced.columns if c not in ['date', 'ticker']]
    
    def dedupe_features(feature_list):
        seen = set()
        return [x for x in feature_list if not (x in seen or seen.add(x))]
    
    model_configs = [
        ("Technical only", tech, tech_features),
        
        ("Technical + Score", 
         tech.merge(score, on=["ticker","date"], how="inner"), 
         dedupe_features(tech_features+["sent_score"])),
        
        ("Technical + Triplet", 
         tech.merge(triplet, on=["ticker","date"], how="inner"), 
         dedupe_features(tech_features+["p_pos","p_neu","p_neg"])),
        
        ("Technical + Enhanced Sentiment", 
         tech.merge(sent_enhanced, on=["ticker","date"], how="inner"),
         dedupe_features(tech_features+enhanced_sent_cols)),
    ]
    
    print(f"\n[info] Model configurations: {len(model_configs)}")
    print(f"       1. Technical only (baseline)")
    print(f"       2. Technical + Score (simple sentiment)")
    print(f"       3. Technical + Triplet (FinBERT probabilities)")
    print(f"       4. Technical + Enhanced Sentiment ({len(enhanced_sent_cols)} features)")
    
    for i, (model_name, df_model, features) in enumerate(model_configs, 1):
        print(f"\n{'='*70}")
        print(f"Model {i}: {model_name}")
        print(f"  Features: {len(features)}")
        print(f"{'='*70}")
        
        val_acc, val_auc, test_acc, test_auc, model = evaluate_model_with_gru(
            df_model, features, split=args.split,
            val_size=args.val_size, test_size=args.test_size,
            sequence_length=args.sequence_length,
            hidden_size=args.hidden_size, num_layers=args.num_layers,
            dropout=args.dropout, learning_rate=args.learning_rate,
            batch_size=args.batch_size, epochs=args.epochs,
            patience=args.patience
        )
        results.append([f"GRU: {model_name}", val_acc, val_auc, test_acc, test_auc])
        
        print(f"\n  Results:")
        print(f"    Val Accuracy: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        print(f"    Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    
    results_df = pd.DataFrame(results, columns=["Model", "Val Accuracy", "Val AUC", "Test Accuracy", "Test AUC"])
    print(f"\n{'='*70}")
    print("=== Final GRU Results Summary ===")
    print(f"{'='*70}")
    print(results_df.round(4).to_string(index=False))
    print(f"{'='*70}\n")
    
    Path("reports").mkdir(parents=True, exist_ok=True)
    results_df.to_csv("reports/gru_model_results.csv", index=False)
    print(f"[info] Saved results to reports/gru_model_results.csv")

if __name__ == "__main__":
    main()

