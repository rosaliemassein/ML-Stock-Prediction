#!/usr/bin/env python3
"""
Train an LSTM (PyTorch) on sequences of technical + sentiment features.

- Uses data/processed/merge_T_only_h5.csv (technical + labels)
- Uses data/processed/sentiment_features_enhanced.csv (enhanced sentiment)
- Builds H-day sequences per ticker
- Chronological train/val/test split (by last date of sequence)
- Optimizes BCE loss, early-stops on validation AUC
- Saves metrics + model + scaler + feature list
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 1) Sequence building
# -----------------------------

def make_sequence_dataset(df, feature_cols, label_col="y", seq_len=5):
    """
    Build (X, y, last_date) sequences per ticker.
    X shape: (N_samples, seq_len, n_features)
    y shape: (N_samples,)
    dates: last date of each sequence
    """
    df = df.dropna(subset=feature_cols + [label_col]).copy()
    df = df.sort_values(["ticker", "date"])

    X_list, y_list, date_list = [], [], []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date")
        feats = g[feature_cols].values  # (T, F)
        labels = g[label_col].values
        dates = g["date"].values

        if len(g) < seq_len:
            continue

        for i in range(seq_len - 1, len(g)):
            X_window = feats[i - seq_len + 1:i + 1]
            y_t = labels[i]
            d_t = dates[i]

            X_list.append(X_window)
            y_list.append(y_t)
            date_list.append(d_t)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates = np.array(date_list)

    return X, y, dates


def chrono_split_by_dates(X, y, dates, val_size=0.15, test_size=0.15):
    """
    Chronological split on the last date of each sequence.
    Returns: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    all_dates = np.sort(np.unique(dates))
    train_cut = int((1 - val_size - test_size) * len(all_dates))
    val_cut = int((1 - test_size) * len(all_dates))

    train_date = all_dates[train_cut]
    val_date = all_dates[val_cut]

    train_mask = dates < train_date
    val_mask = (dates >= train_date) & (dates < val_date)
    test_mask = dates >= val_date

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# -----------------------------
# 2) PyTorch Dataset & Model
# -----------------------------

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # (N, seq_len, n_features)
        self.y = torch.from_numpy(y)  # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        out = self.dropout(last_hidden)
        logits = self.fc(out).squeeze(1)  # (batch,)
        return logits  # raw logits for BCEWithLogitsLoss


# -----------------------------
# 3) Training / Evaluation
# -----------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)


def eval_model(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            logits = model(Xb)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(targets, preds)
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = np.nan
    return acc, auc


# -----------------------------
# 4) Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=5,
                        help="Sequence length (H days)")
    parser.add_argument("--label-col", type=str, default="y",
                        help="Label column (e.g. 'y' or 'y_h5')")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (epochs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"PyTorch LSTM SEQUENCE MODEL on {device}")
    print(f"seq_len={args.seq_len}, label_col={args.label_col}")
    print("=" * 70)

    # ---- Load & merge data ----
    print("\n[1/4] Loading data...")
    tech = pd.read_csv("data/processed/merge_T_only_h5.csv", parse_dates=["date"])
    sent = pd.read_csv("data/processed/sentiment_features_enhanced.csv", parse_dates=["date"])
    sent["ticker"] = sent["ticker"].astype(str).str.upper()

    df = tech.merge(sent, on=["ticker", "date"], how="inner")

    # ---- Feature selection ----
    non_features = {"date", "ticker", args.label_col}
    feature_cols = [c for c in df.columns if c not in non_features]

    print(f"[info] Total features: {len(feature_cols)}")
    print(f"[info] Label column: {args.label_col}")
    print(f"[info] Example features: {feature_cols[:10]}")

    # ---- Build sequences ----
    print("\n[2/4] Building sequences...")
    X, y, dates = make_sequence_dataset(
        df,
        feature_cols=feature_cols,
        label_col=args.label_col,
        seq_len=args.seq_len
    )
    print(f"[info] X shape: {X.shape} (N, seq_len, n_features)")
    print(f"[info] y shape: {y.shape}, unique dates: {len(np.unique(dates))}")

    if X.shape[0] < 100:
        print("[WARN] Very few sequences – results will be noisy.")

    # ---- Chronological split ----
    print("\n[3/4] Chronological split...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = chrono_split_by_dates(
        X, y, dates,
        val_size=args.val_size,
        test_size=args.test_size
    )

    print(f"  Train samples: {X_train.shape[0]}")
    print(f"  Val   samples: {X_val.shape[0]}")
    print(f"  Test  samples: {X_test.shape[0]}")

    # ---- Scale features ----
    print("\n[3.1/4] Scaling features (fit on train)...")
    n_features = X_train.shape[2]
    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)

    # ---- DataLoaders ----
    train_ds = SequenceDataset(X_train_scaled, y_train)
    val_ds = SequenceDataset(X_val_scaled, y_val)
    test_ds = SequenceDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ---- Model, optimizer, loss ----
    print("\n[4/4] Building model...")
    model = LSTMClassifier(
        input_size=n_features,
        hidden_size=args.hidden_units,
        num_layers=1,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # ---- Training with early stopping on val AUC ----
    best_val_auc = -np.inf
    best_state = None
    patience_counter = 0

    print("\n=== Training ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_acc, train_auc = eval_model(model, train_loader, device)
        val_acc, val_auc = eval_model(model, val_loader, device)

        print(f"Epoch {epoch:02d} | "
              f"Loss={train_loss:.4f} | "
              f"Train AUC={train_auc:.4f} | "
              f"Val AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            patience_counter = 0
            print("  -> New best val AUC, saving state")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("  -> Early stopping triggered")
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final evaluation ----
    print("\n=== Final Evaluation (best model) ===")
    train_acc, train_auc = eval_model(model, train_loader, device)
    val_acc, val_auc = eval_model(model, val_loader, device)
    test_acc, test_auc = eval_model(model, test_loader, device)

    print(f"Train  Acc={train_acc:.4f}, AUC={train_auc:.4f}")
    print(f"Val    Acc={val_acc:.4f}, AUC={val_auc:.4f}")
    print(f"Test   Acc={test_acc:.4f}, AUC={test_auc:.4f}")

    # ---- Save model + meta ----
    from pathlib import Path
    import os, pickle

    # Make sure folders exist (relative to current working dir)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    # Model path
    model_path = Path("models") / f"lstm_pytorch_seqlen{args.seq_len}_hidden{args.hidden_units}.pt"
    torch.save(model.state_dict(), model_path)
    print("\n[info] Saved model state_dict to:", model_path.resolve())

    # Save scaler + features
    with open(Path("models") / "lstm_pytorch_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(Path("models") / "lstm_pytorch_features.txt", "w") as f:
        f.write("\n".join(feature_cols))

    # Save metrics
    summary_df = pd.DataFrame([{
        "seq_len": args.seq_len,
        "hidden_units": args.hidden_units,
        "dropout": args.dropout,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "train_acc": train_acc,
        "train_auc": train_auc,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "n_features": n_features
    }])

    summary_path = Path("reports") / "lstm_sequence_results_pytorch.csv"
    summary_df.to_csv(summary_path, index=False)
    print("[info] Saved metrics summary to:", summary_path.resolve())

    print("\nDone ✅")


if __name__ == "__main__":
    main()
