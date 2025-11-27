#!/usr/bin/env python3
"""
PyTorch LSTM sequence model for CS229 stock project.

- Uses same data sources as tree models:
  * data/processed/merge_T_only_h5.csv
  * data/processed/sentiment_features_enhanced.csv
- Chronological split (train / val / test)
- Small hyperparameter sweep over:
    - seq_len
    - hidden units
    - dropout
    - learning rate
- Saves:
    reports/lstm_results.csv
    reports/lstm_test_predictions.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

# ---------------------------------------------------------
# 0. DEVICE
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {device}")

# ---------------------------------------------------------
# 1. DATA LOADING / PREP
# ---------------------------------------------------------
print("\n[1/4] Loading & preparing data...")

tech = pd.read_csv("data/processed/merge_T_only_h5.csv", parse_dates=["date"])
sent = pd.read_csv("data/processed/sentiment_features_enhanced.csv", parse_dates=["date"])
sent["ticker"] = sent["ticker"].astype(str).str.upper()

# Merge like in analyze_models.py
df = tech.merge(sent, on=["ticker", "date"], how="left")
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

# Technical features (same logic as analyze_models.py)
tech_features = [
    "ret_1d",
    "rsi",
    "sma_20",
    "price_to_sma_20",
    "mom_20",
    "vol_20",
    "bb_width_20",
    "bb_pct_20",
    "macd",
    "macd_signal",
    "vol_ratio",
    "close_position",
]
tech_features = [f for f in tech_features if f in df.columns]

# Sentiment features: only MA-ish ones to keep dimension reasonable
sent_cols = [c for c in sent.columns if c not in ["date", "ticker"]]
sent_ma_feats = [c for c in sent_cols if "_sma_" in c or "_ema_" in c or "_ma_" in c]
print(f"[info] #tech features       : {len(tech_features)}")
print(f"[info] #sentiment MA feats : {len(sent_ma_feats)}")

feature_cols = tech_features + sent_ma_feats
feature_cols = [f for f in feature_cols if f in df.columns]
print(f"[info] #TOTAL LSTM features: {len(feature_cols)}")

label_col = "y_h5"
df = df.dropna(subset=feature_cols + [label_col]).copy()

# Chronological split by date
dates = np.sort(df["date"].unique())
train_end = int(0.7 * len(dates))
val_end = int(0.85 * len(dates))

train = df[df["date"] < dates[train_end]]
val = df[(df["date"] >= dates[train_end]) & (df["date"] < dates[val_end])]
test = df[df["date"] >= dates[val_end]]

print(f"[info] Train dates: {train['date'].min().date()} → {train['date'].max().date()} (n={len(train)})")
print(f"[info] Val   dates: {val['date'].min().date()} → {val['date'].max().date()} (n={len(val)})")
print(f"[info] Test  dates: {test['date'].min().date()} → {test['date'].max().date()} (n={len(test)})")

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols])
X_val = scaler.transform(val[feature_cols])
X_test = scaler.transform(test[feature_cols])

y_train = train[label_col].values.astype(np.float32)
y_val = val[label_col].values.astype(np.float32)
y_test = test[label_col].values.astype(np.float32)

# ---------------------------------------------------------
# 2. SEQUENCE CREATION (ticker-wise)
# ---------------------------------------------------------
def make_sequences_tickerwise(df_part, X_part, y_part, seq_len, feature_cols):
    """
    Build sequences within each ticker so sequences don't cross tickers.
    """
    xs, ys = [], []
    start_idx = 0
    for _, grp in df_part.groupby("ticker"):
        n = len(grp)
        if n <= seq_len:
            start_idx += n
            continue

        for i in range(n - seq_len):
            idx_start = start_idx + i
            idx_end = idx_start + seq_len
            xs.append(X_part[idx_start:idx_end])
            ys.append(y_part[idx_end])  # predict next step
        start_idx += n

    if not xs:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,), dtype=np.float32)
    return np.stack(xs), np.array(ys, dtype=np.float32)


# ---------------------------------------------------------
# 3. MODEL
# ---------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # no dropout inside since num_layers=1
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        out = out[:, -1, :]            # last time step
        out = self.dropout(out)
        out = self.fc(out)
        return torch.sigmoid(out).flatten()


# ---------------------------------------------------------
# 4. TRAIN ONE CONFIG
# ---------------------------------------------------------
def train_one_config(seq_len, hidden, dropout, lr, max_epochs=40, batch_size=32):
    print(f"\n--- LSTM config: seq_len={seq_len}, hidden={hidden}, dropout={dropout}, lr={lr} ---")

    Xtr, ytr = make_sequences_tickerwise(train, X_train, y_train, seq_len, feature_cols)
    Xva, yva = make_sequences_tickerwise(val,   X_val,   y_val,   seq_len, feature_cols)
    Xte, yte = make_sequences_tickerwise(test,  X_test,  y_test,  seq_len, feature_cols)

    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        print("[warn] One of the splits has zero sequences, skipping this config.")
        return {
            "seq_len": seq_len,
            "hidden": hidden,
            "dropout": dropout,
            "lr": lr,
            "train_acc": np.nan,
            "train_auc": np.nan,
            "val_acc": np.nan,
            "val_auc": np.nan,
            "test_acc": np.nan,
            "test_auc": np.nan,
        }

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    yte_t = torch.tensor(yte, dtype=torch.float32, device=device)

    model = LSTMModel(n_features=len(feature_cols), hidden=hidden, dropout=dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = -np.inf
    best_state_dict = None
    patience = 5
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        epoch_losses = []

        for i in range(0, len(Xtr_t), batch_size):
            idx = perm[i : i + batch_size]
            xb = Xtr_t[idx]
            yb = ytr_t[idx]

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = model(Xva_t).detach().cpu().numpy()
        val_preds = (val_probs > 0.5).astype(int)

        try:
            val_auc = roc_auc_score(yva, val_probs)
        except ValueError:
            # Happens if only one class present in yva
            val_auc = np.nan
        val_acc = accuracy_score(yva, val_preds)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={np.mean(epoch_losses):.4f}  "
            f"val_ACC={val_acc:.4f}  val_AUC={val_auc:.4f}"
        )

        # Early stopping on val AUC
        if np.isnan(val_auc):
            continue
        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state_dict = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[info] Early stopping after {epoch} epochs (no val AUC improvement).")
                break

    # Load best model if we saved it
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Final metrics on train / val / test
    model.eval()
    with torch.no_grad():
        train_probs = model(Xtr_t).cpu().numpy()
        val_probs = model(Xva_t).cpu().numpy()
        test_probs = model(Xte_t).cpu().numpy()

    train_preds = (train_probs > 0.5).astype(int)
    val_preds = (val_probs > 0.5).astype(int)
    test_preds = (test_probs > 0.5).astype(int)

    def safe_auc(y_true, y_prob):
        try:
            return roc_auc_score(y_true, y_prob)
        except ValueError:
            return np.nan

    res = {
        "seq_len": seq_len,
        "hidden": hidden,
        "dropout": dropout,
        "lr": lr,
        "train_acc": accuracy_score(ytr, train_preds),
        "train_auc": safe_auc(ytr, train_probs),
        "val_acc": accuracy_score(yva, val_preds),
        "val_auc": safe_auc(yva, val_probs),
        "test_acc": accuracy_score(yte, test_preds),
        "test_auc": safe_auc(yte, test_probs),
    }

    # Also return test_probs and yte for ROC plotting
    res["y_test_true"] = yte
    res["y_test_prob"] = test_probs

    return res


# ---------------------------------------------------------
# 5. SMALL HYPERPARAM GRID & SWEEP
# ---------------------------------------------------------
print("\n[2/4] Running LSTM hyperparameter sweep...")

config_grid = [
    {"seq_len": 3, "hidden": 32, "dropout": 0.0, "lr": 1e-3},
    {"seq_len": 5, "hidden": 32, "dropout": 0.2, "lr": 1e-3},
    {"seq_len": 5, "hidden": 64, "dropout": 0.2, "lr": 1e-3},
    {"seq_len": 7, "hidden": 64, "dropout": 0.2, "lr": 5e-4},
]

all_results = []
best_config = None
best_val_auc_overall = -np.inf

for cfg in config_grid:
    res = train_one_config(**cfg)
    all_results.append(res)
    print(
        f"  -> Config result: "
        f"val_AUC={res['val_auc']:.4f}, test_AUC={res['test_auc']:.4f}"
    )

    if not np.isnan(res["val_auc"]) and res["val_auc"] > best_val_auc_overall:
        best_val_auc_overall = res["val_auc"]
        best_config = res

print("\n" + "=" * 70)
print("Best LSTM config based on validation AUC:")
if best_config is not None:
    print(
        f"  seq_len={best_config['seq_len']}, hidden={best_config['hidden']}, "
        f"dropout={best_config['dropout']}, lr={best_config['lr']}"
    )
    print(
        f"  Train: ACC={best_config['train_acc']:.4f}, AUC={best_config['train_auc']:.4f}"
    )
    print(
        f"  Val:   ACC={best_config['val_acc']:.4f},   AUC={best_config['val_auc']:.4f}"
    )
    print(
        f"  Test:  ACC={best_config['test_acc']:.4f},  AUC={best_config['test_auc']:.4f}"
    )
else:
    print("  [error] No valid configuration (all val_AUC were NaN?)")
print("=" * 70)

# ---------------------------------------------------------
# 6. SAVE RESULTS
# ---------------------------------------------------------
print("\n[3/4] Saving LSTM metrics...")

Path("reports").mkdir(parents=True, exist_ok=True)

rows = []
for r in all_results:
    rows.append(
        {
            "Model": "LSTM",
            "Train Accuracy": r["train_acc"],
            "Train AUC": r["train_auc"],
            "Val Accuracy": r["val_acc"],
            "Val AUC": r["val_auc"],
            "Test Accuracy": r["test_acc"],
            "Test AUC": r["test_auc"],
            "seq_len": r["seq_len"],
            "hidden_units": r["hidden"],
            "dropout": r["dropout"],
            "lr": r["lr"],
        }
    )

lstm_df = pd.DataFrame(rows)
lstm_df.to_csv("reports/lstm_results.csv", index=False)
print("[info] Saved metrics table to reports/lstm_results.csv")

# Save test predictions from best config for ROC comparison
if best_config is not None and "y_test_true" in best_config:
    test_pred_df = pd.DataFrame(
        {
            "y_true": best_config["y_test_true"],
            "y_prob": best_config["y_test_prob"],
        }
    )
    test_pred_df.to_csv("reports/lstm_test_predictions.csv", index=False)
    print("[info] Saved best-config test predictions to reports/lstm_test_predictions.csv")

print("\n[4/4] Done ✅")
