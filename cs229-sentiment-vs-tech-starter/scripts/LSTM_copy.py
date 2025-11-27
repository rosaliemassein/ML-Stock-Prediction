import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# LOAD DATA (same as trees)
# ---------------------------------------------------------
tech = pd.read_csv("data/processed/merge_T_only_h5.csv", parse_dates=["date"])
sent = pd.read_csv("data/processed/sentiment_features_enhanced.csv", parse_dates=["date"])
sent["ticker"] = sent["ticker"].astype(str).str.upper()

# ---- NEW: rebuild ret_fwd_5d and y_h5 if they are missing ----
if "y_h5" not in tech.columns:
    tech = tech.sort_values(["ticker", "date"])
    # close price 5 days in the future (per ticker)
    tech["close_fwd_5d"] = tech.groupby("ticker")["close"].shift(-5)
    # forward 5-day simple return
    tech["ret_fwd_5d"] = (tech["close_fwd_5d"] - tech["close"]) / tech["close"]
    # binary label: 1 if future 5-day return > 0, else 0
    tech["y_h5"] = (tech["ret_fwd_5d"] > 0).astype(int)
    tech = tech.drop(columns=["close_fwd_5d"])

# merge with sentiment (now tech already has y_h5)
df = tech.merge(sent, on=["ticker", "date"], how="left")
df = df.sort_values(["ticker", "date"])

# Same features used in tree models
tech_features = [
    c for c in tech.columns 
    if c not in ['date','ticker','open','high','low','close',
                 'volume','y','ret_fwd_5d','y_h5']
]
enhanced_sent_cols = [c for c in sent.columns if c not in ['date','ticker']]
enhanced_ma_features = [
    c for c in enhanced_sent_cols 
    if '_sma_' in c or '_ema_' in c or '_ma_' in c
]
all_features = [f for f in (tech_features + enhanced_ma_features) if f in df.columns]

label_col = "y_h5"

# ---------------------------------------------------------
# CREATE CHRONO SPLIT
# ---------------------------------------------------------
df = df.dropna(subset=all_features + [label_col])
dates = np.sort(df["date"].unique())
train_end = int(0.7 * len(dates))
val_end   = int(0.85 * len(dates))

train = df[df["date"] < dates[train_end]]
val   = df[(df["date"] >= dates[train_end]) & (df["date"] < dates[val_end])]
test  = df[df["date"] >= dates[val_end]]

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(train[all_features])
X_val   = scaler.transform(val[all_features])
X_test  = scaler.transform(test[all_features])

y_train = train[label_col].values
y_val   = val[label_col].values
y_test  = test[label_col].values

# ---------------------------------------------------------
# CREATE SEQUENCES FOR LSTM
# ---------------------------------------------------------
seq_len = 5

def make_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(xs), np.array(ys)

Xtr, ytr = make_sequences(X_train, y_train, seq_len)
Xva, yva = make_sequences(X_val,   y_val,   seq_len)
Xte, yte = make_sequences(X_test,  y_test,  seq_len)

# Convert to torch
Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(device)
ytr_t = torch.tensor(ytr, dtype=torch.float32).to(device)
Xva_t = torch.tensor(Xva, dtype=torch.float32).to(device)
yva_t = torch.tensor(yva, dtype=torch.float32).to(device)
Xte_t = torch.tensor(Xte, dtype=torch.float32).to(device)
yte_t = torch.tensor(yte, dtype=torch.float32).to(device)

# ---------------------------------------------------------
# LSTM MODEL
# ---------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return torch.sigmoid(self.fc(out))

model = LSTMModel(n_features=len(all_features)).to(device)
criterion = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
epochs = 30
batch_size = 64

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(len(Xtr_t))
    losses = []

    for i in range(0, len(Xtr_t), batch_size):
        idx = perm[i:i+batch_size]
        xb = Xtr_t[idx]
        yb = ytr_t[idx]

        opt.zero_grad()
        preds = model(xb).flatten()
        loss = criterion(preds, yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # validation AUC each epoch (for monitoring)
    model.eval()
    with torch.no_grad():
        val_probs_epoch = model(Xva_t).flatten().cpu().numpy()
        val_auc_epoch = roc_auc_score(yva, val_probs_epoch)

    print(f"Epoch {epoch+1}: train_loss={np.mean(losses):.4f}  val_AUC={val_auc_epoch:.4f}")

# ---------------------------------------------------------
# FINAL TRAIN / VAL / TEST EVALUATION
# ---------------------------------------------------------
model.eval()
with torch.no_grad():
    train_probs = model(Xtr_t).flatten().cpu().numpy()
    val_probs   = model(Xva_t).flatten().cpu().numpy()
    test_probs  = model(Xte_t).flatten().cpu().numpy()

train_auc = roc_auc_score(ytr, train_probs)
val_auc   = roc_auc_score(yva, val_probs)
test_auc  = roc_auc_score(yte, test_probs)

train_acc = accuracy_score(ytr, (train_probs > 0.5).astype(int))
val_acc   = accuracy_score(yva, (val_probs   > 0.5).astype(int))
test_acc  = accuracy_score(yte, (test_probs  > 0.5).astype(int))

print("\n=============================")
print("LSTM RESULTS")
print("=============================")
print("Train AUC:", train_auc, " Train ACC:", train_acc)
print("Val   AUC:", val_auc,   " Val   ACC:", val_acc)
print("Test  AUC:", test_auc,  " Test  ACC:", test_acc)

# ---------------------------------------------------------
# SAVE RESULTS + TEST PREDICTIONS
# ---------------------------------------------------------
Path("reports").mkdir(parents=True, exist_ok=True)

lstm_df = pd.DataFrame([{
    "Model": "LSTM",
    "Train Accuracy": train_acc,
    "Train AUC": train_auc,
    "Val Accuracy": val_acc,
    "Val AUC": val_auc,
    "Test Accuracy": test_acc,
    "Test AUC": test_auc
}])
lstm_df.to_csv("reports/lstm_results.csv", index=False)
print("Saved metrics to reports/lstm_results.csv")

test_preds_df = pd.DataFrame({
    "y_true": yte,
    "y_prob": test_probs
})
test_preds_df.to_csv("reports/lstm_test_predictions.csv", index=False)
print("Saved test predictions to reports/lstm_test_predictions.csv")
