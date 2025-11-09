#!/usr/bin/env python
"""
Build sentiment features using FinBERT model.
Generates triplet (p_pos, p_neu, p_neg) and score (p_pos - p_neg) for each headline.
"""
import os
import sys
import math
import shutil
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Build sentiment features using FinBERT")
    parser.add_argument("--input", default="data/raw/news/headlines.csv", help="Input headlines CSV")
    parser.add_argument("--out_triplet", default="data/processed/sent_headlines_triplet.csv", help="Output triplet CSV")
    parser.add_argument("--out_score", default="data/processed/sent_headlines_score.csv", help="Output score CSV")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
    args = parser.parse_args()

    # --- 1) Clean up environment that breaks huggingface_hub ---
    for k in ("HF_HUB_HEADERS", "HUGGINGFACE_HUB_HEADERS"):
        if k in os.environ:
            os.environ.pop(k)

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Optional: if a local folder named 'ProsusAI/finbert' shadows the remote repo, remove it
    local_shadow = Path("ProsusAI/finbert")
    if local_shadow.exists() and local_shadow.is_dir():
        shutil.rmtree(local_shadow)

    # --- 2) Imports (after env cleanup) ---
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

    # --- 3) Load FinBERT pipeline ---
    model_id = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, use_safetensors=True)
    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, top_k=None)  # returns all labels with probabilities

    print("✅ FinBERT loaded and ready")

    # --- 4) IO paths ---
    inp = Path(args.input)
    out_triplet = Path(args.out_triplet)
    out_score = Path(args.out_score)
    out_triplet.parent.mkdir(parents=True, exist_ok=True)
    out_score.parent.mkdir(parents=True, exist_ok=True)

    # --- 5) Load headlines ---
    df = pd.read_csv(inp)
    # Expect at least columns: ['date','ticker','text']
    required_cols = {"date", "ticker", "text"}
    # Robust remap to lower in case of capitalization
    df.columns = [c.lower() for c in df.columns]
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns {required_cols - set(df.columns)} in {inp}")

    # --- 6) Run FinBERT in batches and collect outputs ---
    def run_batch(texts):
        # pipe returns list of list-of-dicts, one list per input
        # each inner list has dicts like {'label':'positive','score':0.73}, etc.
        res = pipe(texts)
        triplets = []
        for one in res:  # one = list for a single text
            probs = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            for d in one:
                probs[d["label"].lower()] = float(d["score"])
            p_pos = probs["positive"]
            p_neu = probs["neutral"]
            p_neg = probs["negative"]
            triplets.append((p_pos, p_neu, p_neg, p_pos - p_neg))
        return triplets

    BATCH = args.batch_size
    rows = []
    for i in tqdm(range(0, len(df), BATCH), desc="FinBERT"):
        batch = df.iloc[i:i+BATCH]
        out = run_batch(batch["text"].tolist())
        for (p_pos, p_neu, p_neg, score), (_, row) in zip(out, batch.iterrows()):
            rows.append({
                "date": row["date"],
                "ticker": row["ticker"],
                "p_pos": p_pos,
                "p_neu": p_neu,
                "p_neg": p_neg,
                "sent_score": score
            })

    sent = pd.DataFrame(rows)

    # --- 7) Save triplet + score as requested ---
    sent[["date", "ticker", "p_pos", "p_neu", "p_neg"]].to_csv(out_triplet, index=False)
    sent[["date", "ticker", "sent_score"]].to_csv(out_score, index=False)

    print(f"[ok] wrote {out_triplet} | rows={len(sent)} cols=5")
    print(f"[ok] wrote {out_score}   | rows={len(sent)} cols=3")


if __name__ == "__main__":
    main()

