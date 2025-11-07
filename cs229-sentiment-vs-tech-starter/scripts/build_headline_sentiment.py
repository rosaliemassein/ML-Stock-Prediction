#!/usr/bin/env python

import argparse, os, re
import pandas as pd
import numpy as np

# Optional ML deps
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from transformers import AutoModel, AutoTokenizer as AutoTok2
import torch

from sklearn.decomposition import PCA


EVENT_RE = re.compile(r"(earnings|guidance|merger|acquisition|M&A|SEC|lawsuit|layoff|dividend|split|downgrade|upgrade|regulator|antitrust)", re.I)

def naive_lexicon_score(text: str) -> float:
    """Very small fallback scorer to keep pipeline running w/o transformers."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    pos = ["beat", "beats", "surge", "rally", "upgrade", "strong", "raise", "growth", "profit", "record", "bull"]
    neg = ["miss", "plunge", "selloff", "downgrade", "weak", "cut", "loss", "probe", "fraud", "bear", "lawsuit"]
    t = text.lower()
    s = sum(w in t for w in pos) - sum(w in t for w in neg)
    # map to [-1,1] with tanh-like squashing
    return float(np.tanh(s / 3.0))

def load_finbert_score_pipeline(model_name="ProsusAI/finbert"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    return TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True)

def load_bert_embedder(model_name="bert-base-uncased"):
    tok = AutoTok2.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    return tok, mdl

def encode_batch_score(texts, pipe):
    # returns (sent_score,) where score ~ pos - neg using FinBERT label mapping
    out = []
    for t in texts:
        if _HAS_TRANSFORMERS:
            res = pipe(t[:512])[0]  # list of dicts
            # FinBERT labels: ['positive','negative','neutral'] or similar
            d = {r['label'].lower(): r['score'] for r in res}
            pos = d.get('positive', 0.0)
            neg = d.get('negative', 0.0)
            out.append(pos - neg)
        else:
            out.append(naive_lexicon_score(t))
    return np.array(out, dtype=float)

def encode_batch_triplet(texts, pipe):
    P = []
    for t in texts:
        if _HAS_TRANSFORMERS:
            res = pipe(t[:512])[0]
            d = {r['label'].lower(): r['score'] for r in res}
            P.append([d.get('positive', 0.0), d.get('neutral', 0.0), d.get('negative', 0.0)])
        else:
            s = naive_lexicon_score(t)
            # simple mapping into a distribution
            pos = max(0.0, 0.5 + 0.5*s)
            neg = max(0.0, 0.5 - 0.5*s)
            neu = max(0.0, 1.0 - (pos+neg))
            Z = pos + neu + neg + 1e-12
            P.append([pos/Z, neu/Z, neg/Z])
    return np.array(P, dtype=float)

def encode_batch_embed(texts, tok, mdl, device=None):
    embs = []
    for t in texts:
        if not _HAS_EMBED:
            # fallback tiny bag-of-words-ish numeric hash to 32-d
            v = np.zeros(32, dtype=float)
            for w in re.findall(r"\w+", t.lower()):
                v[hash(w) % 32] += 1.0
            v = v / (np.linalg.norm(v) + 1e-12)
            embs.append(v)
            continue
        with torch.no_grad():
            enc = tok(t[:512], return_tensors="pt", truncation=True)
            outputs = mdl(**enc)
            # [CLS] = outputs.last_hidden_state[:,0,:]
            cls = outputs.last_hidden_state[:,0,:].squeeze(0).cpu().numpy()
            embs.append(cls)
    return np.vstack(embs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Headlines CSV with columns: date,ticker,text,...")
    ap.add_argument("--encoder", choices=["score","triplet","embed_pca16"], default="score")
    ap.add_argument("--events_only", action="store_true", help="Keep only event-type headlines")
    ap.add_argument("--pca_dim", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["date"])
    # Basic cleaning
    df = df.dropna(subset=["date","ticker","text"]).copy()
    df["ticker"] = df["ticker"].astype(str)

    if args.events_only:
        df = df[df["text"].astype(str).str.contains(EVENT_RE, na=False)]
        print(f"[info] events_only: {len(df)} rows after filter")

    if args.encoder in ("score","triplet") and _HAS_TRANSFORMERS:
        pipe = load_finbert_score_pipeline()
    else:
        pipe = None

    if args.encoder == "embed_pca16":
        tok, mdl = load_bert_embedder()  # or FinBERT if you prefer

    # Encode per headline
    texts = df["text"].astype(str).tolist()
    if args.encoder == "score":
        df["sent_score"] = encode_batch_score(texts, pipe)

    elif args.encoder == "triplet":
        P = encode_batch_triplet(texts, pipe)
        df["p_pos"], df["p_neu"], df["p_neg"] = P[:,0], P[:,1], P[:,2]

    else:  # embed_pca16
        E = encode_batch_embed(texts, tok, mdl)
        # PCA to pca_dim
        # Adjust n_components if we have too few samples
        max_components = min(E.shape[0], E.shape[1])
        n_comp = min(args.pca_dim, max_components)
        if n_comp < args.pca_dim:
            print(f"[warning] Only {E.shape[0]} samples available, reducing PCA from {args.pca_dim} to {n_comp} components")
        pca = PCA(n_components=n_comp, random_state=0)
        Z = pca.fit_transform(E)
        # Pad with zeros if needed
        if Z.shape[1] < args.pca_dim:
            Z = np.pad(Z, ((0,0),(0,args.pca_dim-Z.shape[1])))
        for j in range(args.pca_dim):
            df[f"emb_{j+1}"] = Z[:,j]

    # Aggregate to daily (ticker, date)
    keys = ["ticker","date"]
    if args.encoder == "score":
        out = df.groupby(keys, as_index=False)["sent_score"].mean()
    elif args.encoder == "triplet":
        out = df.groupby(keys, as_index=False)[["p_pos","p_neu","p_neg"]].mean()
    else:
        cols = [c for c in df.columns if c.startswith("emb_")]
        out = df.groupby(keys, as_index=False)[cols].mean()

    if args.out is None:
        enc = args.encoder
        args.out = f"data/processed/sent_headlines_{enc}.csv"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out} ({len(out)} rows)")

if __name__ == "__main__":
    main()
