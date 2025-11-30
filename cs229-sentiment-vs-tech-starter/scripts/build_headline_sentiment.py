#!/usr/bin/env python
import argparse, os
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

def finbert_pipeline():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    m = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(m)
    mdl = AutoModelForSequenceClassification.from_pretrained(m, use_safetensors=True)
    return TextClassificationPipeline(model=mdl, tokenizer=tok, top_k=None, device=-1)

def encode_score(pipe, texts):
    out = []
    for t in texts:
        pred = pipe(t)[0]
        d = {p["label"].lower(): float(p["score"]) for p in pred}
        p_pos, p_neu, p_neg = d.get("positive",0.0), d.get("neutral",0.0), d.get("negative",0.0)
        out.append(p_pos - p_neg)
    return pd.Series(out, name="sent_score")

def encode_triplet(pipe, texts):
    rows = []
    for t in texts:
        pred = pipe(t)[0]
        d = {p["label"].lower(): float(p["score"]) for p in pred}
        rows.append({
            "p_pos": d.get("positive",0.0),
            "p_neu": d.get("neutral",0.0),
            "p_neg": d.get("negative",0.0),
        })
    return pd.DataFrame(rows)

def encode_embed_pca16(texts):
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    mdl = SentenceTransformer("all-MiniLM-L6-v2")
    emb = mdl.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
    Z = PCA(n_components=16, random_state=1337).fit_transform(emb)
    cols = [f"embed_pca16_{i:02d}" for i in range(Z.shape[1])]
    return pd.DataFrame(Z, columns=cols)

def aggregate_llm_scores(df):
    needed = {"date","ticker","llm_importance_score","text"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Input must contain columns: {needed}")
    
    pipe = finbert_pipeline()
    print("[info] computing sentiment scores with FinBERT...")
    df["sent_score"] = encode_score(pipe, df["text"])
    
    def agg_func(group):
        weights = group["llm_importance_score"].values
        scores = group["sent_score"].values
        weighted_products = weights * scores
        n = len(group)
        
        if n == 1:
            return pd.Series({
                "sentiment_raw_weight": weights[0],
                "sentiment_weighted_sum": weights[0],
                "sentiment_weighted_avg": weighted_products[0],
                "sentiment_max": weighted_products[0],
                "num_headlines": n
            })
        else:
            weighted_sum = np.sum(weighted_products)
            weighted_avg = weighted_sum / n
            max_val = np.max(weighted_products)
            avg_weight = np.mean(weights)
            
            return pd.Series({
                "sentiment_raw_weight": avg_weight,
                "sentiment_weighted_sum": weighted_sum,
                "sentiment_weighted_avg": weighted_avg,
                "sentiment_max": max_val,
                "num_headlines": n
            })
    
    agg = df.groupby(["date","ticker"], as_index=False).apply(agg_func, include_groups=False)
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with columns: date,ticker,text or date,ticker,llm_importance_score")
    ap.add_argument("--encoder", choices=["score","triplet","embed_pca16","aggregate_llm"], help="Encoder to use")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["date"]).sort_values(["ticker","date"])
    
    if args.encoder == "aggregate_llm":
        out = aggregate_llm_scores(df)
    else:
        needed = {"date","ticker","text"}
        if not needed.issubset(df.columns):
            raise ValueError(f"{args.input} must contain columns: {needed}")

        if args.encoder in {"score","triplet"}:
            pipe = finbert_pipeline()
            if args.encoder == "score":
                feat = encode_score(pipe, df["text"])
                out = pd.concat([df[["date","ticker"]], feat], axis=1)
            else:
                feat = encode_triplet(pipe, df["text"])
                out = pd.concat([df[["date","ticker"]], feat], axis=1)
        else:
            feat = encode_embed_pca16(df["text"])
            out = pd.concat([df[["date","ticker"]], feat], axis=1)

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out} | rows={len(out)} | cols={len(out.columns)}")

if __name__ == "__main__":
    main()
