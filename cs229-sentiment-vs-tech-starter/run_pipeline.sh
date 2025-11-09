#!/bin/bash
# Pipeline to run all data processing and training steps
# This script extracts the script running parts from EDA_Baseline-FinalResult.ipynb

set -e  # Exit on any error

# Change to project directory
cd /home/alex/src/cs229/ML-Stock-Prediction/cs229-sentiment-vs-tech-starter

# Set PYTHONPATH
export PYTHONPATH=.

echo "=== Step 1: Fetch OHLCV data ==="
uv run scripts/fetch_ohlcv.py --config configs/default.yaml

echo ""
echo "=== Step 2: Build technical features ==="
uv run scripts/build_technical_features.py --config configs/default.yaml

echo ""
echo "=== Step 3: Make labels (H=5) ==="
uv run scripts/make_labels.py \
  --input data/processed/technical_only.csv \
  --horizon_days 5 --threshold 0.0 \
  --out data/processed/technical_only_h5.csv

echo ""
echo "=== Step 4: Fetch headlines ==="
START="2025-08-06"
END="2025-11-04"
TICKS="AAPL,TSLA,MSFT,SPY,NVDA,GOOG,AMZN,META,NFLX,AMD"
OUT="data/raw/news/headlines.csv"

uv run scripts/fetch_headlines.py \
  --config configs/default.yaml \
  --start $START --end $END \
  --tickers $TICKS \
  --out $OUT

echo ""
echo "=== Step 5: Build FinBERT sentiment features ==="
# Ensure output dir exists
mkdir -p data/processed

# Remove buggy header env vars
unset HF_HUB_HEADERS
unset HUGGINGFACE_HUB_HEADERS

# Quiet noisy warnings
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

uv run scripts/build_sentiment_finbert.py \
  --input data/raw/news/headlines.csv \
  --out_triplet data/processed/sent_headlines_triplet.csv \
  --out_score data/processed/sent_headlines_score.csv

echo ""
echo "=== Step 6: Build headline sentiment embeddings (PCA16) ==="
uv run scripts/build_headline_sentiment.py \
  --input data/raw/news/headlines.csv \
  --encoder embed_pca16 \
  --out data/processed/sent_headlines_embed_pca16.csv

echo ""
echo "=== Step 7: Build news features with decay and event weights ==="
uv run scripts/build_news_simple.py \
  --headlines data/raw/news/headlines.csv \
  --sent_score data/processed/sent_headlines_score.csv \
  --sent_triplet data/processed/sent_headlines_triplet.csv \
  --sent_embed data/processed/sent_headlines_embed_pca16.csv \
  --out data/processed/news_features_simple.csv

echo ""
echo "=== Step 8: Merge technical + news features ==="
uv run scripts/merge_T_plus_news_simple.py \
  --tech data/processed/merge_T_only_h5.csv \
  --news data/processed/news_features_simple.csv \
  --out data/processed/merge_T_plus_news_simple_h5.csv

echo ""
echo "=== Step 9: Train logistic regression (random split) ==="
uv run scripts/train_logreg.py

echo ""
echo "=== Step 10: Train logistic regression (chronological split) ==="
uv run scripts/train_logreg.py --split chrono

echo ""
echo "=== Step 11: Train logistic regression with decay + event weights ==="
python scripts/train_logreg_simple.py --csv data/processed/merge_T_plus_news_simple_h5.csv

echo ""
echo "=== Pipeline complete! ==="

