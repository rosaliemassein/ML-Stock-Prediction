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
OUT="data/raw/news/headlines.csv"

uv run scripts/fetch_headlines.py \
  --config configs/default.yaml \
  --out $OUT

echo ""
echo "=== Step 5: Build headline sentiment features ==="
# Ensure output dir exists
mkdir -p data/processed

# Remove buggy header env vars
unset HF_HUB_HEADERS
unset HUGGINGFACE_HUB_HEADERS

# Quiet noisy warnings
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# Build sentiment score (p_pos - p_neg)
uv run scripts/build_headline_sentiment.py \
  --input data/raw/news/headlines.csv \
  --encoder score \
  --out data/processed/sent_headlines_score.csv

echo ""
echo "=== Step 6: Build sentiment triplet features ==="
uv run scripts/build_headline_sentiment.py \
  --input data/raw/news/headlines.csv \
  --encoder triplet \
  --out data/processed/sent_headlines_triplet.csv

echo ""
echo "=== Step 7: Build sentiment embedding features ==="
uv run scripts/build_headline_sentiment.py \
  --input data/raw/news/headlines.csv \
  --encoder embed_pca16 \
  --out data/processed/sent_headlines_embed_pca16.csv

echo ""
echo "=== Step 8: Build enhanced sentiment features (MA, volatility, momentum) ==="
uv run scripts/build_sentiment_features.py \
  --headlines data/raw/news/headlines.csv \
  --sent_score data/processed/sent_headlines_score.csv \
  --sent_triplet data/processed/sent_headlines_triplet.csv \
  --out data/processed/sentiment_features_enhanced.csv \
  --windows 3,5,10 \
  --vol_window 10

echo ""
echo "=== Step 9: Build news features with decay and event weights ==="
uv run scripts/build_news_simple.py \
  --headlines data/raw/news/headlines.csv \
  --sent_score data/processed/sent_headlines_score.csv \
  --sent_triplet data/processed/sent_headlines_triplet.csv \
  --sent_embed data/processed/sent_headlines_embed_pca16.csv \
  --out data/processed/news_features_simple.csv

echo ""
echo "=== Step 10: Merge technical + news features ==="
uv run scripts/merge_T_plus_news_simple.py \
  --tech data/processed/merge_T_only_h5.csv \
  --news data/processed/news_features_simple.csv \
  --out data/processed/merge_T_plus_news_simple_h5.csv

# echo ""
# echo "=== Step 11: Train XGBoost, LightGBM, and Random Forest models ==="
# uv run scripts/train_xgboost.py \
#   --model all \
#   --split chrono \
#   --tune \

# echo ""
# echo "=== Step 11.5: Train models with top-K features (10, 20, 30, 50) ==="
# uv run scripts/train_top_features.py \
#   --split chrono \
#   --top-k 10,20,30,50 \
#   --tune \


echo ""
echo "=== Pipeline complete! All models trained. ==="

