# Do Sentiment Signals Improve Stock Movement Prediction?
*Ablation of technical vs. text-based features for next-day return prediction (CS229)*

**Authors:** Rosalie Massein · Zhenyu Chen  
**Term:** Fall 2025

## Overview
This repo tests whether **FinBERT-based sentiment** adds predictive power beyond **technical indicators** for binary next‑day return prediction.

### Project framing
- Label: `y_t = 1` if return at `t+1` > 0 else `0`
- Feature sets:
  1) Technical only
  2) Sentiment only (FinBERT P(pos)−P(neg) aggregated daily)
  3) Combined
- Models: Logistic Regression, Linear SVM, Random Forest, XGBoost (optionally MLP)
- Eval: Accuracy, AUC, calibration curves, error/regime analysis

## Quickstart
```bash
# 0) Create environment with uv
uv sync  # Creates .venv and installs dependencies from pyproject.toml

# 1) Set config (tickers, dates) in configs/default.yaml

# 2) Fetch OHLCV & headlines (requires internet access & API keys where relevant)
uv run python scripts/fetch_ohlcv.py --config configs/default.yaml
uv run python scripts/fetch_headlines.py --config configs/default.yaml

# 3) Build features
uv run python scripts/build_technical_features.py --config configs/default.yaml
uv run python scripts/build_sentiment_features.py --config configs/default.yaml

# 4) Train & evaluate
uv run python scripts/train_baselines.py --config configs/default.yaml
uv run python scripts/train_ensembles.py --config configs/default.yaml

# 5) Analysis
uv run python scripts/run_ablation.py --config configs/default.yaml
```

## Repo layout
```
data/
  raw/         # downloaded data (OHLCV, headlines)
  processed/   # merged, feature tables
notebooks/     # EDA & reports
src/
  data/        # loaders, downloaders
  features/    # feature engineering (technical/sentiment)
  models/      # training & evaluation
  analysis/    # ablation, calibration, regime/error analysis
  utils/       # common helpers (logging, io, time-split CV)
configs/       # YAML configs (tickers, date ranges, model params)
reports/
  figures/     # saved plots for poster/report
poster/        # poster assets
scripts/       # CLI entrypoints that call src modules
```

## Milestones (suggested)
- **Week 1**: Technical dataset + LR/SVM baselines
- **Week 3**: FinBERT sentiment integrated
- **Week 4**: Ablation complete
- **Week 6**: LLM significance weighting (optional)
- **Week 8**: Poster + final report

## Reproducibility
- **Time-split CV** (rolling origin) to avoid leakage
- Seeded training / fixed folds
- Config‑driven pipelines (see `configs/default.yaml`)

## Notes
- News/sentiment steps assume access to public sources/APIs; feel free to swap providers.
- Keep the `data/` folder **out** of git (see `.gitignore`).

---
