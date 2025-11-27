#!/usr/bin/env python3
"""
Enhanced sentiment feature builder with technical indicators.
Computes moving averages, volatility, momentum, etc. for sentiment scores.
"""
import argparse
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, '.')
from src.features.sentiment_technical import compute_sentiment_features, add_news_count_features


def aggregate_duplicates(df):
    """
    Aggregate duplicate ticker-date combinations by taking the mean.
    This handles multiple headlines on the same day for the same ticker.
    
    Args:
        df: DataFrame with columns: date, ticker, and sentiment columns
    
    Returns:
        DataFrame with one row per ticker-date combination
    """
    if df.empty:
        return df
    
    sentiment_cols = [c for c in df.columns if c not in ['date', 'ticker']]
    
    dupes_before = df.groupby(['ticker', 'date']).size().max()
    if dupes_before > 1:
        print(f"[info] Found up to {dupes_before} rows per ticker-date, aggregating by mean...")
        df = df.groupby(['ticker', 'date'], as_index=False)[sentiment_cols].mean()
        print(f"[info] After aggregation: {len(df)} rows (one per ticker-date)")
    else:
        print(f"[info] No duplicates found, all ticker-date combinations are unique")
    
    return df


def main():
    ap = argparse.ArgumentParser(description='Build enhanced sentiment features with MA, volatility, momentum, etc.')
    ap.add_argument("--headlines", required=True, help="Raw headlines CSV")
    ap.add_argument("--sent_score", default="", help="Sentiment score CSV")
    ap.add_argument("--sent_triplet", default="", help="Sentiment triplet CSV (p_pos, p_neu, p_neg)")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--windows", default="3,5,10", help="Window sizes for MA (comma-separated)")
    ap.add_argument("--vol_window", type=int, default=10, help="Window for volatility calculation")
    args = ap.parse_args()
    
    windows = tuple(int(x) for x in args.windows.split(','))
    
    # === 1. Load sentiment data ===
    dfs_to_merge = []
    
    if args.sent_score and Path(args.sent_score).exists():
        score_df = pd.read_csv(args.sent_score, parse_dates=["date"])
        score_df["ticker"] = score_df["ticker"].astype(str).str.upper()
        score_df = score_df.sort_values(["ticker", "date"])
        
        print(f"[info] Loaded {len(score_df)} sentiment score rows")
        print(f"[info] Columns: {score_df.columns.tolist()}")
        
        score_df = aggregate_duplicates(score_df)
        
        score_enhanced = compute_sentiment_features(score_df, windows=windows, vol_window=args.vol_window)
        dfs_to_merge.append(score_enhanced)
    
    if args.sent_triplet and Path(args.sent_triplet).exists():
        triplet_df = pd.read_csv(args.sent_triplet, parse_dates=["date"])
        triplet_df["ticker"] = triplet_df["ticker"].astype(str).str.upper()
        triplet_df = triplet_df.sort_values(["ticker", "date"])
        
        print(f"[info] Loaded {len(triplet_df)} sentiment triplet rows")
        
        triplet_df = aggregate_duplicates(triplet_df)
        
        triplet_enhanced = compute_sentiment_features(triplet_df, windows=windows, vol_window=args.vol_window)
        dfs_to_merge.append(triplet_enhanced)
    
    # === 2. Add news count features ===
    if Path(args.headlines).exists():
        headlines = pd.read_csv(args.headlines, parse_dates=["date"])
        headlines["ticker"] = headlines["ticker"].astype(str).str.upper()
        
        print(f"[info] Loaded {len(headlines)} headlines")
        
        news_counts = add_news_count_features(headlines, windows=windows)
        news_counts = aggregate_duplicates(news_counts)
        
        dfs_to_merge.append(news_counts)
    
    # === 3. Merge all features ===
    if not dfs_to_merge:
        raise ValueError("No sentiment data loaded!")
    
    result = dfs_to_merge[0]
    
    for i, df in enumerate(dfs_to_merge[1:], 1):
        # Identify overlapping columns (除了 date, ticker)
        overlap_cols = [c for c in df.columns if c in result.columns and c not in ['date', 'ticker']]
        
        if overlap_cols:
            print(f"[warn] Merge {i}: Found {len(overlap_cols)} overlapping columns, keeping first occurrence")
            df = df.drop(columns=overlap_cols)
        
        result = result.merge(df, on=["date", "ticker"], how="outer", suffixes=('', f'_dup{i}'))
    
    # === 4. Clean up and save ===
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    result = result.loc[:, ~result.columns.duplicated()]
    
    # Rename any _x, _y suffixes that might remain
    rename_dict = {}
    for col in result.columns:
        if col.endswith('_x'):
            rename_dict[col] = col[:-2]
        elif col.endswith('_y'):
            # Drop _y columns if _x exists
            continue
    
    if rename_dict:
        result = result.rename(columns=rename_dict)
        y_cols = [c for c in result.columns if c.endswith('_y')]
        if y_cols:
            result = result.drop(columns=y_cols)
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    
    print(f"\n[ok] Wrote {args.out}")
    print(f"     Rows: {len(result)}")
    print(f"     Columns: {len(result.columns)}")
    print(f"     Features per sentiment type:")
    
    base_cols = ['date', 'ticker']
    feature_cols = [c for c in result.columns if c not in base_cols]
    
    ma_features = [c for c in feature_cols if '_sma_' in c or '_ema_' in c or '_ma_' in c]
    mom_features = [c for c in feature_cols if '_mom_' in c or '_roc_' in c or '_trend_' in c]
    vol_features = [c for c in feature_cols if '_std_' in c or '_cv_' in c]
    range_features = [c for c in feature_cols if '_min_' in c or '_max_' in c or '_position_' in c]
    news_features = [c for c in feature_cols if 'news_' in c]
    cross_features = [c for c in feature_cols if c.startswith('sent_') and '_' not in c[5:]]
    other_features = [c for c in feature_cols if c not in ma_features + mom_features + vol_features + range_features + news_features + cross_features]
    
    print(f"       - Moving Averages: {len(ma_features)}")
    print(f"       - Momentum/Trend: {len(mom_features)}")
    print(f"       - Volatility: {len(vol_features)}")
    print(f"       - Range/Position: {len(range_features)}")
    print(f"       - News Count: {len(news_features)}")
    print(f"       - Cross-sentiment: {len(cross_features)}")
    print(f"       - Other: {len(other_features)}")
    print(f"     Total features: {len(feature_cols)}")


if __name__ == "__main__":
    main()

