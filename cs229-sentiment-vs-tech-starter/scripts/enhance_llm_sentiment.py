#!/usr/bin/env python3
"""
Enhanced LLM sentiment feature builder with technical indicators.
Computes moving averages, volatility, momentum, etc. for LLM-weighted sentiment scores.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def compute_temporal_features(df, base_col, windows=(3, 5, 10), vol_window=10):
    """
    Compute temporal features for a given base column.
    
    Args:
        df: DataFrame with ticker and date sorted
        base_col: Column name to compute features from
        windows: Tuple of window sizes for moving averages
        vol_window: Window size for volatility calculation
    
    Returns:
        DataFrame with original columns + new temporal features
    """
    result = df.copy()
    
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_data = df[mask].sort_values('date')
        
        for w in windows:
            result.loc[mask, f'{base_col}_sma_{w}'] = ticker_data[base_col].rolling(w, min_periods=1).mean().values
            result.loc[mask, f'{base_col}_ema_{w}'] = ticker_data[base_col].ewm(span=w, min_periods=1).mean().values
            result.loc[mask, f'{base_col}_mom_{w}'] = ticker_data[base_col].diff(w).values
            result.loc[mask, f'{base_col}_roc_{w}'] = ticker_data[base_col].pct_change(w).values
            
            sma = ticker_data[base_col].rolling(w, min_periods=1).mean()
            result.loc[mask, f'{base_col}_dist_sma_{w}'] = (ticker_data[base_col] - sma).values
            
            ema = ticker_data[base_col].ewm(span=w, min_periods=1).mean()
            result.loc[mask, f'{base_col}_dist_ema_{w}'] = (ticker_data[base_col] - ema).values
            
            result.loc[mask, f'{base_col}_min_{w}'] = ticker_data[base_col].rolling(w, min_periods=1).min().values
            result.loc[mask, f'{base_col}_max_{w}'] = ticker_data[base_col].rolling(w, min_periods=1).max().values
            
            min_val = ticker_data[base_col].rolling(w, min_periods=1).min()
            max_val = ticker_data[base_col].rolling(w, min_periods=1).max()
            range_val = max_val - min_val
            range_val = range_val.replace(0, np.nan)
            result.loc[mask, f'{base_col}_position_{w}'] = ((ticker_data[base_col] - min_val) / range_val).values
            
            sma_trend = ticker_data[base_col].rolling(w, min_periods=1).mean().diff(1)
            result.loc[mask, f'{base_col}_trend_{w}'] = sma_trend.values
        
        result.loc[mask, f'{base_col}_std_{vol_window}'] = ticker_data[base_col].rolling(vol_window, min_periods=1).std().values
        
        mean_val = ticker_data[base_col].rolling(vol_window, min_periods=1).mean()
        std_val = ticker_data[base_col].rolling(vol_window, min_periods=1).std()
        result.loc[mask, f'{base_col}_cv_{vol_window}'] = (std_val / mean_val.abs()).values
        
        result.loc[mask, f'{base_col}_cumsum'] = ticker_data[base_col].cumsum().values
        result.loc[mask, f'{base_col}_ewm_cumsum'] = ticker_data[base_col].ewm(span=vol_window, min_periods=1).mean().cumsum().values
    
    return result


def create_cross_features(df):
    """Create cross-feature combinations from LLM sentiment columns."""
    result = df.copy()
    
    if 'sentiment_combined' in df.columns and 'num_headlines' in df.columns:
        result['sentiment_per_headline'] = df['sentiment_combined'] / (df['num_headlines'] + 1e-8)
    
    if 'sentiment_max' in df.columns and 'sentiment_combined' in df.columns:
        result['sentiment_max_ratio'] = df['sentiment_max'] / (df['sentiment_combined'].abs() + 1e-8)
    
    if 'sentiment_weighted_sum' in df.columns and 'num_headlines' in df.columns:
        result['sentiment_avg_weighted'] = df['sentiment_weighted_sum'] / (df['num_headlines'] + 1e-8)
    
    if 'sentiment_max' in df.columns:
        result['sentiment_max_abs'] = df['sentiment_max'].abs()
    
    if 'sentiment_combined' in df.columns:
        result['sentiment_combined_abs'] = df['sentiment_combined'].abs()
        result['sentiment_positive'] = (df['sentiment_combined'] > 0).astype(float)
        result['sentiment_negative'] = (df['sentiment_combined'] < 0).astype(float)
    
    return result


def main():
    ap = argparse.ArgumentParser(description="Enhance LLM sentiment features with temporal indicators")
    ap.add_argument("--input", required=True, help="LLM sentiment CSV (sentiment_features.csv)")
    ap.add_argument("--out", required=True, help="Output CSV with enhanced features")
    ap.add_argument("--windows", default="3,5,10", help="Window sizes for MA (comma-separated)")
    ap.add_argument("--vol_window", type=int, default=10, help="Window for volatility calculation")
    args = ap.parse_args()
    
    windows = tuple(int(x) for x in args.windows.split(','))
    
    print(f"[info] Loading LLM sentiment features from {args.input}")
    df = pd.read_csv(args.input, parse_dates=["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    print(f"[info] Loaded {len(df)} rows")
    print(f"[info] Columns: {df.columns.tolist()}")
    
    df['sentiment_raw_weight'] = df['sentiment_raw_weight'].fillna(0)
    df['sentiment_weighted_sum'] = df['sentiment_weighted_sum'].fillna(0)
    if 'sentiment_weighted_avg' in df.columns:
        df['sentiment_weighted_avg'] = df['sentiment_weighted_avg'].fillna(0)
    df['sentiment_combined'] = df['sentiment_raw_weight'] + df['sentiment_weighted_sum']
    
    print(f"\n[info] Computing temporal features for sentiment columns...")
    
    sentiment_cols = ['sentiment_raw_weight', 'sentiment_weighted_sum', 'sentiment_weighted_avg',
                      'sentiment_max', 'sentiment_combined', 'num_headlines']
    sentiment_cols = [c for c in sentiment_cols if c in df.columns]
    
    result = df.copy()
    
    for col in sentiment_cols:
        if col in df.columns:
            print(f"  Processing {col}...")
            result = compute_temporal_features(result, col, windows=windows, vol_window=args.vol_window)
    
    print(f"\n[info] Creating cross-features...")
    result = create_cross_features(result)
    
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Replace inf and -inf with NaN, then fill NaN with 0
    print(f"\n[info] Cleaning inf and NaN values...")
    result = result.replace([np.inf, -np.inf], np.nan)
    
    # Count NaNs before filling
    nan_counts = result.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts) > 0:
        print(f"  Found NaN values in {len(nan_counts)} columns")
        print(f"  Top 5 columns with NaNs: {nan_counts.head().to_dict()}")
    
    result = result.fillna(0)
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    
    print(f"\n[ok] Wrote {args.out}")
    print(f"     Rows: {len(result)}")
    print(f"     Columns: {len(result.columns)}")
    
    base_cols = ['date', 'ticker']
    feature_cols = [c for c in result.columns if c not in base_cols]
    
    ma_features = [c for c in feature_cols if '_sma_' in c or '_ema_' in c or '_dist_' in c]
    mom_features = [c for c in feature_cols if '_mom_' in c or '_roc_' in c or '_trend_' in c]
    vol_features = [c for c in feature_cols if '_std_' in c or '_cv_' in c]
    range_features = [c for c in feature_cols if '_min_' in c or '_max_' in c or '_position_' in c]
    cumulative_features = [c for c in feature_cols if '_cumsum' in c]
    cross_features = [c for c in feature_cols if 'sentiment_' in c and c not in sentiment_cols 
                      and c not in ma_features + mom_features + vol_features + range_features + cumulative_features]
    
    print(f"\n     Features breakdown:")
    print(f"       - Moving Averages: {len(ma_features)}")
    print(f"       - Momentum/Trend: {len(mom_features)}")
    print(f"       - Volatility: {len(vol_features)}")
    print(f"       - Range/Position: {len(range_features)}")
    print(f"       - Cumulative: {len(cumulative_features)}")
    print(f"       - Cross-features: {len(cross_features)}")
    print(f"       - Base features: {len(sentiment_cols)}")
    print(f"     Total features: {len(feature_cols)}")


if __name__ == "__main__":
    main()

