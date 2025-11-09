import pandas as pd
import numpy as np

def compute_sentiment_features(df, windows=(3, 5, 10), vol_window=10):
    """
    Compute technical-style features for sentiment data.
    
    Args:
        df: DataFrame with columns: date, ticker, and sentiment columns (e.g., sent_score, p_pos, p_neu, p_neg)
        windows: tuple of window sizes for moving averages
        vol_window: window for volatility calculation
    
    Returns:
        DataFrame with original columns plus sentiment technical features
    """
    out = df.sort_values(['ticker', 'date']).copy()
    
    # Identify sentiment columns
    sent_cols = [c for c in df.columns if c in ['sent_score', 'p_pos', 'p_neu', 'p_neg'] or 
                 c.startswith(('sent_', 'p_pos', 'p_neu', 'p_neg'))]
    
    if not sent_cols:
        print("[warn] No sentiment columns found")
        return out
    
    # For each sentiment column, compute features
    for sent_col in sent_cols:
        if sent_col not in out.columns:
            continue
            
        # Group by ticker for time series operations
        grouped = out.groupby('ticker')[sent_col]
        
        # === 1. Moving Averages (SMA & EMA) ===
        for w in windows:
            # Simple Moving Average
            out[f'{sent_col}_sma_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
            
            # Exponential Moving Average
            out[f'{sent_col}_ema_{w}'] = grouped.transform(lambda x: x.ewm(span=w, adjust=False).mean())
            
            # Distance from moving average (normalized)
            out[f'{sent_col}_dist_sma_{w}'] = out[sent_col] - out[f'{sent_col}_sma_{w}']
            out[f'{sent_col}_dist_ema_{w}'] = out[sent_col] - out[f'{sent_col}_ema_{w}']
        
        # === 2. Momentum & Rate of Change ===
        for w in windows:
            # Simple momentum (difference)
            out[f'{sent_col}_mom_{w}'] = grouped.transform(lambda x: x.diff(w))
            
            # Rate of change (percentage-like for sentiment)
            out[f'{sent_col}_roc_{w}'] = grouped.transform(
                lambda x: x.diff(w) / (x.shift(w).abs() + 0.1)  # Add small constant to avoid div by zero
            )
        
        # === 3. Volatility & Standard Deviation ===
        # Rolling standard deviation
        out[f'{sent_col}_std_{vol_window}'] = grouped.transform(
            lambda x: x.rolling(vol_window, min_periods=2).std()
        )
        
        # Coefficient of variation (normalized volatility)
        out[f'{sent_col}_cv_{vol_window}'] = out[f'{sent_col}_std_{vol_window}'] / (
            out[f'{sent_col}_sma_{vol_window}'].abs() + 0.1
        )
        
        # === 4. Min/Max over windows ===
        for w in [5, 10]:
            out[f'{sent_col}_max_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).max())
            out[f'{sent_col}_min_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).min())
            
            # Position within range (0 = at min, 1 = at max)
            range_val = out[f'{sent_col}_max_{w}'] - out[f'{sent_col}_min_{w}']
            out[f'{sent_col}_position_{w}'] = (
                (out[sent_col] - out[f'{sent_col}_min_{w}']) / (range_val + 1e-6)
            )
        
        # === 5. Trend Indicators ===
        # Simple linear trend (slope over window)
        for w in [5, 10]:
            out[f'{sent_col}_trend_{w}'] = grouped.transform(
                lambda x: x.rolling(w, min_periods=2).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0,
                    raw=True
                )
            )
        
        # === 6. Cumulative metrics ===
        # Cumulative sum (overall sentiment accumulation)
        out[f'{sent_col}_cumsum'] = grouped.transform('cumsum')
        
        # Exponentially weighted cumulative sum
        out[f'{sent_col}_ewm_cumsum'] = grouped.transform(
            lambda x: x.ewm(span=20, adjust=False).mean() * len(x)
        )
    
    # === 7. Cross-sentiment features (if we have triplet) ===
    if {'p_pos', 'p_neu', 'p_neg'}.issubset(out.columns):
        # Sentiment polarity (pos - neg)
        if 'sent_score' not in out.columns:
            out['sent_score'] = out['p_pos'] - out['p_neg']
        
        # Sentiment strength (total non-neutral)
        out['sent_strength'] = out['p_pos'] + out['p_neg']
        
        # Sentiment ratio (pos/neg ratio)
        out['sent_ratio'] = out['p_pos'] / (out['p_neg'] + 0.01)
        
        # Uncertainty (neutral proportion)
        out['sent_uncertainty'] = out['p_neu']
        
        # Dominant sentiment (which is highest)
        out['sent_dominant'] = out[['p_pos', 'p_neu', 'p_neg']].idxmax(axis=1).map({
            'p_pos': 1, 'p_neu': 0, 'p_neg': -1
        })
        
        # Sentiment consistency (std across triplet)
        out['sent_consistency'] = out[['p_pos', 'p_neu', 'p_neg']].std(axis=1)
    
    # === 8. News volume features (count of articles) ===
    # This would need the raw data before aggregation, but we can approximate
    # by tracking how sentiment volatility changes (proxy for article count)
    
    return out


def add_news_count_features(headlines_df, windows=(3, 5, 10)):
    """
    Add features based on news article counts per day.
    
    Args:
        headlines_df: Raw headlines with columns: date, ticker, text
        windows: Window sizes for moving averages
    
    Returns:
        DataFrame with date, ticker, and news count features
    """
    # Count articles per day per ticker
    counts = headlines_df.groupby(['date', 'ticker']).size().reset_index(name='news_count')
    counts = counts.sort_values(['ticker', 'date'])
    
    grouped = counts.groupby('ticker')['news_count']
    
    # Moving averages of news count
    for w in windows:
        counts[f'news_count_ma_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
        counts[f'news_count_std_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=2).std())
    
    # News surge indicator (current vs average)
    counts['news_surge_5'] = counts['news_count'] / (counts['news_count_ma_5'] + 0.5)
    counts['news_surge_10'] = counts['news_count'] / (counts['news_count_ma_10'] + 0.5)
    
    # Cumulative news count
    counts['news_cumcount'] = grouped.transform('cumsum')
    
    # Days since last news
    counts['days_since_news'] = grouped.transform(
        lambda x: x.eq(0).cumsum() - x.eq(0).cumsum().where(x.ne(0)).ffill().fillna(0)
    ).astype(int)
    
    return counts


