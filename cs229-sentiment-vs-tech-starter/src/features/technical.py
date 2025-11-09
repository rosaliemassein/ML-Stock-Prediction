import pandas as pd
import numpy as np

def compute_technical_features(df, windows=(5,10,20), rsi_window=14, vol_window=20):
    """
    Compute comprehensive technical indicators for stock price prediction.
    
    Args:
        df: DataFrame with columns: date, close, open, high, low, volume
        windows: tuple of window sizes for moving averages
        rsi_window: window for RSI calculation
        vol_window: window for volatility calculation
    
    Returns:
        DataFrame with original columns plus technical features
    """
    # Preserve ticker column if it exists
    has_ticker = 'ticker' in df.columns
    if has_ticker:
        ticker_col = df['ticker'].copy()
    
    out = df.copy()
    
    # === Basic Returns ===
    out['ret_1d'] = out['close'].pct_change(fill_method=None)
    
    # === Moving Averages (SMA & EMA) ===
    for w in windows:
        out[f'sma_{w}'] = out['close'].rolling(w).mean()
        out[f'ema_{w}'] = out['close'].ewm(span=w, adjust=False).mean()
        out[f'mom_{w}'] = out['close'].pct_change(w, fill_method=None)
        
        # Price to SMA ratio (normalized position)
        out[f'price_to_sma_{w}'] = out['close'] / out[f'sma_{w}'] - 1
        out[f'price_to_ema_{w}'] = out['close'] / out[f'ema_{w}'] - 1
    
    # === Volatility Indicators ===
    # Historical volatility
    out[f'vol_{vol_window}'] = out['ret_1d'].rolling(vol_window).std() * np.sqrt(252)
    
    # Average True Range (ATR)
    high_low = out['high'] - out['low']
    high_close = np.abs(out['high'] - out['close'].shift())
    low_close = np.abs(out['low'] - out['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out['atr_14'] = true_range.rolling(14).mean()
    out['atr_pct'] = out['atr_14'] / out['close']  # Normalized ATR
    
    # Bollinger Bands
    for w in [20]:
        sma = out['close'].rolling(w).mean()
        std = out['close'].rolling(w).std()
        out[f'bb_upper_{w}'] = sma + 2 * std
        out[f'bb_lower_{w}'] = sma - 2 * std
        out[f'bb_width_{w}'] = (out[f'bb_upper_{w}'] - out[f'bb_lower_{w}']) / sma
        # %B: position within bands (0 = lower band, 1 = upper band)
        out[f'bb_pct_{w}'] = (out['close'] - out[f'bb_lower_{w}']) / (out[f'bb_upper_{w}'] - out[f'bb_lower_{w}'])
    
    # === Momentum Indicators ===
    # RSI
    delta = out['close'].diff()
    up = delta.clip(lower=0).rolling(rsi_window).mean()
    down = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rs = up / (down + 1e-12)
    out['rsi'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    for w in [14]:
        low_min = out['low'].rolling(w).min()
        high_max = out['high'].rolling(w).max()
        out[f'stoch_k_{w}'] = 100 * (out['close'] - low_min) / (high_max - low_min + 1e-12)
        out[f'stoch_d_{w}'] = out[f'stoch_k_{w}'].rolling(3).mean()  # %D is 3-period SMA of %K
    
    # Williams %R
    for w in [14]:
        low_min = out['low'].rolling(w).min()
        high_max = out['high'].rolling(w).max()
        out[f'willr_{w}'] = -100 * (high_max - out['close']) / (high_max - low_min + 1e-12)
    
    # Rate of Change (ROC)
    for w in [10, 20]:
        out[f'roc_{w}'] = (out['close'] - out['close'].shift(w)) / out['close'].shift(w)
    
    # Commodity Channel Index (CCI)
    for w in [20]:
        typical_price = (out['high'] + out['low'] + out['close']) / 3
        sma_tp = typical_price.rolling(w).mean()
        mean_dev = typical_price.rolling(w).apply(lambda x: np.abs(x - x.mean()).mean())
        out[f'cci_{w}'] = (typical_price - sma_tp) / (0.015 * mean_dev + 1e-12)
    
    # MACD
    ema_12 = out['close'].ewm(span=12, adjust=False).mean()
    ema_26 = out['close'].ewm(span=26, adjust=False).mean()
    out['macd'] = ema_12 - ema_26
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
    out['macd_hist'] = out['macd'] - out['macd_signal']
    
    # === Trend Strength ===
    # ADX (Average Directional Index)
    for w in [14]:
        # Directional movement
        high_diff = out['high'].diff()
        low_diff = -out['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_dm_smooth = pd.Series(plus_dm, index=out.index).rolling(w).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=out.index).rolling(w).mean()
        atr_smooth = true_range.rolling(w).mean()
        
        plus_di = 100 * plus_dm_smooth / (atr_smooth + 1e-12)
        minus_di = 100 * minus_dm_smooth / (atr_smooth + 1e-12)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
        out[f'adx_{w}'] = dx.rolling(w).mean()
    
    # === Volume Indicators ===
    if 'volume' in out.columns:
        # Volume moving average and ratio
        out['vol_sma_20'] = out['volume'].rolling(20).mean()
        out['vol_ratio'] = out['volume'] / (out['vol_sma_20'] + 1e-12)
        
        # On Balance Volume (OBV)
        obv = (np.sign(out['close'].diff()) * out['volume']).fillna(0).cumsum()
        out['obv'] = obv
        out['obv_ema_20'] = obv.ewm(span=20, adjust=False).mean()
        
        # Volume Weighted Average Price (VWAP) - rolling 20-day
        typical_price = (out['high'] + out['low'] + out['close']) / 3
        out['vwap_20'] = (typical_price * out['volume']).rolling(20).sum() / (out['volume'].rolling(20).sum() + 1e-12)
        out['price_to_vwap'] = out['close'] / out['vwap_20'] - 1
    
    # === Price Pattern Features ===
    # Intraday range
    out['high_low_range'] = (out['high'] - out['low']) / out['close']
    
    # Close position in daily range (0 = at low, 1 = at high)
    out['close_position'] = (out['close'] - out['low']) / (out['high'] - out['low'] + 1e-12)
    
    # Upper and lower shadows (candle analysis)
    body = np.abs(out['close'] - out['open'])
    upper_shadow = out['high'] - np.maximum(out['close'], out['open'])
    lower_shadow = np.minimum(out['close'], out['open']) - out['low']
    out['upper_shadow'] = upper_shadow / (body + 1e-12)
    out['lower_shadow'] = lower_shadow / (body + 1e-12)
    
    # Body size relative to recent average
    out['body_size'] = body / out['close']
    out['body_size_ratio'] = body / (body.rolling(10).mean() + 1e-12)
    
    return out
