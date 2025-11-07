import pandas as pd
import numpy as np

def compute_technical_features(df, windows=(5,10,20), rsi_window=14, vol_window=20):
    # df has columns: date, close, open, high, low, volume
    out = df.copy()
    out['ret_1d'] = out['close'].pct_change()
    for w in windows:
        out[f'sma_{w}'] = out['close'].rolling(w).mean()
        out[f'mom_{w}'] = out['close'].pct_change(w)
    # volatility
    out[f'vol_{vol_window}'] = out['ret_1d'].rolling(vol_window).std() * np.sqrt(252)
    # RSI
    delta = out['close'].diff()
    up = delta.clip(lower=0).rolling(rsi_window).mean()
    down = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rs = up / (down + 1e-12)
    out['rsi'] = 100 - (100 / (1 + rs))
    return out
