import numpy as np
from typing import List, Tuple

def time_series_splits(dates, n_splits=5, test_size_days=60, gap_days=5):
    # dates: pandas.DatetimeIndex (sorted)
    # yields (train_idx, test_idx)
    import pandas as pd
    dates = pd.to_datetime(dates).sort_values()
    anchors = np.linspace(0, len(dates)-1, n_splits+1, dtype=int)[1:]
    for a in anchors:
        test_start = max(0, a - test_size_days)
        gap_start = max(0, test_start - gap_days)
        train_idx = np.arange(0, gap_start)
        test_idx  = np.arange(test_start, a)
        yield train_idx, test_idx
