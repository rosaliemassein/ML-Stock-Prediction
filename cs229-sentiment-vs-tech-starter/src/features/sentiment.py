import pandas as pd

def aggregate_daily_sentiment(headlines_df, how='mean'):
    # headlines_df: columns [date, ticker, p_pos, p_neu, p_neg]
    df = headlines_df.copy()
    df['sentiment'] = df['p_pos'] - df['p_neg']
    if how == 'mean':
        agg = df.groupby(['date', 'ticker'])['sentiment'].mean().reset_index()
    elif how == 'median':
        agg = df.groupby(['date', 'ticker'])['sentiment'].median().reset_index()
    else:
        agg = df.groupby(['date', 'ticker'])['sentiment'].mean().reset_index()
    return agg
