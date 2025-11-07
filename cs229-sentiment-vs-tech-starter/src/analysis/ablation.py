import pandas as pd

def prepare_feature_sets(df, tech_cols, sent_cols, label_col='y'):
    X_tech = df[tech_cols].dropna()
    X_sent = df[sent_cols].dropna()
    X_comb = df[sorted(set(tech_cols) | set(sent_cols))].dropna()
    y = df.loc[X_comb.index, label_col]
    return X_tech, X_sent, X_comb, y
