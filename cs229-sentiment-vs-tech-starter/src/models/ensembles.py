import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def train_eval_ensembles(X_train, y_train, X_test, y_test):
    out = {}
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=1337)
    rf.fit(X_train, y_train)
    pr = rf.predict_proba(X_test)[:,1]
    out['random_forest'] = dict(
        acc=float(accuracy_score(y_test, (pr>0.5).astype(int))),
        auc=float(roc_auc_score(y_test, pr))
    )
    xgb = XGBClassifier(n_estimators=600, max_depth=4, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=1337, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    px = xgb.predict_proba(X_test)[:,1]
    out['xgboost'] = dict(
        acc=float(accuracy_score(y_test, (px>0.5).astype(int))),
        auc=float(roc_auc_score(y_test, px))
    )
    return out
