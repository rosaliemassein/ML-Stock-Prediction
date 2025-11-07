import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score

def train_eval_baselines(X_train, y_train, X_test, y_test):
    results = {}
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    p = lr.predict_proba(X_test)[:,1]
    results['logistic'] = dict(
        acc=float((p>0.5).astype(int).mean() if y_test is None else accuracy_score(y_test, (p>0.5).astype(int))),
        auc=float(roc_auc_score(y_test, p))
    )
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    p_svm = svm.decision_function(X_test)
    results['linear_svm'] = dict(
        acc=float(accuracy_score(y_test, (p_svm>0).astype(int))),
        auc=float(roc_auc_score(y_test, p_svm))
    )
    return results
