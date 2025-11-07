#!/usr/bin/env python

import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

def add_intercept(X):
    """Add a column of ones as the first column."""
    X = np.asarray(X, dtype=float)
    return np.c_[np.ones((X.shape[0], 1)), X]

class LogisticRegression:
    """
    Logistic regression trained by Newton's method (CS229).
    Maximizes log-likelihood:
      l(theta) = sum_i y_i log h + (1-y_i) log (1-h),
      where h = sigmoid(X theta).
    """
    def __init__(self, max_iter=100, eps=1e-6, theta_0=None, verbose=True, l2=0.0):
        """
        Args:
            max_iter: maximum Newton updates.
            eps: L1 norm of step for convergence.
            theta_0: initial theta (1D array) or None -> zeros.
            verbose: print progress if True.
            l2: L2 regularization strength (lambda); 0 = no regularization.
        """
        self.theta = None if theta_0 is None else np.asarray(theta_0, dtype=float)
        self.max_iter = int(max_iter)
        self.eps = float(eps)
        self.verbose = bool(verbose)
        self.l2 = float(l2)

    def fit(self, X, y):
        """X: (n,d), y: (n,) in {0,1}."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        if self.theta is None:
            self.theta = np.zeros(d, dtype=float)

        I = np.eye(d)
        for t in range(self.max_iter):
            z = X @ self.theta                # (n,)
            h = _sigmoid(z)                   # (n,)

            # gradient with optional L2
            grad = (X.T @ (h - y)) / n
            if self.l2 > 0:
                grad += self.l2 * self.theta

            # Hessian X^T W X (no explicit diag) + L2*I
            w = h * (1 - h)                   # (n,)
            H = (X * w[:, None]).T @ X / n
            if self.l2 > 0:
                H = H + self.l2 * I

            # Newton step: solve H step = -grad
            step = -np.linalg.solve(H, grad)
            self.theta += step

            if self.verbose:
                if t % 10 == 0 or np.linalg.norm(step, 1) < self.eps:
                    ll = (y*np.log(h + 1e-12) + (1-y)*np.log(1-h + 1e-12)).mean()
                    reg = -0.5*self.l2*np.sum(self.theta**2) if self.l2>0 else 0.0
                    print(f"iter {t:3d} | step_L1 {np.linalg.norm(step,1):.3e} | avg_loglik {ll+reg:.6f}")

            if np.linalg.norm(step, 1) < self.eps:
                break

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return _sigmoid(X @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # ----- optional helpers -----
    def save(self, path, feature_names=None):
        np.savez(path, theta=self.theta, feature_names=np.array(feature_names, dtype=object) if feature_names is not None else None)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        theta = z["theta"]
        feature_names = None
        if "feature_names" in z.files and z["feature_names"] is not None:
            feature_names = list(z["feature_names"])
        model = LogisticRegression()
        model.theta = theta
        return model, feature_names

