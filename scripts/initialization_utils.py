"""Shared emission payload for centroid-based PhyloGMM initializers."""

import numpy as np


def initialization_parameters(X, mu):
    """Return the emission JSON payload used by initializations.py.

    As with the k-means++ seed initializers, retain the supplied centers while
    estimating within-group variances from nearest-center assignments. Empty
    groups use the global population variance and have zero mixture weight.
    """
    X, mu = np.asarray(X, dtype=float), np.asarray(mu, dtype=float)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty, finite observation matrix")
    if (mu.ndim != 2 or not len(mu) or mu.shape[1] != X.shape[1]
            or not np.isfinite(mu).all()):
        raise ValueError("mu must be a nonempty, finite centroid matrix with X's feature dimension")
    K = len(mu)
    labels = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(2).argmin(1)
    sigma_diag = np.tile(X.var(axis=0), (K, 1))
    for k in range(K):
        assigned = X[labels == k]
        if len(assigned):
            sigma_diag[k] = assigned.var(axis=0)
    sigma_diag = np.maximum(sigma_diag, 1e-6)
    pi = np.bincount(labels, minlength=K).astype(float) / len(X)
    return {"mu": mu.tolist(), "sigma_diag": sigma_diag.tolist(), "pi": pi.tolist()}

