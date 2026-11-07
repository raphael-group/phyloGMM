#!/usr/bin/env python3
"""Compare a phyloGMM fit with the simulation's true parameters.

Usage:
    python compare_to_truth.py <result.json> [data_dir]

Fitted states are matched to true states with the Hungarian algorithm on the
squared distance between fitted and true means. Then reports:
  ARI            adjusted Rand index of the fitted leaf labels vs the true labels
  mu RMSE        per-coordinate RMSE of the matched means
  Sigma error    mean relative Frobenius error of the matched covariances
  Q error        relative Frobenius error of the off-diagonal rates
  pi on root     fitted root probability on the true root state
Requires numpy and scipy.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


def ari(a, b):
    """Adjusted Rand index of two integer labelings."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    n = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(n, (a, b), 1)
    comb = lambda x: (x * (x - 1) / 2).sum()
    s, sa, sb, tot = comb(n), comb(n.sum(1)), comb(n.sum(0)), comb(np.array([len(a)]))
    exp = sa * sb / tot
    return (s - exp) / (0.5 * (sa + sb) - exp)


def main():
    res_path = sys.argv[1]
    data = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    r = json.load(open(res_path))
    t = json.load(open(os.path.join(data, "true_params.json")))
    mu, mu_t = np.array(r["mu"]), np.array(t["mu"])
    K, d = mu_t.shape

    cost = ((mu_t[:, None, :] - mu[None]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    perm = cols[np.argsort(rows)]  # true state i <-> fitted state perm[i]

    with open(os.path.join(data, "true_labels.csv")) as f:
        truth = {row["cell"]: int(row["state"]) for row in csv.DictReader(f)}
    fitted = np.array(r["leaf_labels"])
    true = np.array([truth[c] for c in r["leaf_names"]])

    Q, Q_t = np.array(r["Q"])[np.ix_(perm, perm)], np.array(t["Q"])
    off = ~np.eye(K, dtype=bool)
    S = np.array(r["Sigma"])[perm] if "Sigma" in r else None
    S_t = np.array(t["Sigma"])
    pi, pi_t = np.array(r["pi"])[perm], np.array(t["pi"])

    print("log-likelihood      %.4f" % r["log_likelihood"])
    print("ARI                 %.3f" % ari(fitted, true))
    print("mu RMSE             %.4f" % np.sqrt(cost[rows, cols].sum() / (K * d)))
    if S is not None:
        print("Sigma error         %.3f" % np.mean([np.linalg.norm(S[i] - S_t[i]) / np.linalg.norm(S_t[i]) for i in range(K)]))
    print("Q error (offdiag)   %.3f" % (np.linalg.norm((Q - Q_t)[off]) / np.linalg.norm(Q_t[off])))
    print("pi on true root     %.3f" % pi[np.argmax(pi_t)])
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print("\nFitted Q (rows/columns in true-state order):\n", np.round(Q, 2) + 0.0)
    print("\nTrue Q:\n", np.round(Q_t, 2) + 0.0)


if __name__ == "__main__":
    main()
