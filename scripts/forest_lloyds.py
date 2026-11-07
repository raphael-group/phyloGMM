#!/usr/bin/env python
"""Plain and tree-regularized Lloyd's (k-means) on a forest of lineage trees.

Objective (one coherent function; Lloyd's = block-coordinate descent):
  J(z, mu) = sum_{leaves} ||x_i - mu_{z_i}||^2  +  lambda * sum_{tree edges} w_uv * 1[z_u != z_v]
  w_uv = 1/branch_length.  lambda=0 -> plain k-means on leaves.

z-step: minimize J over ALL node labels (leaves + internal ancestors) given mu.
        On a tree this is EXACT via Sankoff min-sum DP, O(nodes*K):
          leaf potential   phi_i(j) = ||x_i - mu_j||^2 ;  internal potential = 0
          child->parent msg contribution(j) = min( cost_child[j], min_c cost_child[c] + lambda*w )
mu-step: mu_j = mean of LEAVES currently labeled j (Potts term has no mu).

All trees share one set of means mu over the pooled leaves; the DP runs per tree.
Used by make_initializations.py.
"""
import os, sys, re, threading, argparse
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

K    = 10
BLFLOOR = 1e-2

# ---------------------------------------------------------------- newick ----
def parse_newick(s):
    s = s.strip().rstrip(';'); pos = [0]
    def clade():
        node = {'ch': [], 'len': 0.0, 'name': None}
        if s[pos[0]] == '(':
            pos[0] += 1
            while True:
                node['ch'].append(clade())
                if s[pos[0]] == ',': pos[0] += 1; continue
                if s[pos[0]] == ')': pos[0] += 1; break
        m = re.match(r'[^:,()]+', s[pos[0]:])
        if m: node['name'] = m.group(0); pos[0] += len(m.group(0))
        if pos[0] < len(s) and s[pos[0]] == ':':
            pos[0] += 1; m = re.match(r'[-0-9.eE+]+', s[pos[0]:])
            node['len'] = float(m.group(0)); pos[0] += len(m.group(0))
        return node
    return clade()

def tree_to_arrays(root, name2g):
    """Linearize to post-order: children come before parents. root = last id."""
    po = []
    def visit(node):
        cids = [visit(c) for c in node['ch']]
        lid = len(po); node['lid'] = lid; node['cids'] = cids; po.append(node); return lid
    visit(root)
    n = len(po)
    children = [nd['cids'] for nd in po]
    blen = np.array([nd['len'] for nd in po], dtype=np.float64)
    parent = np.full(n, -1, dtype=int)
    for lid, cids in enumerate(children):
        for c in cids: parent[c] = lid
    leaf_g = np.full(n, -1, dtype=int)
    for lid, nd in enumerate(po):
        if not nd['ch']:
            leaf_g[lid] = name2g[nd['name']]
    return dict(n=n, children=children, blen=blen, parent=parent, leaf_g=leaf_g, root=n - 1)

# ---------------------------------------------------------------- z-step ----
def sankoff(tree, emis, lam, K, N):
    """Return global leaf-label array (-1 for non-leaves-of-this-tree)."""
    n = tree['n']; children = tree['children']; blen = tree['blen']; leaf_g = tree['leaf_g']; parent = tree['parent']
    cost = np.zeros((n, K))
    for lid in range(n):                         # post-order (children before parent)
        cu = emis[leaf_g[lid]].copy() if leaf_g[lid] >= 0 else np.zeros(K)
        for c in children[lid]:
            cc = cost[c]; w = lam / max(blen[c], BLFLOOR)
            cu += np.minimum(cc, cc.min() + w)
        cost[lid] = cu
    labels = np.empty(n, dtype=int); root = tree['root']
    for lid in range(n - 1, -1, -1):             # reverse post-order (parent before child)
        if lid == root:
            labels[lid] = int(cost[lid].argmin())
        else:
            w = lam / max(blen[lid], BLFLOOR); cand = cost[lid] + w; cand[labels[parent[lid]]] -= w
            labels[lid] = int(cand.argmin())
    out = np.full(N, -1, dtype=int)
    lg = leaf_g[leaf_g >= 0]; out[lg] = labels[leaf_g >= 0]
    return out

# ---------------------------------------------------------------- lloyds ----
def kmeanspp_seed(X, K, rng):
    n = X.shape[0]; centers = [int(rng.integers(n))]
    d2 = ((X - X[centers[0]]) ** 2).sum(1)
    for _ in range(K - 1):
        p = d2 / d2.sum(); c = int(rng.choice(n, p=p)); centers.append(c)
        d2 = np.minimum(d2, ((X - X[c]) ** 2).sum(1))
    return X[centers].copy()

def greedy_kmeanspp_seed(X, K, rng, n_local_trials=None):
    """Seed centers with scikit-learn's greedy k-means++ strategy.

    At each step, sample ``n_local_trials`` candidates from the usual D^2
    distribution and keep the candidate that minimizes the resulting total
    squared distance to the nearest center.  A value of 1 recovers ordinary
    k-means++; the scikit-learn default is ``2 + floor(log(K))``.
    """
    n = X.shape[0]
    if K < 1:
        raise ValueError("K must be at least 1")
    if K > n:
        raise ValueError(f"n_samples={n} should be >= K={K}")
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(K))
    if n_local_trials < 1:
        raise ValueError("n_local_trials must be at least 1")

    centers = [int(rng.integers(n))]
    d2 = ((X - X[centers[0]]) ** 2).sum(1)
    current_potential = d2.sum()

    for _ in range(K - 1):
        # Draw candidates with replacement from the D^2 distribution.  The
        # inverse-CDF form also matches scikit-learn when the potential is 0.
        draws = rng.random(n_local_trials) * current_potential
        candidates = np.searchsorted(np.cumsum(d2), draws)
        candidates = np.minimum(candidates, n - 1)

        candidate_d2 = np.empty((n_local_trials, n), dtype=np.float64)
        for trial, candidate in enumerate(candidates):
            new_d2 = ((X - X[candidate]) ** 2).sum(1)
            candidate_d2[trial] = np.minimum(d2, new_d2)

        candidate_potentials = candidate_d2.sum(1)
        best_trial = int(candidate_potentials.argmin())
        centers.append(int(candidates[best_trial]))
        d2 = candidate_d2[best_trial]
        current_potential = candidate_potentials[best_trial]

    return X[centers].copy()

def run_lloyds(X, forest, mu0, lam, K, max_iter=40):
    N = X.shape[0]; mu = mu0.copy(); prev = None
    for _ in range(max_iter):
        emis = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(2)      # (N,K)
        if lam == 0:
            lab = emis.argmin(1)
        else:
            lab = np.full(N, -1, dtype=int)
            for tree in forest:
                l = sankoff(tree, emis, lam, K, N); lab = np.where(l >= 0, l, lab)
        if prev is not None and np.array_equal(lab, prev): 
            print("lloyds converged after", _ + 1, "iterations", flush=True)
            break
        prev = lab
        for k in range(K):
            m = lab == k
            if m.sum() > 0: mu[k] = X[m].mean(0)
    return lab

def params(X, lab, K, covariance_type="diag"):
    """Estimate cluster means and population covariances (normalization by N).

    Returns means of shape (K, D) and either diagonal variances of shape
    (K, D) for covariance_type="diag" (the default), or full covariance
    matrices of shape (K, D, D) for covariance_type="full". Empty clusters
    use the mean and covariance of all observations. Singleton clusters
    have zero covariance.
    """
    if covariance_type not in ("diag", "full"):
        raise ValueError("covariance_type must be 'diag' or 'full'")

    Dm = X.shape[1]; mu = np.zeros((K, Dm))
    vs = np.zeros((K, Dm) if covariance_type == "diag" else (K, Dm, Dm))
    for k in range(K):
        m = lab == k
        cluster = X[m] if m.any() else X
        mu[k] = cluster.mean(0)
        if covariance_type == "diag":
            vs[k] = cluster.var(0)
        else:
            centered = cluster - mu[k]
            vs[k] = centered.T @ centered / cluster.shape[0]
    return mu, vs

def match_err(mp, vp, mt, vt):
    C = ((mp[:, None, :] - mt[None, :, :]) ** 2).sum(2)
    ri, ci = linear_sum_assignment(C)
    return (float(np.mean(np.linalg.norm(mp[ri] - mt[ci], axis=1))),
            float(np.mean(np.linalg.norm(vp[ri] - vt[ci], axis=1))))

def match_by_overlap(lab, st, Kk):
    M = np.zeros((Kk, Kk))
    for k in range(Kk):
        mk = lab == k
        for j in range(Kk):
            M[k, j] = np.sum(mk & (st == j))
    ri, ci = linear_sum_assignment(-M)
    ct = np.empty(Kk, dtype=int); ct[ri] = ci
    return ct   # ct[k] = matched true state for predicted cluster k

def match_err_overlap(mp, vp, mt, vt, lab, st, Kk):
    ct = match_by_overlap(lab, st, Kk)
    return (float(np.mean(np.linalg.norm(mp - mt[ct], axis=1))),
            float(np.mean(np.linalg.norm(vp - vt[ct], axis=1))))

def sankoff_cost(tree, emis, lam, Kk):
    n = tree['n']; children = tree['children']; blen = tree['blen']; leaf_g = tree['leaf_g']
    cost = np.zeros((n, Kk))
    for lid in range(n):
        cu = emis[leaf_g[lid]].copy() if leaf_g[lid] >= 0 else np.zeros(Kk)
        for c in children[lid]:
            cc = cost[c]; w = lam / max(blen[c], BLFLOOR)
            cu += np.minimum(cc, cc.min() + w)
        cost[lid] = cu
    return float(cost[tree['root']].min())

def objective(X, forest, mu, lam, Kk):   # each approach's OWN objective: WCSS (lam=0) or full J
    emis = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(2)
    if lam == 0:
        return float(emis.min(1).sum())
    return float(sum(sankoff_cost(t, emis, lam, Kk) for t in forest))

# ---------------------------------------------------------------- driver ----
def simulate_rate_matrix_random(K: int, seed: int | None = None) -> np.ndarray:
    """Generate a random K × K continuous-time Markov chain rate matrix."""
    if not isinstance(K, int) or K <= 0:
        raise ValueError("K must be a positive integer.")

    rng = np.random.default_rng(seed)

    # Sample off-diagonal transition rates uniformly from [0, 1).
    Q = rng.uniform(0.0, 1.0, size=(K, K))
    np.fill_diagonal(Q, 0.0)

    # Set diagonal entries so every row sums to zero.
    np.fill_diagonal(Q, -Q.sum(axis=1))

    return Q

def simulate_rate_matrix_fix(K: int, diagonal_value = 0.5) -> np.ndarray:
    """Generate a K × K continuous-time Markov chain rate matrix with fixed diagonal value."""
    if not isinstance(K, int) or K <= 0:
        raise ValueError("K must be a positive integer.")

    Q = np.full((K, K), 1.0)*diagonal_value/(K-1)

    # Set diagonal entries so every row sums to zero.
    np.fill_diagonal(Q, -diagonal_value)

    return Q


