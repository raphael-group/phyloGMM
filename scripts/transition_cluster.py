#!/usr/bin/env python
"""transition-cluster: split cherries by a per-pair test, then cluster each part.

Input   forest of trees with observations; number of centroids K.
Output  phyloGMM emission initialization (mu, sigma_diag, pi), or K centroids as CSV.

  sigmahat^2  mean over all cherries of ||a - b||^2 / (2d).
  1. pairs     collect every cherry (a, b).
  2. split     per-pair test ||a - b||^2 vs 2 d sigmahat^2 -> SAME set / SPAN set.
  3. SAME      cluster the midpoints (z-scored per coordinate) into K. Each cluster with
               >= 2 pairs emits one candidate, the mean of its cells, support 2n. A
               same-state midpoint has variance sigma^2/2, better than either cell alone.
  4. SPAN      cluster with the two-centroid co-clustering Lloyd into 2K clusters (the
               default; pass n_span_clusters to override), run with lam = 0 so the two ends
               are unshrunk. A cluster whose ends are separated by more than
               2 d sigmahat^2 / n emits BOTH ends, support n each; one whose ends are not
               separated still emits ONE candidate at the pooled mean, support 2n.
  5. select    pool the candidates and take K by support-weighted k-means, padding by
               farthest-point from the observed cells if fewer than K survive.

The point of the split is that the same-state majority no longer drowns the transitions:
cherry_separability.py measures the test's power as AUC 0.80-0.87 on all three real
datasets but 0.66 on simulation d=30, where the best single threshold (0.953) does not even
beat guessing "same-state" for everything (0.959). So this is expected to work on real data
and not on our simulation.

Dependencies: numpy, pandas and scikit-learn, plus the shared initialization_utils serializer.
The variants this supersedes -- transition_lloyd (co-clustering
alone) and transition_overcluster (co-clustering on ALL pairs, over-clustered then
selected) -- are in archive/. The two routines still needed from them, the co-clustering
Lloyd and the support-weighted selection, are inlined below.

    python3 scripts/transition_cluster.py --dir path/to/stage --K 11 --out init.json
    python3 scripts/transition_cluster.py --trees tree_t5.nwk tree_t6.nwk \
        --obs obs_t5.csv obs_t6.csv --k 11 --seed 0 --out init.json

JSON uses the same format as initializations.py: mu and sigma_diag have shape
(K, D), and pi has shape (K,). The transition centroids are retained as mu.
Nearest-centroid assignments of all observations supply population variances
(floored at 1e-6) and mixture weights. Q is initialized separately. An --out path
ending in .csv retains the original centroid-only output.
"""
from __future__ import annotations
import glob
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

if __package__:
    from .initialization_utils import initialization_parameters
else:
    from initialization_utils import initialization_parameters


# --------------------------------------------------------------- inlined from v2/v2.5
def cherry_pairs(forest, X):
    """Extract pairs from the forest dictionaries built by tree_lloyds_init.

    leaf_g already indexes the pooled observation matrix, including offsets
    across trees. Only binary nodes with two observed leaf children qualify.
    """
    pairs = []
    for tree in forest:
        children, leaf_g = tree["children"], tree["leaf_g"]
        for ch in children:
            if (len(ch) == 2 and all(not children[c] for c in ch)
                    and all(leaf_g[c] >= 0 for c in ch)):
                pairs.append([leaf_g[ch[0]], leaf_g[ch[1]]])
    indices = np.asarray(pairs, dtype=int).reshape(-1, 2)
    return X[indices[:, 0]], X[indices[:, 1]]


def cherry_pairs_sim(F, X):
    A, B = [], []
    for T, o in zip(F.trees, F.offs):
        for u, ch in enumerate(T.kids):
            if len(ch) == 2 and not T.kids[ch[0]] and not T.kids[ch[1]]:
                A.append(X[o + T.lo[ch[0]]]); B.append(X[o + T.lo[ch[1]]])
    return np.array(A), np.array(B)

def transition_lloyd(A, B, K, lam, n_iter=50, seed=0, tol=1e-7):
    """-> (muA, muB, labels, orient). Two centroids per cluster; orientation latent."""
    N, d = A.shape
    mid = (A + B) / 2
    # initialise: k-means on midpoints for the partition, then split each cluster's ends
    # along its own dominant displacement direction, which breaks the A/B symmetry.
    z = (mid - mid.mean(0)) / np.maximum(mid.std(0), 1e-9)
    lab = KMeans(K, n_init=4, random_state=seed).fit(z).labels_
    P, Q = A.copy(), B.copy()
    for k in range(K):
        m = lab == k
        if m.sum() < 2: continue
        D = A[m] - B[m]
        v = np.linalg.eigh(D.T @ D)[1][:, -1]
        flip = (D @ v) < 0
        idx = np.flatnonzero(m)[flip]
        P[idx], Q[idx] = B[idx], A[idx]
    muA = np.stack([P[lab == k].mean(0) if (lab == k).any() else mid.mean(0) for k in range(K)])
    muB = np.stack([Q[lab == k].mean(0) if (lab == k).any() else mid.mean(0) for k in range(K)])

    prev = np.inf
    for _ in range(n_iter):
        # ---- E-step: cluster and orientation jointly ----
        # cost(i,k) = min over orientation, plus the lambda penalty this pair would pay
        cAA = ((A[:, None, :] - muA[None]) ** 2).sum(-1)      # a -> A end
        cBB = ((B[:, None, :] - muB[None]) ** 2).sum(-1)      # b -> B end
        cAB = ((A[:, None, :] - muB[None]) ** 2).sum(-1)      # a -> B end
        cBA = ((B[:, None, :] - muA[None]) ** 2).sum(-1)      # b -> A end
        c0, c1 = cAA + cBB, cBA + cAB
        pen = 0.0 if not np.isfinite(lam) else lam * ((muA - muB) ** 2).sum(1)[None, :]
        cost = np.minimum(c0, c1) + pen
        lab = cost.argmin(1)
        orient = (c1 < c0)[np.arange(N), lab]                 # True = swap a and b
        P = np.where(orient[:, None], B, A)
        Q = np.where(orient[:, None], A, B)

        # ---- M-step: closed form of the 2x2 system ----
        for k in range(K):
            m = lab == k
            if not m.any():
                continue
            abar, bbar = P[m].mean(0), Q[m].mean(0)
            if not np.isfinite(lam):
                muA[k] = muB[k] = (abar + bbar) / 2            # lambda -> infinity
            else:
                muA[k] = ((1 + lam) * abar + lam * bbar) / (1 + 2 * lam)
                muB[k] = (lam * abar + (1 + lam) * bbar) / (1 + 2 * lam)
        J = float(cost[np.arange(N), lab].sum())
        if np.isfinite(prev) and abs(prev - J) <= tol * max(abs(prev), 1.0):
            break
        prev = J
    return muA, muB, lab, orient


def select(cand, sup, K, s2, d, X, rule, seed=0):
    """Trim to K.

    support   top-K by support.
    dedup     greedy by support, refusing a candidate that fails the look-alike test
              ||c_i - c_j||^2 <= s2 d (1/n_i + 1/n_j) against one already kept. MEASURED
              INERT: on mela at 4K clusters it rejects 1 of 496 pairs, because with
              supports in the hundreds the threshold (median 1.34 sd) sits an order of
              magnitude below the actual candidate spacing (median 10.43 sd). It asks
              "are these the same point", not "are these far enough apart".
    kmeans    k-means on the candidates themselves, weighted by support, into K groups;
              keep the highest-support candidate of each group. Enforces spread at the
              scale the candidates actually occupy rather than at the noise scale.
    """
    order = np.argsort(-sup)
    if rule == "support":
        keep = list(order[:K])
    else:
        keep = []
        for i in order:
            if len(keep) >= K:
                break
            if any(((cand[i] - cand[j]) ** 2).sum() <= s2 * d * (1 / sup[i] + 1 / sup[j])
                   for j in keep):
                continue
            keep.append(i)
    if rule == "kmeans":
        if len(cand) <= K:
            keep = list(range(len(cand)))
        else:
            g = KMeans(K, n_init=4, random_state=seed).fit(cand, sample_weight=sup).labels_
            keep = [int(np.flatnonzero(g == c)[np.argmax(sup[g == c])])
                    for c in range(K) if (g == c).any()]
    C = cand[keep] if len(keep) else np.empty((0, d))
    while len(C) < K:                                                  # pad by farthest point
        base = C if len(C) else X[:1]
        g = np.min(((X[:, None, :] - base[None]) ** 2).sum(2), axis=1)
        C = np.vstack([C, X[int(g.argmax())]]) if len(C) else X[int(g.argmax())][None]
    return C


# ------------------------------------------------------------------ the method
def transition_cluster(A, B, X, K, n_span_clusters=None, seed=0):
    """-> (K centroids, diagnostics dict)."""
    A, B, X = (np.asarray(values, dtype=float) for values in (A, B, X))
    if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty, finite observation matrix")
    if (A.ndim != 2 or A.shape != B.shape or A.shape[1] != X.shape[1]
            or not np.isfinite(A).all() or not np.isfinite(B).all()):
        raise ValueError("A and B must be finite cherry matrices with the same feature dimension as X")
    if not len(A):
        raise ValueError("no cherries found (internal nodes with two observed leaf children)")
    if not isinstance(K, (int, np.integer)) or not 1 <= K <= len(X):
        raise ValueError("K must be an integer between 1 and the number of observations")
    if n_span_clusters is not None and (
            not isinstance(n_span_clusters, (int, np.integer)) or n_span_clusters < 1):
        raise ValueError("n_span_clusters must be a positive integer")
    N, d = A.shape
    s2 = float((((A - B) ** 2).sum(1)).mean() / (2 * d))       # cherry-contrast sigma^2
    dist2 = ((A - B) ** 2).sum(1)
    same = dist2 <= 2 * d * s2                                  # the per-pair test
    cand, sup = [], []

    # --- SAME set: midpoints -> K state centroids ---
    if same.sum() >= K:
        mid = (A[same] + B[same]) / 2
        z = (mid - mid.mean(0)) / np.maximum(mid.std(0), 1e-9)
        km = KMeans(K, n_init=4, random_state=seed).fit(z)
        for c in range(K):
            m = km.labels_ == c
            if m.sum() >= 2:
                cand.append(mid[m].mean(0)); sup.append(2 * int(m.sum()))

    # --- SPAN set: two-centroid Lloyd -> transition ends ---
    Ms = n_span_clusters if n_span_clusters is not None else 2 * K
    n_trans = 0
    if (~same).sum() >= 2 * Ms:
        As, Bs = A[~same], B[~same]
        muA, muB, lab, orient = transition_lloyd(As, Bs, Ms, lam=0.0, seed=seed)
        P = np.where(orient[:, None], Bs, As)
        Q = np.where(orient[:, None], As, Bs)
        for k in range(Ms):
            m = lab == k
            n = int(m.sum())
            if n < 2:
                continue
            if ((muA[k] - muB[k]) ** 2).sum() > 2 * d * s2 / n:  # a real transition
                n_trans += 1
                cand.append(P[m].mean(0)); sup.append(n)
                cand.append(Q[m].mean(0)); sup.append(n)
            else:
                cand.append(np.vstack([P[m], Q[m]]).mean(0)); sup.append(2 * n)

    cand, sup = np.asarray(cand).reshape(-1, d), np.asarray(sup)
    # The documented farthest-point padding also covers small forests where
    # neither half has enough pairs to emit a supported candidate.
    C = select(cand, sup, K, s2, d, X, "kmeans", seed=seed)
    return C, dict(n_same=int(same.sum()), n_span=int((~same).sum()),
                   pct_flagged_same=round(100 * float(same.mean()), 1),
                   n_candidates=len(cand), n_transition_clusters=n_trans)


# ----------------------------------------------------------------------------
# Command line: tree_<name>.nwk / obs_<name>.csv -> emission JSON or centroids CSV
#
#   python transition_cluster.py --dir path/to/stage --K 11 --out init_means.csv
#
# obs CSV: first column the cell name, the remaining columns the coordinates.
# Same input convention as phylo_moment_init.py.
# ----------------------------------------------------------------------------
def parse_newick(s):
    s = s.strip().rstrip(";"); names, parent = [], []
    def new(p):
        names.append(None); parent.append(p); return len(names) - 1
    cur = new(-1); stack = []; i = 0
    while i < len(s):
        c = s[i]
        if c == "(":
            stack.append(cur); cur = new(cur); i += 1
        elif c == ",":
            cur = new(stack[-1]); i += 1
        elif c == ")":
            cur = stack.pop(); i += 1
        else:
            j = i
            while j < len(s) and s[j] not in "(),": j += 1
            tok = s[i:j].strip()
            names[cur] = (tok.rsplit(":", 1)[0] if ":" in tok else tok).strip()
            i = j
    return names, np.array(parent)


def load_cherries(directory=None, tree_paths=None, obs_paths=None):
    """-> (A, B, X, cells, n_trees) from a directory or explicit paired files.

    A cherry has exactly two observed leaf children. Observation rows are
    aligned within each tree, so identifiers may repeat across different trees.
    X contains only tree leaves, as in tree_lloyds_init.build_forest.
    """
    if directory is not None:
        if tree_paths is not None or obs_paths is not None:
            raise ValueError("use either a directory or explicit tree/observation paths")
        tree_paths = sorted(glob.glob(os.path.join(directory, "tree_*.nwk")))
        if not tree_paths:
            raise ValueError(f"no tree_*.nwk in {directory}")
        obs_paths = [os.path.join(directory, f"obs_{os.path.basename(t)[5:-4]}.csv")
                     for t in tree_paths]
    if not tree_paths or not obs_paths or len(tree_paths) != len(obs_paths):
        raise ValueError("need exactly one observation file per tree (paired by order)")
    Xs, cells, A_idx, B_idx, off = [], [], [], [], 0
    features = None
    for t, obs_path in zip(tree_paths, obs_paths):
        with open(t) as handle:
            names, par = parse_newick(handle.read())
        kids = [[] for _ in names]
        for v, p in enumerate(par):
            if p >= 0:
                kids[p].append(v)
        leaves = [names[v] for v, ch in enumerate(kids) if not ch]
        if any(not name for name in leaves) or len(set(leaves)) != len(leaves):
            raise ValueError(f"{t}: tree leaf names must be nonempty and unique within a tree")
        obs = pd.read_csv(obs_path, dtype=str)
        if obs.shape[1] < 2:
            raise ValueError(f"{obs_path}: expected a cell identifier and at least one feature column")
        columns = list(obs.columns[1:])
        if features is None:
            features = columns
        elif columns != features:
            raise ValueError(f"{obs_path}: feature columns must match across observation files")
        ids = obs.iloc[:, 0]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"{obs_path}: observation identifiers must be nonempty and unique")
        obs.index = ids
        missing = [name for name in leaves if name not in obs.index]
        if missing:
            raise ValueError(f"{obs_path}: tree leaves have no observation row (e.g. {missing[:3]})")
        values = obs.loc[leaves].iloc[:, 1:].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{obs_path}: observations must be finite")
        Xs.append(values)
        row = {name: off + i for i, name in enumerate(leaves)}
        cells.extend(leaves)
        for ch in kids:
            if len(ch) != 2:
                continue
            if all(not kids[c] for c in ch):
                A_idx.append(row[names[ch[0]]]); B_idx.append(row[names[ch[1]]])
        off += len(leaves)
    X = np.vstack(Xs)
    if not A_idx:
        raise ValueError("no cherries found (internal nodes with two observed leaf children)")
    return X[np.array(A_idx)], X[np.array(B_idx)], X, cells, len(tree_paths)


def _main(argv=None):
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Transition-cluster phyloGMM emission initialisation.")
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dir", help="directory containing tree_*.nwk and matching obs_*.csv")
    inputs.add_argument("--trees", nargs="+", help="Newick trees paired by order with --obs")
    ap.add_argument("--obs", nargs="+", help="observation CSVs paired by order with --trees")
    ap.add_argument("--K", "--k", dest="K", type=int, required=True)
    ap.add_argument("--span-clusters", type=int, default=None,
                    help="clusters for the SPAN half (default 2K)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="init.json",
                    help="emission JSON path (default: init.json); use .csv for centroids only")
    a = ap.parse_args(argv)
    if a.dir and a.obs:
        ap.error("--obs can only be used with --trees")
    if a.trees and (not a.obs or len(a.trees) != len(a.obs)):
        ap.error("need exactly one --obs per --tree (paired by order)")
    try:
        A, B, X, cells, n_trees = load_cherries(a.dir, a.trees, a.obs)
        C, info = transition_cluster(A, B, X, a.K, n_span_clusters=a.span_clusters, seed=a.seed)
        payload = initialization_parameters(X, C)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        pd.DataFrame(C, columns=[f"x{j}" for j in range(X.shape[1])]).to_csv(out, index_label="state")
    else:
        with out.open("w") as handle:
            json.dump(payload, handle, allow_nan=False)
            handle.write("\n")
    summary = dict(trees=n_trees, cells=len(X), cherries=len(A), K=a.K,
                   span_clusters=a.span_clusters or 2 * a.K, seed=a.seed, **info)
    with out.with_name(out.stem + "_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=1, allow_nan=False)
    print(json.dumps(summary, indent=1))
    print(f"wrote {a.out}  ({len(C)} centroids x {X.shape[1]} dims)")


if __name__ == "__main__":
    _main()
