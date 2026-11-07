#!/usr/bin/env python
"""Tree-group: centroid initialisation for a phylo-GMM.

Given a forest of trees whose leaves carry d-dimensional observations, propose K
candidate centroids. Uses no labels and has no tuned parameters.

    1. Cut each tree into MINIMAL CLADES of at least d leaves. Walking up from
       the leaves, a clade qualifies the moment it holds >= d leaves while each
       of its children holds fewer. Average each clade's leaves.

       Why d: the mean of n leaves from one cluster sits sigma*sqrt(d/n) from
       that cluster's true centroid, so n = d puts it at exactly one standard
       deviation, at every dimension. Smaller clades are purer but noisier;
       larger ones are quieter but likelier to span a state change.

    2. MERGE clades whose means differ by no more than sampling noise,
       anywhere in the forest:

           ||m_a - m_b||^2  <=  sigma2_hat * d * (1/n_a + 1/n_b)

       since E||m_a - m_b||^2 = ||mu_a - mu_b||^2 + sigma^2 d (1/n_a + 1/n_b).
       Size-weighted average of each merged set. This lets a state that is thin
       in every individual tree accumulate enough leaves to be estimated.

    3. SELECT K by greedy farthest-point: start from the candidate farthest
       from the global mean, then repeatedly take the candidate whose distance
       to its nearest already-chosen candidate is largest.

sigma2_hat is estimated from sister pairs: the median over cherries of
||x_a - x_b||^2 / (2d), divided by median(chi2_d)/d to remove the median bias.
Sister leaves share a latent state ~96% of the time, so the median is robust to
the minority that straddle a transition.

Usage
-----
    from treegroup import tree_group_init
    mu = tree_group_init(X, trees, K)          # (K, d)

or from the command line:

    python treegroup.py --k 10 \
        --tree t5.nwk --obs obs_t5.csv \
        --tree t6.nwk --obs obs_t6.csv \
        --out init.json
"""
from __future__ import annotations
import numpy as np
from scipy.stats import chi2

__all__ = ["tree_group_init", "sigma2_from_sisters", "minimal_clades",
           "merge_lookalikes", "farthest_point"]


# --------------------------------------------------------------------------
# tree representation
# --------------------------------------------------------------------------
class Tree:
    """A rooted tree with leaves in DFS order.

    kids  : list[list[int]]  children of each node
    lo/hi : int arrays, the half-open DFS leaf interval [lo, hi) of each node,
            so the leaves under node u are rows lo[u]..hi[u]-1 of that tree's
            observation block.
    """

    def __init__(self, kids, lo, hi, root):
        self.kids, self.lo, self.hi, self.root = kids, lo, hi, root

    @property
    def n_leaves(self):
        return self.hi[self.root] - self.lo[self.root]

    def cherries(self):
        """Yield (leaf_pos_a, leaf_pos_b) for every node whose two children are
        both leaves."""
        for u, ch in enumerate(self.kids):
            if len(ch) == 2 and not self.kids[ch[0]] and not self.kids[ch[1]]:
                yield self.lo[ch[0]], self.lo[ch[1]]


# --------------------------------------------------------------------------
# step 0 -- sigma^2 from sister pairs
# --------------------------------------------------------------------------
def sigma2_from_sisters(X, trees, offsets, d):
    """Median over cherries of ||x_a - x_b||^2 / (2d), de-biased.

    If both sisters are in state s then x_a - x_b ~ N(0, 2 sigma^2 I_d), so the
    contrast is sigma^2 * chi2_d / d and its median is sigma^2 * median(chi2_d)/d.
    Dividing by that factor makes the estimator unbiased at the median.
    """
    v = []
    for t, off in zip(trees, offsets):
        for a, b in t.cherries():
            v.append(((X[off + a] - X[off + b]) ** 2).sum() / (2.0 * d))
    if not v:
        raise ValueError("no cherries found; cannot estimate sigma^2")
    return float(np.median(v)) / (chi2.ppf(0.5, d) / d)


# --------------------------------------------------------------------------
# step 1 -- minimal clades of at least d leaves
# --------------------------------------------------------------------------
def minimal_clades(X, trees, offsets, d, min_size=None):
    """Return (means, sizes, spans) for the minimal clades with >= min_size leaves.

    A clade qualifies when it has >= min_size leaves and every child has fewer.
    Such clades never nest, so they do not overlap. They do not cover every leaf:
    a small branch joining a large one may never accumulate min_size leaves of
    its own (~60-65% coverage in our data). That is fine -- the goal is good
    candidates, not a partition.
    """
    m = int(max(min_size if min_size is not None else d, 2))
    means, sizes, spans = [], [], []
    for t, off in zip(trees, offsets):
        nl = t.n_leaves
        cs = np.cumsum(np.vstack([np.zeros(X.shape[1]), X[off:off + nl]]), axis=0)
        for u in range(len(t.kids)):
            a, b = t.lo[u], t.hi[u]
            if b - a < m:
                continue
            if all((t.hi[c] - t.lo[c]) < m for c in t.kids[u]):
                means.append((cs[b] - cs[a]) / (b - a))
                sizes.append(b - a)
                spans.append((off + a, off + b))
    if not means:
        raise ValueError(f"no clade reached {m} leaves; is d larger than the trees?")
    return np.asarray(means), np.asarray(sizes, float), spans


# --------------------------------------------------------------------------
# step 2 -- merge clades that differ by no more than noise
# --------------------------------------------------------------------------
def merge_lookalikes(means, sizes, sigma2, d):
    """Size-weighted average of every clade statistically indistinguishable from
    each clade. Returns (pooled_means, pooled_counts, membership_matrix)."""
    g = np.einsum("ij,ij->i", means, means)
    D2 = g[:, None] + g[None, :] - 2.0 * means @ means.T
    thr = sigma2 * d * (1.0 / sizes[:, None] + 1.0 / sizes[None, :])
    A = D2 <= thr
    np.fill_diagonal(A, True)                      # a clade always merges with itself
    W = A * sizes[None, :]
    counts = W.sum(1)
    pooled = (W @ means) / np.maximum(counts, 1e-12)[:, None]
    return pooled, counts, A


# --------------------------------------------------------------------------
# step 3 -- greedy farthest-point
# --------------------------------------------------------------------------
def farthest_point(P, K):
    """Indices of K points, starting from the one farthest from the mean of P,
    then greedy max-min."""
    if len(P) < K:
        raise ValueError(f"only {len(P)} candidates for K={K}")
    g = np.einsum("ij,ij->i", P, P)
    gm = P.mean(0)
    sel = [int(np.argmax(g - 2.0 * P @ gm + gm @ gm))]
    best = None
    for _ in range(K - 1):
        j = sel[-1]
        dd = g + g[j] - 2.0 * P @ P[j]
        best = dd if best is None else np.minimum(best, dd)
        b = best.copy()
        b[sel] = -np.inf
        sel.append(int(np.argmax(b)))
    return sel


# --------------------------------------------------------------------------
# the method
# --------------------------------------------------------------------------
def tree_group_init(X, trees, K, offsets=None, sigma2=None, min_size=None,
                    return_details=False):
    """K candidate centroids, shape (K, d).

    X       : (n_leaves_total, d) observations, leaves of tree i occupying rows
              offsets[i] .. offsets[i] + trees[i].n_leaves
    trees   : list[Tree]
    K       : number of centroids
    sigma2  : per-gene emission variance. Estimated from sister pairs if None.
    min_size: clade size floor. Defaults to d, which is the derived choice.
    """
    X = np.asarray(X, float)
    d = X.shape[1]
    if offsets is None:
        offsets, o = [], 0
        for t in trees:
            offsets.append(o)
            o += t.n_leaves
    if sigma2 is None:
        sigma2 = sigma2_from_sisters(X, trees, offsets, d)
    means, sizes, spans = minimal_clades(X, trees, offsets, d, min_size)
    pooled, counts, A = merge_lookalikes(means, sizes, sigma2, d)
    sel = farthest_point(pooled, K)
    mu = pooled[sel]
    if not return_details:
        return mu
    return mu, dict(sigma2=sigma2, n_clades=len(sizes), clade_sizes=sizes,
                    n_candidates=len(pooled), selected=sel,
                    leaves_per_candidate=counts[sel],
                    clades_merged=A[sel].sum(1), spans=spans, membership=A)


# --------------------------------------------------------------------------
# newick parsing and I/O -- so the module runs standalone
# --------------------------------------------------------------------------
def parse_newick(text):
    """Parse a newick string into (Tree, leaf_names in DFS order)."""
    s = text.strip().rstrip(";")
    i = 0
    kids, par, lab = [], [], []

    def new(name):
        kids.append([]); par.append(-1); lab.append(name)
        return len(kids) - 1

    def token():
        nonlocal i
        j = i
        while i < len(s) and s[i] not in "(),:":
            i += 1
        name = s[j:i]
        if i < len(s) and s[i] == ":":            # skip branch length
            i += 1
            while i < len(s) and s[i] not in "(),":
                i += 1
        return name

    stack, pend = [], []
    while i < len(s):
        c = s[i]
        if c == "(":
            stack.append([]); i += 1
        elif c == ",":
            i += 1
        elif c == ")":
            i += 1
            ch = stack.pop()
            u = new(token())
            for v in ch:
                kids[u].append(v); par[v] = u
            (stack[-1] if stack else pend).append(u)
        else:
            (stack[-1] if stack else pend).append(new(token()))
    root = pend[0] if pend else len(kids) - 1

    # DFS leaf intervals
    n = len(kids)
    lo = np.zeros(n, int); hi = np.zeros(n, int)
    names, counter, st = [], 0, [(root, False)]
    while st:
        u, done = st.pop()
        if done:
            hi[u] = counter; continue
        lo[u] = counter
        if not kids[u]:
            names.append(lab[u]); counter += 1; hi[u] = counter; continue
        st.append((u, True))
        for v in reversed(kids[u]):
            st.append((v, False))
    return Tree(kids, lo, hi, root), names


def load_forest(tree_paths, obs_paths):
    """Read newick + observation CSVs into (X, trees, offsets).

    Each observation CSV must have the leaf name in its first column and the d
    measurements in the remaining columns. Rows are reordered to the tree's DFS
    leaf order.
    """
    import pandas as pd
    Xs, trees, offsets, off = [], [], [], 0
    for tp, op in zip(tree_paths, obs_paths):
        t, names = parse_newick(open(tp).read())
        obs = pd.read_csv(op)
        table = dict(zip(obs.iloc[:, 0].astype(str),
                         obs.iloc[:, 1:].to_numpy(float)))
        missing = [n for n in names if n not in table]
        if missing:
            raise KeyError(f"{op}: {len(missing)} leaves absent, e.g. {missing[:3]}")
        Xs.append(np.stack([table[n] for n in names]))
        trees.append(t); offsets.append(off); off += t.n_leaves
    return np.concatenate(Xs), trees, offsets


def _main(argv=None):
    import argparse, json
    ap = argparse.ArgumentParser(description="Tree-group centroid initialisation.")
    ap.add_argument("--k", type=int, required=True, help="number of centroids")
    ap.add_argument("--tree", action="append", required=True, help="newick file (repeatable)")
    ap.add_argument("--obs", action="append", required=True, help="observation CSV (repeatable)")
    ap.add_argument("--out", help="write {'mu': [[...]]} here; otherwise print to stdout")
    ap.add_argument("--sigma2", type=float, help="known per-gene variance; estimated if omitted")
    ap.add_argument("--min-size", type=int, help="clade size floor; defaults to d")
    a = ap.parse_args(argv)
    if len(a.tree) != len(a.obs):
        ap.error("--tree and --obs must be given the same number of times")
    X, trees, offsets = load_forest(a.tree, a.obs)
    mu, info = tree_group_init(X, trees, a.k, offsets=offsets, sigma2=a.sigma2,
                               min_size=a.min_size, return_details=True)
    payload = {"mu": mu.tolist(), "sigma2_hat": info["sigma2"],
               "n_clades": info["n_clades"], "n_candidates": info["n_candidates"],
               "leaves_per_centroid": [int(v) for v in info["leaves_per_candidate"]]}
    txt = json.dumps(payload, indent=2)
    if a.out:
        open(a.out, "w").write(txt)
        print(f"{a.k} centroids -> {a.out}  "
              f"(sigma2_hat={info['sigma2']:.4f}, {info['n_clades']} clades, "
              f"{X.shape[0]} leaves, d={X.shape[1]})")
    else:
        print(txt)


if __name__ == "__main__":
    _main()
