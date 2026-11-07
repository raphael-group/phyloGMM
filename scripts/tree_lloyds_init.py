#!/usr/bin/env python
"""Standalone tree-Lloyd's initializer -> phyloGMM init parameters.

Given a FOREST (one or more trees + matching observation tables) and K, minimize
    J(z, mu) = sum_{leaves} ||x - mu_z||^2  +  lambda * sum_{edges} (1/branch_len) * 1[z_u != z_v]
by tree-Lloyd's (exact Sankoff-DP assignment step; mean update), keep the best of
--n_init restarts by the objective J, and write a phyloGMM --init_params_json file.

INPUTS (paired by order; a "forest" = several trees sharing one emission GMM):
  --trees  T1.nwk T2.nwk ...   newick trees; leaf names = cell/leaf ids
  --obs    O1.csv O2.csv ...   obs tables; FIRST column = cell id (matching leaf names),
                               remaining columns = the D emission features (e.g. scVI dims)
  --k K           number of clusters
  --lam 0.5       tree-penalty weight (0 = plain k-means / WCSS only)
  --n_init 10     restarts; keep the best by objective J
  --out init.json phyloGMM init params (mu, sigma_diag, pi)
  --labels-out labels.csv   optional: per-leaf hard cluster labels

OUTPUT:
  init.json : {"mu": [K x D], "sigma_diag": [K x D], "pi": [K]}
              feed to phyloGMM with:  --k K --diag_covariance --init_params_json init.json

Note: mu/sigma/pi are computed from the ORIGINAL observations using the tree-Lloyd's
labels (not smoothed features), so the parameters reflect the real data.
"""
import argparse, json, sys, threading
import numpy as np, pandas as pd
from forest_lloyds import (parse_newick, tree_to_arrays, run_lloyds, kmeanspp_seed,
                           params, objective)

def build_forest(tree_paths, obs_paths):
    roots = [parse_newick(open(t).read()) for t in tree_paths]
    names, name2g = [], {}
    def collect(nd):
        if not nd['ch']:
            if nd['name'] not in name2g:
                name2g[nd['name']] = len(names); names.append(nd['name'])
        for c in nd['ch']: collect(c)
    for r in roots: collect(r)
    forest = [tree_to_arrays(r, name2g) for r in roots]
    obs = pd.concat([pd.read_csv(o, index_col=0) for o in obs_paths])
    missing = [n for n in names if n not in obs.index]
    if missing:
        raise SystemExit(f"{len(missing)} tree leaves have no obs row (e.g. {missing[:3]})")
    X = obs.loc[names].values.astype(float)
    return names, forest, X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", nargs="+", required=True)
    ap.add_argument("--obs", nargs="+", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels-out", default=None)
    a = ap.parse_args()
    if len(a.trees) != len(a.obs):
        raise SystemExit("need exactly one --obs per --tree (paired by order)")

    names, forest, X = build_forest(a.trees, a.obs)
    K = a.k
    best_J, best_lab = None, None
    for s in range(a.n_init):
        mu0 = kmeanspp_seed(X, K, np.random.default_rng(s))
        lab = run_lloyds(X, forest, mu0, a.lam, K)
        Jv = objective(X, forest, params(X, lab, K)[0], a.lam, K)
        if best_J is None or Jv < best_J:
            best_J, best_lab = Jv, lab

    mu, sig = params(X, best_lab, K)
    sig = np.maximum(sig, 1e-6)
    pi = np.bincount(best_lab, minlength=K).astype(float) / len(best_lab)
    json.dump({"mu": mu.tolist(), "sigma_diag": sig.tolist(), "pi": pi.tolist()}, open(a.out, "w"))
    print(f"wrote {a.out}: K={K}, D={X.shape[1]}, leaves={len(names)}, "
          f"trees={len(forest)}, lam={a.lam}, best J={best_J:.1f} over {a.n_init} inits", flush=True)
    if a.labels_out:
        pd.Series(best_lab, index=names, name="cluster").to_csv(a.labels_out)
        print(f"wrote {a.labels_out}", flush=True)

if __name__ == "__main__":
    threading.stack_size(256 * 1024 * 1024); sys.setrecursionlimit(10 ** 7)
    t = threading.Thread(target=main); t.start(); t.join()
