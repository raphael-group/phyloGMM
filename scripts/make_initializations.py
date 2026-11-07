#!/usr/bin/env python3
"""Generate phyloGMM initializations from lineage trees and observations.

Writes one JSON file per (initialization scheme, K, seed), each usable with
    run_phylo_gmm --k K --init_params_json <file> ...
Every file holds Q (a random rate matrix drawn with the same seed), mu, Sigma
(or sigma_diag with --covariance diag) and pi (cluster proportions).

Schemes (the seven used in the paper; all by default):
  greedykmeans        greedy k-means++ seeding, then Lloyd's k-means on the leaves
  vanillakmeanspp     k-means++ seeding, each leaf assigned to its nearest center
  vanillatreelloyds   tree-regularized Lloyd's (exact Sankoff assignment step) with
                      penalty lambda = 0.1, 0.5, 1.0 (three schemes)
  transition_cluster  centroids from sister-cell (cherry) pairs
  treegroup           centroids from minimal clades of the trees
Means / covariances / pi are computed from the original observations under each
scheme's hard labels, plus a ridge of 1e-6 on the covariance diagonal.

Inputs: trees and observation files in pairs (Newick with branch lengths and leaf
names; CSV with the cell name in the first column and the d observation columns
after it), given either explicitly or as a directory of tree_<tag>.nwk / obs_<tag>.csv.

Examples:
  python scripts/make_initializations.py --k 10 --seeds 20 --out inits \\
      --tree t1.nwk --obs o1.csv --tree t2.nwk --obs o2.csv
  python scripts/make_initializations.py --k 8 10 12 --inputs data/ --out inits --covariance diag
  python scripts/make_initializations.py --k 12 --random-q-only --out inits   # Q only, for --fix_labels runs

Then fit from every initialization and keep the best (scripts/select_best_fit.py).
Requires numpy, scipy, pandas and scikit-learn.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forest_lloyds  # noqa: E402
import transition_cluster  # noqa: E402
import tree_lloyds_init  # noqa: E402
import treegroup_init  # noqa: E402

SCHEMES = ["greedykmeans", "vanillakmeanspp", "vanillatreelloyds", "transition_cluster", "treegroup"]
TREE_LLOYD_LAMBDAS = [0.1, 0.5, 1.0]
RIDGE = 1e-6


def pair_inputs(args):
    if args.inputs:
        trees = sorted(glob.glob(os.path.join(args.inputs, "tree_*.nwk")))
        obs = [os.path.join(args.inputs, "obs_%s.csv" % os.path.basename(t)[5:-4]) for t in trees]
        missing = [o for o in obs if not os.path.exists(o)]
        if not trees or missing:
            sys.exit("--inputs %s: need tree_<tag>.nwk with a matching obs_<tag>.csv (missing: %s)"
                     % (args.inputs, missing[:3]))
    else:
        trees, obs = args.tree or [], args.obs or []
        if not trees or len(trees) != len(obs):
            sys.exit("give the same number of --tree and --obs files (or use --inputs DIR)")
    return trees, obs


class Writer:
    def __init__(self, out, covariance, d):
        self.out, self.cov, self.d, self.n = out, covariance, d, 0
        os.makedirs(out, exist_ok=True)

    def __call__(self, name, Q, mu, cov, pi):
        mu = np.asarray(mu, float)
        cov = np.asarray(cov, float)
        rec = {"Q": np.asarray(Q).tolist(), "mu": mu.tolist(), "pi": np.asarray(pi, float).tolist()}
        if self.cov == "diag":
            var = cov if cov.ndim == 2 else np.diagonal(cov, axis1=1, axis2=2)
            rec["sigma_diag"] = (var + RIDGE).tolist()
        else:
            full = cov if cov.ndim == 3 else cov[:, :, None] * np.eye(self.d)
            rec["Sigma"] = (full + RIDGE * np.eye(self.d)[None]).tolist()
        with open(os.path.join(self.out, name + ".json"), "w") as f:
            json.dump(rec, f)
        self.n += 1


def proportions(lab, K):
    return np.bincount(lab, minlength=K).astype(float) / len(lab)


def label_params(X, lab, K, cov):
    return forest_lloyds.params(X, lab, K, covariance_type=cov)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", action="append", help="Newick tree (repeat; paired with --obs)")
    ap.add_argument("--obs", action="append", help="observation CSV for the preceding tree (repeat)")
    ap.add_argument("--inputs", help="directory of tree_<tag>.nwk / obs_<tag>.csv pairs (instead of --tree/--obs)")
    ap.add_argument("--k", type=int, nargs="+", required=True, help="number(s) of cell types")
    ap.add_argument("--seeds", type=int, default=20, help="seeds 0..N-1 per scheme [default: 20]")
    ap.add_argument("--schemes", nargs="+", default=SCHEMES, choices=SCHEMES,
                    help="initialization schemes [default: all]")
    ap.add_argument("--covariance", choices=["full", "diag"], default="full",
                    help="write full Sigma or diagonal sigma_diag; match run_phylo_gmm's --diag_covariance [default: full]")
    ap.add_argument("--random-q-only", action="store_true",
                    help="write only random Q matrices (Q_random_K<k>_seed<s>.json), e.g. for --fix_labels fits")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    seeds = range(args.seeds)

    if args.random_q_only:
        os.makedirs(args.out, exist_ok=True)
        for K in args.k:
            for s in seeds:
                with open(os.path.join(args.out, "Q_random_K%d_seed%d.json" % (K, s)), "w") as f:
                    json.dump({"Q": forest_lloyds.simulate_rate_matrix_random(K, s).tolist()}, f)
        print("wrote %d random Q files to %s" % (len(args.k) * args.seeds, args.out))
        return

    trees, obs = pair_inputs(args)
    names, forest, X = tree_lloyds_init.build_forest(trees, obs)
    n, d = X.shape
    print("%d trees, %d cells, d = %d" % (len(trees), n, d), flush=True)
    write = Writer(args.out, args.covariance, d)
    cov = args.covariance
    cherries = transition_cluster.cherry_pairs(forest, X) if "transition_cluster" in args.schemes else None
    tg = treegroup_init.load_forest(trees, obs) if "treegroup" in args.schemes else None

    for K in args.k:
        if K > n:
            sys.exit("K = %d exceeds the number of cells (%d)" % (K, n))
        for scheme in args.schemes:
            t0 = time.time()
            for s in seeds:
                Q = forest_lloyds.simulate_rate_matrix_random(K, s)
                if scheme == "greedykmeans":
                    mu0 = forest_lloyds.greedy_kmeanspp_seed(X, K, np.random.default_rng(s))
                    lab = forest_lloyds.run_lloyds(X, forest, mu0, 0, K, max_iter=200)
                    mu, S = label_params(X, lab, K, cov)
                    write("%scov_Q_random_greedykmeans_K%d_seed%d" % (cov, K, s), Q, mu, S, proportions(lab, K))
                elif scheme == "vanillakmeanspp":
                    mu0 = forest_lloyds.kmeanspp_seed(X, K, np.random.default_rng(s))
                    lab = ((X[:, None, :] - mu0[None]) ** 2).sum(2).argmin(1)
                    mu, S = label_params(X, lab, K, cov)
                    write("%scov_Q_random_vanillakmeanspp_K%d_seed%d" % (cov, K, s), Q, mu, S, proportions(lab, K))
                elif scheme == "vanillatreelloyds":
                    mu0 = forest_lloyds.kmeanspp_seed(X, K, np.random.default_rng(s))
                    for lam in TREE_LLOYD_LAMBDAS:
                        lab = forest_lloyds.run_lloyds(X, forest, mu0, lam, K, max_iter=200)
                        mu, S = label_params(X, lab, K, cov)
                        write("%scov_Q_random_vanillatreelloyds_lam%s_K%d_seed%d" % (cov, lam, K, s),
                              Q, mu, S, proportions(lab, K))
                elif scheme == "transition_cluster":
                    A, B = cherries
                    if len(A) == 0:
                        sys.exit("transition_cluster needs at least one cherry (two sister leaves)")
                    mu0, _ = transition_cluster.transition_cluster(A, B, X, K, seed=s)
                    e = transition_cluster.initialization_parameters(X, mu0)
                    write("%scov_Q_random_transition_cluster_K%d_seed%d" % (cov, K, s),
                          Q, e["mu"], np.array(e["sigma_diag"]), e["pi"])
                elif scheme == "treegroup":
                    Xt, tg_trees, offsets = tg
                    mu, _ = treegroup_init.tree_group_init(Xt, tg_trees, K, offsets=offsets, return_details=True)
                    lab = ((Xt[:, None, :] - mu[None]) ** 2).sum(2).argmin(1)
                    _, S = label_params(Xt, lab, K, cov)
                    write("%scov_Q_random_treegroup_K%d_seed%d" % (cov, K, s), Q, mu, S, proportions(lab, K))
            print("K = %d  %-18s %4.1f s" % (K, scheme, time.time() - t0), flush=True)
    print("wrote %d initialization files to %s" % (write.n, args.out))


if __name__ == "__main__":
    main()
