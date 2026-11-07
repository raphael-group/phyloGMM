#!/usr/bin/env python3
"""Pick the best of several phyloGMM fits (highest log_likelihood).

Usage:
    python scripts/select_best_fit.py fits/*/result.json [--copy best_result.json]

Prints every fit's log-likelihood (best first) and the best file; --copy also copies it.
EM finds a local optimum, so fit from many initializations (scripts/make_initializations.py)
and keep the fit with the highest log-likelihood.
"""
import argparse
import re
import shutil

LL = re.compile(r'"log_likelihood":\s*(-?[0-9.eE+-]+|-?inf|nan)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="result.json files written by run_phylo_gmm")
    ap.add_argument("--copy", help="copy the best result.json to this path")
    ap.add_argument("--top", type=int, default=10, help="how many fits to list [default: 10]")
    args = ap.parse_args()
    fits = []
    for p in args.results:
        with open(p) as f:
            m = LL.search(f.read(4096))  # log_likelihood is the first field
        if m:
            fits.append((float(m.group(1)), p))
        else:
            print("skipping %s (no log_likelihood; failed or unfinished run?)" % p)
    if not fits:
        raise SystemExit("no finished fits")
    fits.sort(key=lambda x: -x[0] if x[0] == x[0] else float("inf"))
    for ll, p in fits[:args.top]:
        print("%14.4f  %s" % (ll, p))
    print("best: %s (log-likelihood %.4f, %d fits)" % (fits[0][1], fits[0][0], len(fits)))
    if args.copy:
        shutil.copy(fits[0][1], args.copy)


if __name__ == "__main__":
    main()
