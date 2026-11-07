#!/usr/bin/env bash
# Minimal phyloGMM example: 5 simulated lineage trees (10,953 cells),
# 10 cell states, 10-dimensional observations, diagonal covariance.
# Usage (from the repository root, after building):
#   bash examples/sim1_hematopoiesis/run_example.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${PHYLOGMM_BIN:-$HERE/../../build/run_phylo_gmm}"
D="$HERE/data"
OUT="$HERE/output"
mkdir -p "$OUT"

args=()
for t in 5 6 7 8 9; do
    args+=(--tree "$D/tree_t$t.nwk" --obs "$D/obs_t$t.csv")
done

"$BIN" --k 10 --pi_mode free --diag_covariance --pade_expm \
       --max_iter 500 --tol 1e-5 --n_init 1 \
       --init_params_json "$D/init_params.json" \
       "${args[@]}" \
       > "$OUT/result.json" 2> "$OUT/run.log"

echo "wrote $OUT/result.json"
PY="${PYTHON:-python3}"
if "$PY" -c "import numpy, scipy" 2>/dev/null; then
    "$PY" "$HERE/compare_to_truth.py" "$OUT/result.json" "$D"
else
    echo "numpy/scipy not found: skipping the comparison (set PYTHON=/path/to/python to use another interpreter)"
fi
