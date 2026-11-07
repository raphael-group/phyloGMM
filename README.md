# phyloGMM: a phylogenetic mixture model for simultaneous inference of cell types and transitions from single-cell lineage tracing data

phyloGMM takes one or more cell lineage trees together with a continuous observation for every cell
(e.g. a low-dimensional expression embedding such as scVI or PCA), and jointly infers

1. **cell types**: a Gaussian mixture with `K` components, one mean and covariance per cell type, and
2. **cell type transitions**: a continuous-time Markov chain (CTMC) with rate matrix `Q` and root
   distribution `pi`, describing how cell types change along the branches of the trees.

Cell types are latent: each cell's type is inferred from its own observation *and* from its relatives
on the tree. Parameters are fit by expectation-maximization (EM) on the exact marginal likelihood.

If you find phyloGMM useful in your research, please cite:

> TODO: citation

The code and data to reproduce the results in the paper are at TODO (experiments repository).

## Installation

phyloGMM is a C++17 command-line program built with CMake (>= 3.14). Eigen and csv-parser are
downloaded automatically during configuration, so the first `cmake` call needs internet access.

```
git clone https://github.com/raphael-group/phyloGMM.git
cd phyloGMM
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target run_phylo_gmm -j 4
```

This produces the executable `build/run_phylo_gmm`. Compiling takes several minutes and up to a few GB of
memory per parallel job (Eigen templates), so lower `-j` if the compiler is killed. To also build and
run the tests (`tests/`; needs about 30 GB with `-j 2`):

```
cmake --build build -j 2
cd build && ctest
```

To check the build, run the example below
(about 1 minute on one CPU, under 40 MB of memory) and compare with the expected output.

## Usage

```
run_phylo_gmm --k K --tree T1 --obs O1 [--tree T2 --obs O2 ...] [options] > result.json
```

Trees and observation files are given in pairs; all trees are fit jointly with shared parameters.
Results are written to standard output as JSON, and progress messages to standard error.

### Input

1. **Lineage trees** (`--tree`, one per tree): Newick with branch lengths and leaf names, e.g.
   `((cellA:0.4,cellB:0.4)n1:0.6,cellC:1.0)root;`. Every leaf must be a cell that appears in the
   matching observation file. Internal node names are optional. A branch length on the root itself
   is ignored: `pi` is the distribution at the root.
2. **Observations** (`--obs`, one per tree): CSV with a header row; the first column is the cell
   name (matching the tree's leaf names) and the remaining `d` columns are the observation, e.g.

   ```
   cell,x0,x1,x2
   cellA,0.12,-1.30,0.88
   cellB,0.10,-1.25,0.91
   cellC,2.31,0.47,-0.06
   ```

   Every leaf must be observed. To hold out cells, prune them from the tree.
3. **Number of cell types** `--k K`.
4. **Initialization** (optional, recommended) `--init_params_json init.json`: starting values for
   any subset of `Q`, `mu`, `Sigma` (or `sigma_diag`) and `pi`:

   ```
   {"Q": [[K x K]], "mu": [[K x d]], "Sigma": [[[d x d], ... K]], "pi": [K]}
   ```

   Without it, means are initialized by k-means++. EM finds a local optimum, so we recommend fitting
   from many initializations and keeping the fit with the highest `log_likelihood` (see
   [Initializations](#initializations)).

### Output

A single JSON object (standard output). The main fields:

| field | shape | meaning |
|---|---|---|
| `log_likelihood` | scalar | marginal log-likelihood of all observations given the trees |
| `Q` | K x K | CTMC rate matrix (off-diagonal rates `q_ij`, rows sum to 0) |
| `pi` | K | distribution of the cell type at the root of each tree |
| `mu` | K x d | mean of each cell type |
| `Sigma` | K x d x d | covariance of each cell type |
| `leaf_names` | n | cell names, in the order used by the per-cell fields |
| `posteriors` | n x K | posterior probability of each cell type for each cell |
| `leaf_labels` | n | most probable cell type of each cell |
| `tree_index_per_leaf` | n | which input tree each cell belongs to (0-based) |
| `internal_node_ids`, `internal_node_posteriors` | m, m x K | the same for internal (ancestral) nodes |
| `expected_transitions` | K x K | expected number of i -> j transitions over all branches, given the data |
| `expected_dwell_times` | K | expected total time spent in each cell type over all branches |
| `per_restart` | list | log-likelihood trace and initial/final parameters of each EM restart |

### Example

`examples/sim1_hematopoiesis/` contains one replicate of the simulated hematopoiesis data from the
paper: 5 lineage trees (10,953 cells), `K = 10` cell types, 10-dimensional observations, with the
true parameters (`data/true_params.json`) and true cell types (`data/true_labels.csv`).

```
bash examples/sim1_hematopoiesis/run_example.sh
```

runs

```
build/run_phylo_gmm --k 10 --pi_mode free --diag_covariance --pade_expm \
    --max_iter 500 --tol 1e-5 --n_init 1 \
    --init_params_json examples/sim1_hematopoiesis/data/init_params.json \
    --tree examples/sim1_hematopoiesis/data/tree_t5.nwk --obs examples/sim1_hematopoiesis/data/obs_t5.csv \
    ... (trees t6 to t9) \
    > examples/sim1_hematopoiesis/output/result.json
```

and compares the fit with the truth (`compare_to_truth.py`, needs numpy and scipy). It takes about 1 minute.
Fitted cell types are matched to the true ones by their means. Expected output:

```
log-likelihood      -33056.6941
ARI                 0.956
mu RMSE             0.0105
Sigma error         0.052
Q error (offdiag)   0.074
pi on true root     1.000

Fitted Q (rows/columns in true-state order):
 [[-0.4   0.1   0.11  0.    0.    0.09  0.    0.    0.    0.1 ]
 [ 0.   -0.28  0.    0.    0.28  0.    0.    0.    0.    0.  ]
 [ 0.    0.   -0.19  0.    0.    0.    0.    0.    0.19  0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.16 -0.47  0.    0.15  0.15  0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]]

True Q:
 [[-0.4   0.1   0.1   0.    0.    0.1   0.    0.    0.    0.1 ]
 [ 0.   -0.3   0.    0.    0.3   0.    0.    0.    0.    0.  ]
 [ 0.    0.   -0.2   0.    0.    0.    0.    0.    0.2   0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.15 -0.45  0.    0.15  0.15  0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.  ]]
```

The full expected result is in `examples/sim1_hematopoiesis/expected_output/result.json`.

### Initializations

`scripts/make_initializations.py` generates initialization files from your trees and observations
(Python 3 with numpy, scipy, pandas and scikit-learn; `pip install -r scripts/requirements.txt`):

```
python scripts/make_initializations.py --k 10 --seeds 20 --out inits \
    --tree tree1.nwk --obs obs1.csv --tree tree2.nwk --obs obs2.csv
```

(or `--inputs DIR` for a directory of `tree_<tag>.nwk` / `obs_<tag>.csv` pairs; several values
after `--k` are allowed). It writes one JSON per scheme and seed, `<cov>_Q_random_<scheme>_K<k>_seed<s>.json`,
with a random `Q` and `mu`, `Sigma` and `pi` from the scheme's clustering. The seven schemes used in
the paper are k-means (`greedykmeans`), k-means++ seeding (`vanillakmeanspp`), tree-regularized
k-means with three penalty weights (`vanillatreelloyds`), clustering of sister-cell pairs
(`transition_cluster`), and clustering of small clades (`treegroup`); `--seeds 20` gives 140 files.
Use `--covariance diag` together with `run_phylo_gmm --diag_covariance`.

Fit from each file and keep the best:

```
for init in inits/*.json; do
    name=$(basename "$init" .json); mkdir -p fits/$name
    build/run_phylo_gmm --k 10 --pade_expm --max_iter 500 --n_init 1 --init_params_json "$init" \
        --tree tree1.nwk --obs obs1.csv --tree tree2.nwk --obs obs2.csv > fits/$name/result.json
done
python scripts/select_best_fit.py fits/*/result.json --copy best_result.json
```

The fits are independent, so on a cluster run them as a job array.

### Options

```
--k K                     number of cell types [required]
--tree FILE --obs FILE    a lineage tree and its observations; repeat for several trees [required]
--init_params_json FILE   starting values for Q, mu, Sigma/sigma_diag, pi
--pi_mode free|stationary root distribution fit freely (default) or set to the stationary distribution of Q
--diag_covariance         diagonal instead of full covariance matrices
--max_iter N              maximum EM iterations [default: 100]
--tol T                   relative convergence tolerance on the log-likelihood [default: 1e-5]
--n_init N                N k-means++ and N random restarts; the best is reported [default: 10]
                          (fields given in --init_params_json are shared by all restarts)
--parallel_restarts [N]   run restarts in parallel on up to N threads
--reg_covar R             ridge added to covariance diagonals [default: 1e-6]
--pade_expm               numerically robust matrix exponential (recommended; used for all results in the paper)
--verbose                 print the log-likelihood at every EM iteration
```

Fixed-parameter modes:

```
--fix_labels FILE         fix each cell's type to the labels in FILE (CSV: cell,state, 0-based)
                          and fit only Q and pi (a CTMC on known cell types)
--fix_q_json FILE         keep Q fixed at the matrix in FILE; fit the cell types and pi
--fix_emission            keep mu and Sigma fixed; fit Q and pi
--bayes_opt --bayes_opt_truth_json FILE
                          no fitting: evaluate the likelihood and posteriors at the parameters in FILE
```

Run `build/run_phylo_gmm` without arguments for the full list.

## Contact

Please open an issue on GitHub for questions or bug reports.

## License

TODO
