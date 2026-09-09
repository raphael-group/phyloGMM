// test_ctmc_permutation_invariance: CTest unit test asserting that ctmc_model's
// branch transition probabilities do not depend on how the latent states happen
// to be numbered.
//
// Why this test exists
// -------------------
// The state labels 1..k are arbitrary. Relabelling them -- permuting the rows of
// mu/Sigma/pi and BOTH the rows and columns of Q -- describes the identical
// model, so every likelihood must be unchanged. In practice it was not.
//
// The cause was the eigendecomposition route for exp(tQ):
//
//     P_ab(t) = sum_p U[a,p] * exp(lambda_p * t) * U^-1[p,b]
//
// U and U^-1 are dense. When Q is REDUCIBLE -- state b unreachable from state a,
// so P_ab(t) is exactly zero -- that zero has to be rebuilt by cancellation
// between ten non-zero terms. It doesn't cancel cleanly; it lands at +/-1e-15.
// The log-space guard downstream then reads
//
//     (res > 0.0) ? log(res) : NEGATIVE_INFINITY
//
// and maps +1e-15 to log(1e-15) ~ -35 ("possible") but -1e-15 to -inf
// ("impossible"). Which side the residue falls on depends on the ordering of the
// eigenvector basis LAPACK returns, which depends on the ordering of Q. A
// reducible generator also has a repeated eigenvalue (0, with multiplicity equal
// to the number of communicating classes), so that basis is NOT UNIQUE -- there
// is a whole eigenspace of valid answers and the solver returns whichever its
// arithmetic reached. Occasionally it returns a near-degenerate one, U becomes
// numerically singular, and the whole matrix is wrong rather than just its zeros.
//
// The fix is ctmc_model::use_pade_expm, which computes exp(tQ) by scaling-and-
// squaring. That never forms U or U^-1, so structurally-zero entries stay
// exactly 0.0 and the guard is deterministic.
//
// THE IMPORTANT PART OF THIS TEST IS THE FIXTURE, NOT THE ASSERTION.
// A dense random Q passes on the buggy path too: with distinct eigenvalues the
// eigenvectors are unique and every ordering agrees to the last bit. The bug is
// only reachable with a REDUCIBLE Q. The fixture below is a differentiation-tree
// generator -- a rooted 9-edge DAG over 10 states, exactly the shape a cell-fate
// map produces -- with 81 of 90 off-diagonals exactly zero. Do not "simplify"
// this fixture to a random matrix; that silently disarms the test.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "state_transition_models/ctmc.h"

namespace {

constexpr std::size_t K = 10;

// Rooted differentiation tree over 10 states (9 directed edges):
//   0 -> 1,2,5,9    1 -> 4    2 -> 8    4 -> 3,6,7
// Reducible: 81 of the 90 off-diagonal entries are exactly zero, and exp(tQ)
// inherits that sparsity because no reverse or cross-branch path exists.
Eigen::MatrixXd differentiation_tree_Q() {
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(K, K);
    const std::vector<std::pair<int, int>> edges = {
        {0,1},{0,2},{0,5},{0,9},{1,4},{2,8},{4,3},{4,6},{4,7}
    };
    double rate = 0.08;
    for (auto [i, j] : edges) { Q(i, j) = rate; rate += 0.01; }
    for (std::size_t i = 0; i < K; ++i) Q(i, i) = -Q.row(i).sum();
    return Q;
}

phylocore::ctmc_model make_model(const Eigen::MatrixXd& Q, bool pade) {
    phylocore::ctmc_model m;
    m.Q = Q;
    m.use_pade_expm = pade;
    m.build();
    return m;
}

// Entrywise comparison of log P_ab(t) between the original model and a relabelled
// one. Entrywise matters: a matvec sums over states, so a single entry flipping
// from -inf to log(1e-15) is swamped by the large terms and the bug hides. The
// per-entry classification "reachable / unreachable" is the thing that must be
// ordering-independent.
double relabel_mismatch(const Eigen::MatrixXd& Q,
                        const std::vector<int>& perm,
                        bool pade,
                        double t) {
    Eigen::MatrixXd Qp(K, K);
    for (std::size_t i = 0; i < K; ++i)
        for (std::size_t j = 0; j < K; ++j)
            Qp(i, j) = Q(perm[i], perm[j]);

    auto base       = make_model(Q,  pade);
    auto perm_model = make_model(Qp, pade);

    std::monostate d{};
    double worst = 0.0;
    for (std::size_t a = 0; a < K; ++a) {
        for (std::size_t b = 0; b < K; ++b) {
            const double lp  = base.compute_log_transition_probability(
                                   d, 0, t, perm[a], perm[b]);
            const double lpp = perm_model.compute_log_transition_probability(
                                   d, 0, t, a, b);
            const bool inf_l = std::isinf(lp), inf_r = std::isinf(lpp);
            if (inf_l && inf_r) continue;                 // both impossible: agree
            if (inf_l != inf_r) return 1e9;               // one impossible, one not
            worst = std::max(worst, std::abs(lp - lpp));
        }
    }
    return worst;
}

} // namespace

int main() {
    const Eigen::MatrixXd Q = differentiation_tree_Q();

    int zeros = 0;
    for (std::size_t i = 0; i < K; ++i)
        for (std::size_t j = 0; j < K; ++j)
            if (i != j && Q(i, j) == 0.0) ++zeros;
    if (zeros != 81) {
        std::printf("FAIL: fixture is not the intended reducible Q (%d/90 zeros, expected 81)\n", zeros);
        return 1;
    }

    const std::vector<double> times = {0.05, 0.2, 0.5, 1.0, 2.5};
    std::mt19937 rng(12345);
    std::vector<int> perm(K);
    std::iota(perm.begin(), perm.end(), 0);

    constexpr double kTol = 1e-9;
    double worst_pade = 0.0, worst_eigen = 0.0;
    int fail = 0;

    for (int trial = 0; trial < 50; ++trial) {
        std::shuffle(perm.begin(), perm.end(), rng);
        for (double t : times) {
            const double dp = relabel_mismatch(Q, perm, /*pade=*/true,  t);
            const double de = relabel_mismatch(Q, perm, /*pade=*/false, t);
            worst_pade  = std::max(worst_pade,  dp);
            worst_eigen = std::max(worst_eigen, de);
            if (!(dp <= kTol)) {
                if (++fail <= 5)
                    std::printf("FAIL: pade path not invariant, trial %d t=%g, delta=%.3e\n",
                                trial, t, dp);
            }
        }
    }

    // Diagnostic only. The eigendecomposition path is still the default, and is
    // expected to violate invariance on this fixture; we report the magnitude
    // rather than assert on it so this test does not start failing if that path
    // is later fixed or removed.
    std::printf("eigendecomposition path (default): worst relabel mismatch = %.3e\n", worst_eigen);
    std::printf("pade path (--pade_expm)          : worst relabel mismatch = %.3e\n", worst_pade);

    if (fail) {
        std::printf("FAILED: %d permutation/time combinations broke invariance under Pade\n", fail);
        return 1;
    }
    std::printf("PASS: 50 relabellings x %zu branch lengths invariant under Pade (tol %.0e)\n",
                times.size(), kTol);
    return 0;
}
