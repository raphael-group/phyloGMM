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
// and maps +1e-15 to log(1e-15) ~ -35 ("possible") but -1e-15 to the finite
// NEGATIVE_INFINITY sentinel ("impossible"). Which side the residue falls on
// depends on the ordering of the eigenvector basis LAPACK returns, which depends
// on the ordering of Q. A
// reducible generator also has a repeated eigenvalue (0, with multiplicity equal
// to the number of communicating classes), so that basis is NOT UNIQUE -- there
// is a whole eigenspace of valid answers and the solver returns whichever its
// arithmetic reached. Occasionally it returns a near-degenerate one, U becomes
// numerically singular, and the whole matrix is wrong rather than just its zeros.
//
// The fix is ctmc_model::use_pade_expm. Its robust exponential path masks entries
// that are structurally unreachable in Q and, for reducible weight-1 CTMCs, uses
// non-negative uniformization before scaling-and-squaring. Uniformization is
// necessary because the linear solve inside a Pade exponential can introduce
// signed roundoff-sized fill or even change the sign of a tiny reachable value.
//
// THE IMPORTANT PART OF THIS TEST IS THE FIXTURE, NOT THE ASSERTION.
// A dense random Q passes on the buggy path too: with distinct eigenvalues the
// eigenvectors are unique and every ordering agrees to the last bit. The bug is
// only reachable with a REDUCIBLE Q. The fixture below is a differentiation-tree
// generator -- a rooted 9-edge DAG over 10 states, exactly the shape a cell-fate
// map produces -- with 81 of 90 off-diagonals exactly zero. Do not "simplify"
// this fixture to a random matrix; that silently disarms the test.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "constants.h"
#include "state_transition_models/ctmc.h"

namespace {

constexpr std::size_t K = 10;

// Rooted differentiation tree over 10 states (9 directed edges):
//   0 -> 1,2,5,9    1 -> 4    2 -> 8    4 -> 3,6,7
// Reducible: 81 of the 90 off-diagonal entries are exactly zero. Some acquire
// mass through indirect paths, but unreachable reverse and cross-branch pairs
// remain structural zeros in exp(tQ).
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

bool is_impossible(double log_probability) {
    // NEGATIVE_INFINITY is deliberately a finite -1e8 sentinel, not IEEE -inf.
    return log_probability == phylocore::NEGATIVE_INFINITY;
}

// Entrywise comparison of log P_ab(t) between the original model and a relabelled
// one. Entrywise matters: a matvec sums over states, so a single entry flipping
// from the impossible sentinel to log(1e-15) is swamped by the large terms and
// the bug hides. The per-entry classification "reachable / unreachable" is the
// thing that must be ordering-independent.
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
            const bool impossible_l = is_impossible(lp);
            const bool impossible_r = is_impossible(lpp);
            if (impossible_l && impossible_r) continue;
            if (impossible_l != impossible_r) return 1e9;
            if (!std::isfinite(lp) || !std::isfinite(lpp)) return 1e9;
            worst = std::max(worst, std::abs(lp - lpp));
        }
    }
    return worst;
}

// A small reducible generator for which Eigen's Pade solve produces positive
// roundoff-sized fill after the relabelling [1,2,0] unless ctmc_model applies
// its structural-reachability mask. At t=15, the exact zeros below have been
// observed as positive values around 1e-17, which the strict `p > 0` log guard
// would otherwise turn into finite log probabilities near -38.
Eigen::MatrixXd pade_roundoff_counterexample_Q() {
    Eigen::MatrixXd Q(3, 3);
    Q << -0.27, 0.10, 0.17,
          0.00,-0.23, 0.23,
          0.00, 0.00, 0.00;
    return Q;
}

int check_pade_roundoff_counterexample() {
    constexpr std::size_t n = 3;
    constexpr double t = 15.0;
    constexpr double tol = 1e-12;
    const std::array<int, n> perm = {1, 2, 0}; // new index -> old index

    const Eigen::MatrixXd Q = pade_roundoff_counterexample_Q();
    Eigen::MatrixXd Qp(n, n);
    std::array<std::size_t, n> inverse_perm{}; // old index -> new index
    for (std::size_t i = 0; i < n; ++i) {
        inverse_perm[static_cast<std::size_t>(perm[i])] = i;
        for (std::size_t j = 0; j < n; ++j)
            Qp(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                Q(perm[i], perm[j]);
    }

    auto base       = make_model(Q,  /*pade=*/true);
    auto perm_model = make_model(Qp, /*pade=*/true);
    std::monostate d{};
    int fail = 0;

    // Because Q is upper triangular with positive 0->1, 0->2, and 1->2
    // rates, P_ab(t) is structurally impossible exactly when a > b.
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = 0; b < n; ++b) {
            const std::size_t ap = inverse_perm[a];
            const std::size_t bp = inverse_perm[b];
            const double lp = base.compute_log_transition_probability(
                                  d, 0, t, a, b);
            const double lpp = perm_model.compute_log_transition_probability(
                                   d, 0, t, ap, bp);
            const bool expected_impossible = a > b;
            const bool bad = expected_impossible
                           ? (!is_impossible(lp) || !is_impossible(lpp))
                           : (is_impossible(lp) || is_impossible(lpp)
                              || !std::isfinite(lp) || !std::isfinite(lpp)
                              || std::abs(lp - lpp) > tol);
            if (bad) {
                ++fail;
                std::printf(
                    "FAIL: 3-state Pade scalar old(%zu,%zu) new(%zu,%zu): "
                    "base=%.17g permuted=%.17g expected_%s\n",
                    a, b, ap, bp, lp, lpp,
                    expected_impossible ? "impossible" : "reachable");
            }
        }
    }

    // Basis log-vectors expose one matrix entry at a time. Forward mode with
    // basis b returns log P_ab in result[a]; transpose mode with basis a
    // returns log P_ab in result[b]. This covers the optimized likelihood path
    // in both orientations, including the finite impossible-value sentinel.
    for (bool transpose : {false, true}) {
        for (std::size_t basis = 0; basis < n; ++basis) {
            std::vector<double> log_v(n, phylocore::NEGATIVE_INFINITY);
            std::vector<double> log_vp(n, phylocore::NEGATIVE_INFINITY);
            std::vector<double> result(n, phylocore::NEGATIVE_INFINITY);
            std::vector<double> resultp(n, phylocore::NEGATIVE_INFINITY);
            log_v[basis] = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                log_vp[i] = log_v[static_cast<std::size_t>(perm[i])];

            base.compute_log_pmatrix_vector_product(
                d, 0, t, log_v, result, transpose);
            perm_model.compute_log_pmatrix_vector_product(
                d, 0, t, log_vp, resultp, transpose);

            for (std::size_t output = 0; output < n; ++output) {
                const std::size_t a = transpose ? basis : output;
                const std::size_t b = transpose ? output : basis;
                const double lp = result[output];
                const double lpp = resultp[inverse_perm[output]];
                const bool expected_impossible = a > b;
                const bool bad = expected_impossible
                               ? (!is_impossible(lp) || !is_impossible(lpp))
                               : (is_impossible(lp) || is_impossible(lpp)
                                  || !std::isfinite(lp) || !std::isfinite(lpp)
                                  || std::abs(lp - lpp) > tol);
                if (bad) {
                    ++fail;
                    std::printf(
                        "FAIL: 3-state Pade %s matvec P(%zu,%zu): "
                        "base=%.17g permuted=%.17g expected_%s\n",
                        transpose ? "transpose" : "forward", a, b,
                        lp, lpp,
                        expected_impossible ? "impossible" : "reachable");
                }
            }
        }
    }

    if (!fail)
        std::printf(
            "PASS: 3-state Pade roundoff counterexample preserves structural "
            "zeros in scalar and forward/transpose matvec paths\n");
    return fail;
}

int check_transitive_reachability() {
    // Q(0,2) is zero, but 2 is reachable from 0 through 1. The structural mask
    // must use transitive closure rather than merely copying Q's zero pattern.
    Eigen::MatrixXd Q(3, 3);
    Q << -0.10, 0.10, 0.00,
          0.00,-0.20, 0.20,
          0.00, 0.00, 0.00;

    auto model = make_model(Q, /*pade=*/true);
    std::monostate d{};
    const double reachable = model.compute_log_transition_probability(
        d, 0, 2.0, 0, 2);
    const double impossible = model.compute_log_transition_probability(
        d, 0, 2.0, 2, 0);

    // Closed form for the two-stage pure-birth chain with rates a=0.1, b=0.2.
    const double p00 = std::exp(-0.10 * 2.0);
    const double p01 = 0.10 / (0.20 - 0.10)
                     * (std::exp(-0.10 * 2.0) - std::exp(-0.20 * 2.0));
    const double expected_log_p02 = std::log(1.0 - p00 - p01);

    if (is_impossible(reachable) || !std::isfinite(reachable)
        || std::abs(reachable - expected_log_p02) > 1e-12
        || !is_impossible(impossible)) {
        std::printf(
            "FAIL: Pade transitive reachability: log P(0,2)=%.17g, "
            "expected=%.17g, log P(2,0)=%.17g\n",
            reachable, expected_log_p02, impossible);
        return 1;
    }

    // Direct scalar/matvec calls at t=0 must be the exact identity, including
    // graph-reachable off-diagonal pairs whose positive mass appears only at t>0.
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            const double lp = model.compute_log_transition_probability(
                d, 0, 0.0, i, j);
            if ((i == j && lp != 0.0) || (i != j && !is_impossible(lp))) {
                std::printf(
                    "FAIL: Pade t=0 scalar P(%zu,%zu)=%.17g\n", i, j, lp);
                return 1;
            }
        }
    }
    const std::vector<double> log_v = {-0.3, -1.2, -2.1};
    for (bool transpose : {false, true}) {
        std::vector<double> result(3, phylocore::NEGATIVE_INFINITY);
        model.compute_log_pmatrix_vector_product(
            d, 0, 0.0, log_v, result, transpose);
        for (std::size_t i = 0; i < 3; ++i) {
            if (result[i] != log_v[i]) {
                std::printf(
                    "FAIL: Pade t=0 %s matvec[%zu]=%.17g, expected %.17g\n",
                    transpose ? "transpose" : "forward", i,
                    result[i], log_v[i]);
                return 1;
            }
        }
    }

    // On the reducible DAG fixture, state 4 cannot return after leaving, so its
    // diagonal transition is exactly exp(Q_44*t). At t=100 the probability is
    // exp(-45), small enough that dense Pade arithmetic previously changed its
    // sign under some state permutations.
    const Eigen::MatrixXd long_Q = differentiation_tree_Q();
    auto long_model = make_model(long_Q, /*pade=*/true);
    const double long_diagonal = long_model.compute_log_transition_probability(
        d, 0, 100.0, 4, 4);
    const double expected_long_diagonal = 100.0 * long_Q(4, 4);
    if (is_impossible(long_diagonal) || !std::isfinite(long_diagonal)
        || std::abs(long_diagonal - expected_long_diagonal) > 1e-10) {
        std::printf(
            "FAIL: robust long-branch log P(4,4)=%.17g, expected %.17g\n",
            long_diagonal, expected_long_diagonal);
        return 1;
    }
    std::printf(
        "PASS: robust exponential retains indirect/long transitions and t=0 identity\n");
    return 0;
}

int check_uniformization_tail_accuracy() {
    // In this long pure-birth chain, P(0,n-1) first becomes nonzero at the
    // (n-1)th uniformization term. Stopping merely because the next Poisson
    // weight is small relative to the whole matrix loses about 1.3% of this
    // tiny but representable transition probability.
    constexpr std::size_t n = 32;
    constexpr double rate = 0.17;
    constexpr double t = 2.5;
    const double mu = rate * t;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n, n);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        Q(static_cast<Eigen::Index>(i),
          static_cast<Eigen::Index>(i)) = -rate;
        Q(static_cast<Eigen::Index>(i),
          static_cast<Eigen::Index>(i + 1)) = rate;
    }

    // State n-1 is absorbing, hence this endpoint probability is the upper
    // Poisson tail Pr[N >= n-1], N ~ Poisson(mu). Build the tail from its first
    // term to avoid catastrophic cancellation in 1 - CDF(N <= n-2).
    double term = std::exp(-mu);
    for (std::size_t order = 1; order < n; ++order)
        term *= mu / static_cast<double>(order);
    double expected_probability = term;
    for (std::size_t order = n; term > 0.0; ++order) {
        term *= mu / static_cast<double>(order);
        expected_probability += term;
        if (term <= std::numeric_limits<double>::epsilon()
                    * expected_probability)
            break;
    }
    const double expected_log_probability = std::log(expected_probability);

    auto model = make_model(Q, /*pade=*/true);
    std::monostate d{};
    const double actual = model.compute_log_transition_probability(
        d, 0, t, 0, n - 1);
    if (is_impossible(actual) || !std::isfinite(actual)
        || std::abs(actual - expected_log_probability) > 1e-12) {
        std::printf(
            "FAIL: uniformization tail log P(0,%zu)=%.17g, expected %.17g\n",
            n - 1, actual, expected_log_probability);
        return 1;
    }
    std::printf(
        "PASS: uniformization retains tiny reachable Poisson-tail transitions\n");
    return 0;
}

int check_stiff_uniformization() {
    // The global uniformization rate is set by the fast disconnected block.
    // This forces many scaling/squaring steps for t=1 even though states 0/1
    // evolve at rate one. Without row normalization at every square, an
    // O(epsilon) base-series row-sum error grows into several percent of lost
    // probability mass in the slow block.
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(4, 4);
    Q(0, 0) = -1.0;
    Q(0, 1) = 1.0;
    Q(2, 2) = -1e14;
    Q(2, 3) = 1e14;

    auto model = make_model(Q, /*pade=*/true);
    std::monostate d{};
    const double log_p00 = model.compute_log_transition_probability(
        d, 0, 1.0, 0, 0);
    const double log_p01 = model.compute_log_transition_probability(
        d, 0, 1.0, 0, 1);
    const double expected_log_p01 = std::log1p(-std::exp(-1.0));
    if (is_impossible(log_p00) || is_impossible(log_p01)
        || !std::isfinite(log_p00) || !std::isfinite(log_p01)
        || std::abs(log_p00 + 1.0) > 1e-12
        || std::abs(log_p01 - expected_log_p01) > 1e-12) {
        std::printf(
            "FAIL: stiff uniformization slow row log P00=%.17g, P01=%.17g; "
            "expected -1 and %.17g\n",
            log_p00, log_p01, expected_log_p01);
        return 1;
    }

    // Multiplying by the all-ones vector exposes every transition-row sum.
    const std::vector<double> log_ones(4, 0.0);
    std::vector<double> log_row_sums(4, phylocore::NEGATIVE_INFINITY);
    model.compute_log_pmatrix_vector_product(
        d, 0, 1.0, log_ones, log_row_sums, /*transpose=*/false);
    for (std::size_t i = 0; i < log_row_sums.size(); ++i) {
        if (!std::isfinite(log_row_sums[i])
            || std::abs(log_row_sums[i]) > 1e-12) {
            std::printf(
                "FAIL: stiff uniformization log row sum[%zu]=%.17g\n",
                i, log_row_sums[i]);
            return 1;
        }
    }
    std::printf(
        "PASS: stiff uniformization preserves slow dynamics and stochastic rows\n");
    return 0;
}

} // namespace

int main() {
    const Eigen::MatrixXd Q = differentiation_tree_Q();

    int zeros = 0;
    for (std::size_t i = 0; i < K; ++i)
        for (std::size_t j = 0; j < K; ++j)
            if (i != j && Q(i, j) == 0.0) ++zeros;
    if (zeros != 81) {
        std::printf(
            "FAIL: fixture is not the intended reducible Q "
            "(%d/90 zeros, expected 81)\n",
            zeros);
        return 1;
    }

    const std::vector<double> times = {
        0.05, 0.2, 0.5, 1.0, 2.5, 15.0, 100.0
    };
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

    fail += check_pade_roundoff_counterexample();
    fail += check_transitive_reachability();
    fail += check_uniformization_tail_accuracy();
    fail += check_stiff_uniformization();

    // Diagnostic only. The eigendecomposition path is still the default, and is
    // expected to violate invariance on this fixture; we report the magnitude
    // rather than assert on it so this test does not start failing if that path
    // is later fixed or removed.
    std::printf("eigendecomposition path (default): worst relabel mismatch = %.3e\n", worst_eigen);
    std::printf("robust path (--pade_expm)        : worst relabel mismatch = %.3e\n", worst_pade);

    if (fail) {
        std::printf("FAILED: %d robust exponential checks failed\n", fail);
        return 1;
    }
    std::printf(
        "PASS: 50 relabellings x %zu branch lengths invariant under robust "
        "backend (tol %.0e)\n",
        times.size(), kTol);
    return 0;
}
