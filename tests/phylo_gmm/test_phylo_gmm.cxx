#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "em_optimization.h"
#include "models/gtr.h"
#include "models/phylo_gmm_model.h"
#include "phylogeny.h"

using namespace phylocore;

static bool approx(double a, double b, double tol = 1e-5) {
    return std::abs(a - b) < tol;
}

// Construct a normalized GTR rate matrix for k=2 from a single exchangeability
// rate and stationary frequencies [pi0, pi1].
static Eigen::MatrixXd gtr_to_Q(double rate, double pi0, double pi1) {
    Eigen::MatrixXd Q(2, 2);
    Q(0, 1) = rate * pi1;
    Q(1, 0) = rate * pi0;
    Q(0, 0) = -Q(0, 1);
    Q(1, 1) = -Q(1, 0);
    double mu = -(pi0 * Q(0, 0) + pi1 * Q(1, 1));
    Q /= mu;
    return Q;
}

// Balanced 4-leaf binary tree:
//   root(6) -> internal(4) -> leaf0, leaf1
//           -> internal(5) -> leaf2, leaf3
static tree make_4leaf_tree(
    double leaf_blen     = 1.0,
    double internal_blen = 0.5,
    double root_blen     = 0.1)
{
    tree t;
    t.num_leaves = 4;
    t.num_nodes  = 7;
    t.root_id    = 6;
    t.branch_lengths.resize(7);
    t.node_names.resize(7);

    for (size_t i = 0; i < 7; ++i) t.tree.add_vertex(i);
    t.tree.add_edge(6, 4); t.tree.add_edge(6, 5);
    t.tree.add_edge(4, 0); t.tree.add_edge(4, 1);
    t.tree.add_edge(5, 2); t.tree.add_edge(5, 3);

    t.branch_lengths[0] = leaf_blen;
    t.branch_lengths[1] = leaf_blen;
    t.branch_lengths[2] = leaf_blen;
    t.branch_lengths[3] = leaf_blen;
    t.branch_lengths[4] = internal_blen;
    t.branch_lengths[5] = internal_blen;
    t.branch_lengths[6] = root_blen;
    return t;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

static void test_build_valid() {
    Eigen::MatrixXd Q = gtr_to_Q(1.0, 0.4, 0.6);
    Eigen::VectorXd pi(2); pi << 0.4, 0.6;
    Eigen::MatrixXd Y(1, 1); Y(0, 0) = 0.0;
    std::vector<Eigen::VectorXd> mu  = { Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1) };
    std::vector<Eigen::MatrixXd> Sig = { Eigen::MatrixXd::Identity(1, 1),
                                         Eigen::MatrixXd::Identity(1, 1) };
    try {
        phylo_gmm_model m(Q, mu, Sig, pi, Y);
        std::cout << "[PASS] build() succeeds for a valid reversible Q\n";
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] build() threw unexpectedly: " << e.what() << "\n";
    }
}

static void test_build_throws_negative_offdiag() {
    // Q(1,0) = -0.1 — violates off-diagonal non-negativity
    Eigen::MatrixXd Q(2, 2); Q << -1.0, 1.0, -0.1, 0.1;
    Eigen::VectorXd pi(2); pi << 0.5, 0.5;
    Eigen::MatrixXd Y(1, 1); Y(0, 0) = 0.0;
    std::vector<Eigen::VectorXd> mu  = { Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1) };
    std::vector<Eigen::MatrixXd> Sig = { Eigen::MatrixXd::Identity(1, 1),
                                         Eigen::MatrixXd::Identity(1, 1) };
    try {
        phylo_gmm_model m(Q, mu, Sig, pi, Y);
        std::cerr << "[FAIL] build() should have thrown for negative off-diagonal\n";
    } catch (const std::runtime_error&) {
        std::cout << "[PASS] build() throws for negative off-diagonal Q entry\n";
    }
}

static void test_build_throws_bad_row_sum() {
    // Row 0 sums to -0.5, not 0
    Eigen::MatrixXd Q(2, 2); Q << -1.0, 0.5, 1.0, -1.0;
    Eigen::VectorXd pi(2); pi << 0.5, 0.5;
    Eigen::MatrixXd Y(1, 1); Y(0, 0) = 0.0;
    std::vector<Eigen::VectorXd> mu  = { Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1) };
    std::vector<Eigen::MatrixXd> Sig = { Eigen::MatrixXd::Identity(1, 1),
                                         Eigen::MatrixXd::Identity(1, 1) };
    try {
        phylo_gmm_model m(Q, mu, Sig, pi, Y);
        std::cerr << "[FAIL] build() should have thrown for non-zero row sum\n";
    } catch (const std::runtime_error&) {
        std::cout << "[PASS] build() throws when Q row does not sum to zero\n";
    }
}

static void test_pmatrix_agrees_with_gtr() {
    // For a reversible Q, phylo_gmm_model and gtr_model must produce
    // the same P(t)*v and P(t)^T*v at all branch lengths.
    double rate = 1.0, pi0 = 0.4, pi1 = 0.6;

    gtr_model gtr;
    gtr.rates = {{rate}};
    gtr.freqs = {{pi0, pi1}};
    gtr.character_matrix = {};
    gtr.build();

    Eigen::MatrixXd Q = gtr_to_Q(rate, pi0, pi1);
    Eigen::VectorXd pi(2); pi << pi0, pi1;
    Eigen::MatrixXd Y(1, 1); Y(0, 0) = 0.0;
    std::vector<Eigen::VectorXd> mu  = { Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1) };
    std::vector<Eigen::MatrixXd> Sig = { Eigen::MatrixXd::Identity(1, 1),
                                         Eigen::MatrixXd::Identity(1, 1) };
    phylo_gmm_model gmm(Q, mu, Sig, pi, Y);

    gtr_node_data       gd;
    phylo_gmm_node_data pd;
    std::vector<double> log_v       = {std::log(0.7), std::log(0.3)};
    std::vector<double> res_gtr(2), res_gmm(2);
    std::vector<double> rest_gtr(2), rest_gmm(2);

    for (double blen : {0.01, 0.5, 2.0}) {
        gtr.compute_log_pmatrix_vector_product(gd, 0, blen, log_v, res_gtr);
        gmm.compute_log_pmatrix_vector_product(pd, 0, blen, log_v, res_gmm);
        gtr.compute_log_pmatrix_transpose_vector_product(gd, 0, blen, log_v, rest_gtr);
        gmm.compute_log_pmatrix_transpose_vector_product(pd, 0, blen, log_v, rest_gmm);

        for (int i = 0; i < 2; ++i) {
            if (!approx(res_gtr[i], res_gmm[i], 1e-6)) {
                std::cerr << "[FAIL] P(t=" << blen << ")*v[" << i << "]: "
                          << "gtr=" << res_gtr[i] << " gmm=" << res_gmm[i] << "\n";
                return;
            }
            if (!approx(rest_gtr[i], rest_gmm[i], 1e-6)) {
                std::cerr << "[FAIL] P(t=" << blen << ")^T*v[" << i << "]: "
                          << "gtr=" << rest_gtr[i] << " gmm=" << rest_gmm[i] << "\n";
                return;
            }
        }
    }
    std::cout << "[PASS] P(t)*v and P(t)^T*v agree with gtr_model for reversible Q\n";
}

static void test_gaussian_emission() {
    // log N(0; mu_0=0, Sigma=I) and log N(0; mu_1=5, Sigma=I) for d=1
    Eigen::MatrixXd Q = gtr_to_Q(1.0, 0.5, 0.5);
    Eigen::VectorXd pi(2); pi << 0.5, 0.5;
    Eigen::MatrixXd Y(1, 1); Y(0, 0) = 0.0;
    std::vector<Eigen::VectorXd> mu  = { Eigen::VectorXd::Zero(1),
                                         5.0 * Eigen::VectorXd::Ones(1) };
    std::vector<Eigen::MatrixXd> Sig = { Eigen::MatrixXd::Identity(1, 1),
                                         Eigen::MatrixXd::Identity(1, 1) };

    phylo_gmm_model m(Q, mu, Sig, pi, Y);
    phylo_gmm_node_data d;
    std::vector<double> result(2);
    m.compute_taxa_log_inside_likelihood(d, 0, 0, result);

    double expected_0 = -0.5 * std::log(2.0 * M_PI);
    double expected_1 = -0.5 * std::log(2.0 * M_PI) - 12.5;

    if (approx(result[0], expected_0, 1e-8) && approx(result[1], expected_1, 1e-8))
        std::cout << "[PASS] compute_taxa_log_inside_likelihood returns correct Gaussian log-density\n";
    else
        std::cerr << "[FAIL] emission: got [" << result[0] << ", " << result[1]
                  << "], expected [" << expected_0 << ", " << expected_1 << "]\n";
}

static void test_em_improves_likelihood_and_recovers_clusters() {
    // 4-leaf balanced binary tree, k=2, d=1
    // Leaves 0,1 near cluster 0 (y ≈ 0); leaves 2,3 near cluster 1 (y ≈ 5).
    // Initialized at the true parameters — EM should not worsen LL and should
    // keep cluster centers near 0 and 5.
    Eigen::MatrixXd Y(4, 1);
    Y << 0.1, -0.1, 5.1, 4.9;

    Eigen::MatrixXd Q = gtr_to_Q(1.0, 0.5, 0.5);
    Eigen::VectorXd pi(2); pi << 0.5, 0.5;
    std::vector<Eigen::VectorXd> mu  = { Eigen::VectorXd::Zero(1),
                                         5.0 * Eigen::VectorXd::Ones(1) };
    std::vector<Eigen::MatrixXd> Sig = { Eigen::MatrixXd::Identity(1, 1),
                                         Eigen::MatrixXd::Identity(1, 1) };

    phylo_gmm_model m(Q, mu, Sig, pi, Y);
    tree t = make_4leaf_tree();
    std::vector<phylo_gmm_node_data> node_data(t.num_nodes);

    // Initial log-likelihood
    likelihood_buffer buf(1, 2, t.num_nodes);
    double ll0 = compute_inside_log_likelihood(m, t, buf, node_data)[0];

    double ll_final = em_optimize(t, m, node_data, 1e-6, 50);

    if (ll_final >= ll0 - 1e-5)
        std::cout << "[PASS] EM does not decrease log-likelihood ("
                  << ll0 << " -> " << ll_final << ")\n";
    else
        std::cerr << "[FAIL] EM decreased log-likelihood ("
                  << ll0 << " -> " << ll_final << ")\n";

    double c0 = std::min(m.mu[0][0], m.mu[1][0]);
    double c1 = std::max(m.mu[0][0], m.mu[1][0]);
    if (approx(c0, 0.0, 0.5) && approx(c1, 5.0, 0.5))
        std::cout << "[PASS] EM keeps cluster centers near 0 and 5 (got "
                  << c0 << ", " << c1 << ")\n";
    else
        std::cerr << "[FAIL] cluster centers: got " << c0 << " and " << c1
                  << ", expected ~0 and ~5\n";
}

int main() {
    std::cout << "=== phylo_gmm_model tests ===\n";
    test_build_valid();
    test_build_throws_negative_offdiag();
    test_build_throws_bad_row_sum();
    test_pmatrix_agrees_with_gtr();
    test_gaussian_emission();
    test_em_improves_likelihood_and_recovers_clusters();
    std::cout << "=== Done ===\n";
    return 0;
}
