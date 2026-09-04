#ifndef PHYLOCORE_OPTIMIZE_VAN_LOAN_H
#define PHYLOCORE_OPTIMIZE_VAN_LOAN_H

// Van Loan expected-sufficient-statistics helper: a numerically robust drop-in
// alternative to optimize::hobolth_jensen_stats that avoids the
// eigendecomposition of Q entirely.
//
// For each edge e = (parent w -> child u) with branch length t_e, the
// endpoint-conditioned integral that Hobolth-Jensen assembles in eigenspace is
// exactly a Van Loan integral contracted with the same endpoint-weight matrix
// W_e (built identically to hobolth_jensen_stats):
//
//     C_e = int_0^{t_e} exp(Q^T s) * W_e * exp(Q^T (t_e - s)) ds
//         = top-right k x k block of  exp( t_e * [ Q^T  W_e ]
//                                                [  0   Q^T ] )
//
// (Van Loan 1978, "Computing integrals involving the matrix exponential.")
// The block exponential is evaluated with Eigen's scaling-and-squaring
// MatrixExponential, which is backward-stable for defective / near-absorbing Q
// where the Hobolth-Jensen inverse-eigenvector sandwich (U^-1) blows up. All
// arithmetic is real (Q and W_e are real), so there is no complex pipeline.
//
//     E[T_i  | y] = sum_e C_e[i, i]
//     E[N_ij | y] = Q_ij * sum_e C_e[i, j]   (i != j)
//
// The aggregate return type (hobolth_jensen_aggregate) and the W_e assembly are
// shared with hobolth_jensen.h so the two paths are interchangeable behind a
// flag: compute_suff_stats() dispatches to one or the other.
//
// NOTE: targets the ordinary exp(tQ) model (branch weight 1), which is what
// phylo_gmm uses; the active branch operator is then Q == stm.rate_matrix().
// Weighted complete-path CTMC models (w != 1) are not handled here.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "hobolth_jensen.h"   // hobolth_jensen_aggregate
#include "../phylogeny.h"

namespace phylocore {
namespace optimize {

/*!
 * @brief Van Loan expected-sufficient-statistics aggregate.
 *
 * Mirrors hobolth_jensen_stats' output (N_total, T_total) but computes each
 * edge's endpoint-conditioned integral via a 2k x 2k matrix exponential rather
 * than the eigendecomposition of Q. See file header for the identity.
 *
 * @tparam Problem joint_optimization_problem<Model, ...> subclass with the same
 *                 buffer / posterior accessors used by hobolth_jensen_stats.
 *                 Model must expose rate_matrix() (real k x k generator Q).
 */
template <typename Problem>
hobolth_jensen_aggregate van_loan_stats(Problem& problem)
{
    auto& stm = *problem.stm;

    const Eigen::MatrixXd& Q = stm.rate_matrix();   // raw real generator, no eigen
    const std::size_t k = static_cast<std::size_t>(Q.rows());

    if (stm.sites_latent_states.empty()) {
        throw std::runtime_error("van_loan_stats: stm has no sites");
    }
    for (std::size_t c = 1; c < stm.sites_latent_states.size(); ++c) {
        if (stm.sites_latent_states[c].alphabet_size !=
            stm.sites_latent_states[0].alphabet_size) {
            throw std::runtime_error(
                "van_loan_stats: heterogeneous alphabet sizes not supported");
        }
    }
    if (stm.sites_latent_states[0].alphabet_size != k) {
        throw std::runtime_error(
            "van_loan_stats: alphabet size does not match rate matrix dimension");
    }

    hobolth_jensen_aggregate agg;
    agg.N_total = Eigen::MatrixXd::Zero(k, k);
    agg.T_total = Eigen::VectorXd::Zero(k);

    Eigen::MatrixXd C_total = Eigen::MatrixXd::Zero(k, k);
    const Eigen::MatrixXd Qt = Q.transpose();
    const std::size_t num_sites = stm.sites_latent_states.size();

    // Per-edge scratch.
    Eigen::MatrixXd W_e(k, k);
    Eigen::MatrixXd M(2 * k, 2 * k);
    Eigen::MatrixXd E(2 * k, 2 * k);
    std::vector<double> log_sib(k);

    for (std::size_t ti = 0; ti < problem.trees.size(); ++ti) {
        const tree& t          = *problem.trees[ti];
        const auto& inside_ll  = problem.inside_buffer(ti);
        const auto& ei_ll      = problem.edge_inside_buffer(ti);
        const auto& outside_ll = problem.outside_buffer(ti);

        for (std::size_t c = 0; c < num_sites; ++c) {
            const double log_p_y = problem.log_p_y(ti, c);

            for (std::size_t v = 0; v < t.num_nodes; ++v) {
                if (v == t.root_id) continue;

                const std::size_t w =
                    static_cast<std::size_t>(t.tree.predecessors(v)[0]);
                const double t_e = t.branch_lengths[v];

                // log_sib[a] = sum_{sibling z of v} edge_inside(z, a)
                for (std::size_t a = 0; a < k; ++a) log_sib[a] = 0.0;
                for (auto sib : t.tree.successors(w)) {
                    if (static_cast<std::size_t>(sib) == v) continue;
                    for (std::size_t a = 0; a < k; ++a)
                        log_sib[a] += ei_ll(c, sib, a);
                }

                // W_e[a, b] = exp(outside(w, a) + log_sib(a) + inside(v, b) - log p(y))
                for (std::size_t a = 0; a < k; ++a) {
                    const double row_log = outside_ll(c, w, a) + log_sib[a] - log_p_y;
                    for (std::size_t b = 0; b < k; ++b)
                        W_e(a, b) = std::exp(row_log + inside_ll(c, v, b));
                }

                // Van Loan block:  M = [[Q^T, W_e], [0, Q^T]];  C_e = top-right of exp(t_e M)
                M.setZero();
                M.topLeftCorner(k, k)     = t_e * Qt;
                M.bottomRightCorner(k, k) = t_e * Qt;
                M.topRightCorner(k, k)    = t_e * W_e;
                E = M.exp();
                C_total += E.topRightCorner(k, k);
            }
        }
    }

    for (std::size_t i = 0; i < k; ++i)
        agg.T_total[i] = C_total(i, i);

    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < k; ++j)
            if (i != j)
                agg.N_total(i, j) = Q(i, j) * C_total(i, j);

    return agg;
}

} // namespace optimize
} // namespace phylocore

#endif // PHYLOCORE_OPTIMIZE_VAN_LOAN_H
