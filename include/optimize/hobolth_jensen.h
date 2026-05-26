#ifndef PHYLOCORE_OPTIMIZE_HOBOLTH_JENSEN_H
#define PHYLOCORE_OPTIMIZE_HOBOLTH_JENSEN_H

// Header-only Hobolth-Jensen expected-sufficient-statistics helper for a
// reversible CTMC on a (rooted) tree. No IPOPT dependency.
//
// For each edge e = (parent w → child u) with branch length t_e, the helper
// computes the expected number of substitutions E[N_ij | y]_e from state i to
// state j (i ≠ j) and the expected dwell time E[T_i | y]_e in state i,
// integrating analytically over the unobserved trajectory of the latent
// CTMC conditional on the observation y. It then sums these contributions
// across (1) every edge of every tree in problem.trees and (2) every site
// — yielding two aggregate suff-stat tensors suitable for the closed-form
// reversible GTR M-step.
//
// Math reference: Hobolth & Jensen, "Statistical inference in evolutionary
// models of DNA sequences via the EM algorithm" (Stat. Appl. Genet. Mol.
// Biol., 2005), eq. 27. The kernel is
//
//     J_{pq}(t) = ∫_0^t exp(λ_p · s) · exp(λ_q · (t-s)) ds
//               = t · exp(t(λ_p+λ_q)/2) · sinhc(t(λ_p-λ_q)/2)
//
// (sinhc(x) = sinh(x)/x, sinhc(0) = 1), which is numerically stable as
// λ_p → λ_q.
//
// Because reversible_ctmc_model caches the eigendecomposition of the
// SYMMETRIC matrix A = diag(√π)·Q·diag(1/√π) (A = U·diag(λ)·Uᵀ with U
// orthogonal and λ real), we can run the entire computation in real
// arithmetic. With U_Q  = diag(1/√π) · U and U_Q⁻¹ = Uᵀ · diag(√π):
//
//   W_e[a,b] = exp(outside(parent, a) + sibling_edge_inside(a)
//                 + inside(child, b) − log p(y))
//   G_e      = U_Qᵀ · W_e · (U_Q⁻¹)ᵀ
//            = Uᵀ · ( diag(1/√π) · W_e · diag(√π) ) · U          (real)
//   H        = Σ_e  G_e ⊙ J_e          where J_e[p,q] = J_{pq}(t_e)
//   C        = (U_Q⁻¹)ᵀ · H · U_Qᵀ
//            = diag(√π) · U · H · Uᵀ · diag(1/√π)               (real)
//
//   E[T_i  | y]_e  = C[i, i]
//   E[N_ij | y]_e  = Q_ij · C[i, j]    (i ≠ j)
//
// The helper sums these aggregates across edges, trees, and sites.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "../phylogeny.h"

namespace phylocore {
namespace optimize {

/*!
 * @brief Aggregate Hobolth-Jensen sufficient statistics across all (tree,
 *        edge, site) triples.
 *
 *   N_total(i, j) = Σ_{(tree, edge, site)} E[N_ij | y]_e   (i ≠ j)
 *                   (diagonal is left at zero)
 *   T_total(i)    = Σ_{(tree, edge, site)} E[T_i  | y]_e
 *
 * Both tensors are k × k and k respectively, where k is the alphabet size.
 * The helper assumes a single shared alphabet size across all sites (true
 * for GTR on nucleotides). If a mismatched alphabet is encountered, the
 * helper throws std::runtime_error.
 */
struct hobolth_jensen_aggregate {
    Eigen::MatrixXd N_total;  ///< k × k, summed E[N_ij | y] across (tree, edge, site)
    Eigen::VectorXd T_total;  ///< k,     summed E[T_i  | y] across (tree, edge, site)
};

/*!
 * @brief Numerically stable sinhc kernel for the Hobolth-Jensen J-integral.
 *
 *   J_{pq}(t) = t · exp(t(λ_p+λ_q)/2) · sinhc(t(λ_p-λ_q)/2)
 *
 * Uses a small-|x| threshold (|x| < 1e-8) to return 1.0 directly, avoiding
 * 0/0 when λ_p ≈ λ_q.
 */
inline double hobolth_jensen_kernel_real(double lambda_p,
                                         double lambda_q,
                                         double t)
{
    const double half_sum  = 0.5 * t * (lambda_p + lambda_q);
    const double half_diff = 0.5 * t * (lambda_p - lambda_q);
    double sinhc;
    if (std::abs(half_diff) < 1e-8) {
        sinhc = 1.0;
    } else {
        sinhc = std::sinh(half_diff) / half_diff;
    }
    return t * std::exp(half_sum) * sinhc;
}

/*!
 * @brief Compute the Hobolth-Jensen aggregate sufficient statistics for an
 *        already-traversed joint_optimization_problem.
 *
 * Pre-conditions (enforced by optimize::em before calling compute_suff_stats):
 *   - problem.log_likelihood() has been called → inside buffers populated and
 *     log_total_likelihood(tree_idx) is valid.
 *   - compute_edge_inside_log_likelihood + compute_outside_log_likelihood have
 *     been run for every tree → edge_inside_buffer and outside_buffer are
 *     populated.
 *
 * The problem's stm must be a reversible_ctmc_model — accessors
 * eigenvectors(site), eigenvalues(site), sqrt_pi(site), inv_sqrt_pi(site)
 * are dispatched directly on *problem.stm.
 *
 * @tparam Problem joint_optimization_problem<reversible_ctmc_model, ...>
 *                 subclass with the standard buffer + posterior accessors.
 */
template <typename Problem>
hobolth_jensen_aggregate hobolth_jensen_stats(Problem& problem)
{
    const auto& stm = *problem.stm;

    // Resolve k from site 0; require it to be uniform across sites.
    if (stm.sites_latent_states.empty()) {
        throw std::runtime_error(
            "hobolth_jensen_stats: stm has no sites");
    }
    const std::size_t k = stm.sites_latent_states[0].alphabet_size;
    for (std::size_t c = 1; c < stm.sites_latent_states.size(); ++c) {
        if (stm.sites_latent_states[c].alphabet_size != k) {
            throw std::runtime_error(
                "hobolth_jensen_stats: heterogeneous alphabet sizes not supported");
        }
    }

    hobolth_jensen_aggregate agg;
    agg.N_total = Eigen::MatrixXd::Zero(k, k);
    agg.T_total = Eigen::VectorXd::Zero(k);

    const std::size_t num_sites = stm.sites_latent_states.size();

    // Per-site scratch reused inside the inner loop.
    Eigen::MatrixXd W_e(k, k);
    Eigen::MatrixXd Wtilde(k, k);
    Eigen::MatrixXd G_e(k, k);
    Eigen::MatrixXd J_e(k, k);
    Eigen::MatrixXd H(k, k);
    Eigen::MatrixXd C(k, k);
    std::vector<double> log_sib(k);

    for (std::size_t ti = 0; ti < problem.trees.size(); ++ti) {
        const tree& t           = *problem.trees[ti];
        const auto& inside_ll   = problem.inside_buffer(ti);
        const auto& ei_ll       = problem.edge_inside_buffer(ti);
        const auto& outside_ll  = problem.outside_buffer(ti);

        for (std::size_t c = 0; c < num_sites; ++c) {
            const Eigen::MatrixXd& U          = stm.eigenvectors(c);
            const Eigen::VectorXd& lambda     = stm.eigenvalues(c);
            const Eigen::VectorXd& sqrt_pi    = stm.sqrt_pi(c);
            const Eigen::VectorXd& inv_sqrt_pi= stm.inv_sqrt_pi(c);

            // Q[i,j] = inv_sqrt_pi[i] * sqrt_pi[j] · Σ_p U[i,p] λ_p U[j,p]
            // We materialize the full real Q matrix once per site; this is
            // O(k^3) — negligible for k=4.
            Eigen::MatrixXd Q = inv_sqrt_pi.asDiagonal()
                              * U * lambda.asDiagonal() * U.transpose()
                              * sqrt_pi.asDiagonal();

            const double log_p_y = problem.log_p_y(ti, c);

            H.setZero();

            // Walk every non-root node v (each carries its incoming edge).
            for (std::size_t v = 0; v < t.num_nodes; ++v) {
                if (v == t.root_id) continue;

                const std::size_t w =
                    static_cast<std::size_t>(t.tree.predecessors(v)[0]);
                const double t_e = t.branch_lengths[v];

                // log_sib[a] = Σ_{sibling z of v} edge_inside(z, a)
                for (std::size_t a = 0; a < k; ++a) log_sib[a] = 0.0;
                for (auto sib : t.tree.successors(w)) {
                    if (static_cast<std::size_t>(sib) == v) continue;
                    for (std::size_t a = 0; a < k; ++a)
                        log_sib[a] += ei_ll(c, sib, a);
                }

                // W_e[a, b] = exp(outside(w, a) + log_sib(a)
                //                  + inside(v, b) - log p(y))
                for (std::size_t a = 0; a < k; ++a) {
                    const double row_log = outside_ll(c, w, a)
                                         + log_sib[a]
                                         - log_p_y;
                    for (std::size_t b = 0; b < k; ++b) {
                        W_e(a, b) = std::exp(row_log + inside_ll(c, v, b));
                    }
                }

                // Wtilde = diag(1/√π) · W_e · diag(√π)
                // G_e    = Uᵀ · Wtilde · U
                Wtilde = inv_sqrt_pi.asDiagonal() * W_e * sqrt_pi.asDiagonal();
                G_e    = U.transpose() * Wtilde * U;

                // J_e[p, q] = kernel(λ_p, λ_q, t_e)
                for (std::size_t p = 0; p < k; ++p) {
                    for (std::size_t q = 0; q < k; ++q) {
                        J_e(p, q) = hobolth_jensen_kernel_real(
                            lambda[p], lambda[q], t_e);
                    }
                }

                H += G_e.cwiseProduct(J_e);
            }

            // C = diag(√π) · U · H · Uᵀ · diag(1/√π)
            C = sqrt_pi.asDiagonal() * U * H * U.transpose()
                * inv_sqrt_pi.asDiagonal();

            for (std::size_t i = 0; i < k; ++i)
                agg.T_total[i] += C(i, i);

            for (std::size_t i = 0; i < k; ++i)
                for (std::size_t j = 0; j < k; ++j)
                    if (i != j)
                        agg.N_total(i, j) += Q(i, j) * C(i, j);
        }
    }

    return agg;
}

} // namespace optimize
} // namespace phylocore

#endif // PHYLOCORE_OPTIMIZE_HOBOLTH_JENSEN_H
