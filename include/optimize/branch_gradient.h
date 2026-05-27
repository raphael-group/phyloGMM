#ifndef PHYLOCORE_OPTIMIZE_BRANCH_GRADIENT_H
#define PHYLOCORE_OPTIMIZE_BRANCH_GRADIENT_H

// Header-only analytic branch-length gradient helper for a general CTMC on
// a (rooted) tree. No IPOPT dependency.
//
// Computes d log_L / d t_e for every non-root edge e of every tree in
// problem.trees, in the same order they appear in get_params() when the
// problem packs branch lengths (skipping the root entry whenever the model
// has !has_meaningful_root_branch — the convention used by gtr_blens_problem
// and friends).
//
// Math (Hobolth & Jensen 2011; matches the per-edge formulation in the
// phyloGMM paper):
//
//   d p(y_c) / d t_e = Σ_{a,b} M(a) · (Q · P(t_e))_{ab} · I(b)
//
// where, for edge (parent w → child u) at site c:
//   M(a) = exp(outside(c, w, a) + Σ_{siblings z of u} edge_inside(c, z, a))
//   I(b) = exp(inside(c, u, b))
//
// Per-site contribution to d log p(y_c) / d t_e is then the above divided by
// p(y_c). Summing across sites — and across trees — gives the total tree
// log-likelihood gradient at edge e.
//
// (Q · P(t))_{ab} is evaluated via the cached eigendecomposition Q =
// U · Λ · U⁻¹:
//
//   (Q · P(t))_{ab} = Σ_p U[a, p] · λ_p · exp(λ_p · t) · U⁻¹[p, b]
//
// so each (edge, site) contribution costs O(k^2) with a single O(k^3)
// reconstruction of the (Q · P(t)) matrix per (edge, site) — fine for typical
// nucleotide / amino-acid alphabet sizes.
//
// The implementation is templated on the scalar type deduced from the model's
// eigendecomposition of Q (real for reversible_ctmc_model, complex for the
// general ctmc_model), reusing the to_real() overloads from hobolth_jensen.h.
//
// Prerequisite (caller's responsibility, identical to the EM / Hobolth-Jensen
// path):
//   - problem.log_likelihood() has been called so that inside buffers and
//     log_p_y_per_tree_per_site_ are populated.
//   - phylocore::compute_edge_inside_log_likelihood and
//     phylocore::compute_outside_log_likelihood have been run for each tree.
//
// Returned vector layout matches the canonical get_params() layout for
// branch-length problems:
//
//   for each tree t in problem.trees:
//     for each node v in 0..t->num_nodes-1:
//       if (v == t->root_id && !problem.stm->has_meaningful_root_branch): skip
//       append d log_L / d t_v

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include "../phylogeny.h"
#include "hobolth_jensen.h"  // for to_real() overloads

namespace phylocore {
namespace optimize {

/*!
 * @brief Analytic branch-length gradient.
 *
 * See file header for the math and the precondition contract. Returns the
 * per-branch d log_L / d t_e vector in get_params()-style packing order.
 *
 * Templated on the problem subclass (so we can deduce the model's scalar
 * type from stm.eigenvalues_q(0)).
 */
template <typename Problem>
std::vector<double> branch_length_gradient(Problem& problem)
{
    auto& stm = *problem.stm;

    // Deduce scalar / matrix / vector types from the model.
    using EigVecT = std::decay_t<decltype(stm.eigenvalues_q(0))>;
    using Scalar  = typename EigVecT::Scalar;
    using MatT    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VecT    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    static_assert(
        std::is_same<Scalar, double>::value ||
        std::is_same<Scalar, std::complex<double>>::value,
        "branch_length_gradient: model scalar must be double or std::complex<double>");

    if (stm.sites_latent_states.empty()) {
        throw std::runtime_error(
            "branch_length_gradient: stm has no sites");
    }
    const std::size_t k = stm.sites_latent_states[0].alphabet_size;
    for (std::size_t c = 1; c < stm.sites_latent_states.size(); ++c) {
        if (stm.sites_latent_states[c].alphabet_size != k) {
            throw std::runtime_error(
                "branch_length_gradient: heterogeneous alphabet sizes not supported");
        }
    }

    const std::size_t num_sites = stm.sites_latent_states.size();

    // Pre-size output to the get_params() count so the optimizer's gradient
    // adapter can perform a simple element-wise copy.
    std::size_t num_params = 0;
    for (const auto* t : problem.trees) {
        for (std::size_t v = 0; v < t->num_nodes; ++v) {
            if (v == t->root_id && !stm.has_meaningful_root_branch) continue;
            ++num_params;
        }
    }
    std::vector<double> grad(num_params, 0.0);

    // Per-(edge, site) scratch.
    MatT QP(k, k);                  // Q · P(t_e), in the model's Scalar
    std::vector<double> log_sib(k); // sibling contribution at parent

    std::size_t out_idx = 0;
    for (std::size_t ti = 0; ti < problem.trees.size(); ++ti) {
        const tree& t          = *problem.trees[ti];
        const auto& inside_ll  = problem.inside_buffer(ti);
        const auto& ei_ll      = problem.edge_inside_buffer(ti);
        const auto& outside_ll = problem.outside_buffer(ti);

        for (std::size_t v = 0; v < t.num_nodes; ++v) {
            if (v == t.root_id && !stm.has_meaningful_root_branch) {
                continue;
            }

            // The root branch has special semantics (root_prior · P(t_root))
            // and is not handled by this helper — if a model legitimately
            // sets has_meaningful_root_branch=true and a user opts in, we
            // throw early so the failure is loud rather than silently wrong.
            if (v == t.root_id) {
                throw std::runtime_error(
                    "branch_length_gradient: has_meaningful_root_branch=true "
                    "is not yet supported by the analytic branch gradient");
            }

            const std::size_t w  = static_cast<std::size_t>(
                                       t.tree.predecessors(v)[0]);
            const double      te = t.branch_lengths[v];

            double accum = 0.0;

            for (std::size_t c = 0; c < num_sites; ++c) {
                const MatT& U_Q     = stm.eigenvectors_q(c);
                const VecT& lambda  = stm.eigenvalues_q(c);
                const MatT& U_Q_inv = stm.inverse_eigenvectors_q(c);

                // QP[a, b] = Σ_p U[a, p] · λ_p · exp(λ_p · t) · U⁻¹[p, b].
                // Build the diagonal "λ · exp(λ·t)" once per site, then a
                // single k × k × k product.
                VecT diag_term(k);
                for (std::size_t p = 0; p < k; ++p) {
                    diag_term[p] = lambda[p] *
                                   std::exp(Scalar(te) * lambda[p]);
                }
                QP = U_Q * diag_term.asDiagonal() * U_Q_inv;

                // log_sib[a] = Σ_{sibling z of v} edge_inside(z, a)
                for (std::size_t a = 0; a < k; ++a) log_sib[a] = 0.0;
                for (auto sib : t.tree.successors(w)) {
                    if (static_cast<std::size_t>(sib) == v) continue;
                    for (std::size_t a = 0; a < k; ++a) {
                        log_sib[a] += ei_ll(c, sib, a);
                    }
                }

                const double log_p_y_c = problem.log_p_y(ti, c);

                // Per-site contribution:
                //   Σ_{a,b} M(a) · Re(QP[a, b]) · I(b) / p(y_c)
                // where M(a) = exp(outside(w, a) + log_sib(a)) and
                //       I(b) = exp(inside(v, b)).
                //
                // We absorb the - log p(y_c) into the M-side log so that
                // every exp() is small in magnitude.
                double per_site = 0.0;
                for (std::size_t a = 0; a < k; ++a) {
                    const double M_a_log = outside_ll(c, w, a)
                                         + log_sib[a]
                                         - log_p_y_c;
                    const double M_a = std::exp(M_a_log);
                    if (M_a == 0.0) continue;
                    for (std::size_t b = 0; b < k; ++b) {
                        const double I_b = std::exp(inside_ll(c, v, b));
                        per_site += M_a * to_real(QP(a, b)) * I_b;
                    }
                }
                accum += per_site;
            }

            grad[out_idx++] = accum;
        }
    }

    return grad;
}

} // namespace optimize
} // namespace phylocore

#endif // PHYLOCORE_OPTIMIZE_BRANCH_GRADIENT_H
