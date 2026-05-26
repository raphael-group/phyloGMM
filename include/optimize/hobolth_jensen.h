#ifndef PHYLOCORE_OPTIMIZE_HOBOLTH_JENSEN_H
#define PHYLOCORE_OPTIMIZE_HOBOLTH_JENSEN_H

// Header-only Hobolth-Jensen expected-sufficient-statistics helper for a
// general CTMC on a (rooted) tree. No IPOPT dependency.
//
// For each edge e = (parent w → child u) with branch length t_e, the helper
// computes the expected number of substitutions E[N_ij | y]_e from state i to
// state j (i ≠ j) and the expected dwell time E[T_i | y]_e in state i,
// integrating analytically over the unobserved trajectory of the latent
// CTMC conditional on the observation y. It then sums these contributions
// across (1) every edge of every tree in problem.trees and (2) every site
// — yielding two aggregate suff-stat tensors suitable for the closed-form
// CTMC M-step.
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
// This helper is templated on the scalar type deduced from the model's
// eigendecomposition of Q (real for reversible_ctmc_model, complex for the
// general ctmc_model). With Q = U_Q · diag(λ_Q) · U_Q⁻¹:
//
//   W_e[a,b] = exp(outside(parent, a) + sibling_edge_inside(a)
//                 + inside(child, b) − log p(y))
//   G_e      = U_Qᵀ · W_e · (U_Q⁻¹)ᵀ
//   H        = Σ_e  G_e ⊙ J_e          where J_e[p,q] = J_{pq}(t_e)
//   C        = (U_Q⁻¹)ᵀ · H · U_Qᵀ
//
//   E[T_i  | y]_e  = C[i, i]
//   E[N_ij | y]_e  = Q_ij · C[i, j]    (i ≠ j)
//
// For reversible models the symmetric similarity transform A = D·Q·D⁻¹ has
// orthogonal eigenvectors U_A, and U_Q = D⁻¹·U_A, U_Q⁻¹ = U_Aᵀ·D, λ_Q = λ_A,
// all real. The helper consumes these uniformly via the model's
// eigenvectors_q / eigenvalues_q / inverse_eigenvectors_q accessors.
//
// Model API contract (duck-typed via eigendecomp accessors):
//   stm.eigenvectors_q(site_c)         -> const T_matrix& (U_Q)
//   stm.eigenvalues_q(site_c)          -> const T_vector& (λ_Q)
//   stm.inverse_eigenvectors_q(site_c) -> const T_matrix& (U_Q⁻¹)
// where T_matrix / T_vector are Eigen::Matrix<Scalar,...> with Scalar
// either double or std::complex<double>.

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
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

// ── Scalar extraction helpers ────────────────────────────────────────────────
//
// to_real(x): extract the real part regardless of whether the scalar is real
// or complex. Used when collapsing complex aggregates onto the real-valued
// hobolth_jensen_aggregate at the end of the math pipeline.

inline double to_real(double x) { return x; }
inline double to_real(const std::complex<double>& x) { return x.real(); }

/*!
 * @brief Numerically stable sinhc kernel for the Hobolth-Jensen J-integral.
 *
 *   J_{pq}(t) = t · exp(t(λ_p+λ_q)/2) · sinhc(t(λ_p-λ_q)/2)
 *
 * Uses a small-|x| threshold (|x| < 1e-8) to return 1 directly, avoiding
 * 0/0 when λ_p ≈ λ_q. Templated on Scalar so both real (double) and complex
 * (std::complex<double>) eigenvalues are handled — std::sinh / std::exp /
 * std::abs have appropriate complex overloads.
 */
template <typename Scalar>
inline Scalar hobolth_jensen_kernel(Scalar lambda_p,
                                    Scalar lambda_q,
                                    double t)
{
    const Scalar half_sum  = Scalar(0.5) * Scalar(t) * (lambda_p + lambda_q);
    const Scalar half_diff = Scalar(0.5) * Scalar(t) * (lambda_p - lambda_q);
    Scalar sinhc;
    if (std::abs(half_diff) < 1e-8) {
        sinhc = Scalar(1.0);
    } else {
        sinhc = std::sinh(half_diff) / half_diff;
    }
    return Scalar(t) * std::exp(half_sum) * sinhc;
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
 * The problem's stm must expose the uniform "of Q" eigendecomposition API
 * (eigenvectors_q / eigenvalues_q / inverse_eigenvectors_q). The scalar type
 * (double or std::complex<double>) is deduced from those accessors at compile
 * time, so reversible_ctmc_model uses real arithmetic and ctmc_model uses
 * complex arithmetic without runtime dispatch.
 *
 * @tparam Problem joint_optimization_problem<Model, ...> subclass with the
 *                 standard buffer + posterior accessors. Model must implement
 *                 the eigendecomposition-of-Q accessors above.
 */
template <typename Problem>
hobolth_jensen_aggregate hobolth_jensen_stats(Problem& problem)
{
    auto& stm = *problem.stm;

    // ── Deduce scalar / matrix / vector types from the model ────────────────
    using EigVecT  = std::decay_t<decltype(stm.eigenvalues_q(0))>;
    using Scalar   = typename EigVecT::Scalar;
    using MatT     = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VecT     = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    static_assert(
        std::is_same<Scalar, double>::value ||
        std::is_same<Scalar, std::complex<double>>::value,
        "hobolth_jensen_stats: model scalar must be double or std::complex<double>");

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

    // Per-site scratch reused inside the inner loop. Real W_e is the natural
    // form (it comes from exp(log-likelihoods)); we cast it to Scalar before
    // entering the eigenspace math so complex models share the pipeline.
    Eigen::MatrixXd W_e(k, k);
    MatT            W_e_typed(k, k);
    MatT            G_e(k, k);
    MatT            J_e(k, k);
    MatT            H(k, k);
    MatT            C(k, k);
    std::vector<double> log_sib(k);

    for (std::size_t ti = 0; ti < problem.trees.size(); ++ti) {
        const tree& t           = *problem.trees[ti];
        const auto& inside_ll   = problem.inside_buffer(ti);
        const auto& ei_ll       = problem.edge_inside_buffer(ti);
        const auto& outside_ll  = problem.outside_buffer(ti);

        for (std::size_t c = 0; c < num_sites; ++c) {
            const MatT& U_Q     = stm.eigenvectors_q(c);
            const VecT& lambda  = stm.eigenvalues_q(c);
            const MatT& U_Q_inv = stm.inverse_eigenvectors_q(c);

            // Q[i, j] = (U_Q · diag(λ) · U_Q⁻¹)[i, j]. For reversible models
            // this is a real matrix; for general ctmc_model the imaginary
            // parts cancel modulo numerical noise. We always materialize a
            // real Q so we can multiply it cleanly against the (real)
            // N_total aggregate later. The product is k×k → O(k^3),
            // negligible for typical k.
            MatT Q_typed = U_Q * lambda.asDiagonal() * U_Q_inv;
            Eigen::MatrixXd Q(k, k);
            for (std::size_t i = 0; i < k; ++i)
                for (std::size_t j = 0; j < k; ++j)
                    Q(i, j) = to_real(Q_typed(i, j));

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

                // Promote real W_e into the (possibly complex) Scalar field.
                W_e_typed = W_e.template cast<Scalar>();

                // G_e = U_Qᵀ · W_e · (U_Q⁻¹)ᵀ
                G_e = U_Q.transpose() * W_e_typed * U_Q_inv.transpose();

                // J_e[p, q] = kernel(λ_p, λ_q, t_e)
                for (std::size_t p = 0; p < k; ++p) {
                    for (std::size_t q = 0; q < k; ++q) {
                        J_e(p, q) = hobolth_jensen_kernel<Scalar>(
                            lambda[p], lambda[q], t_e);
                    }
                }

                H += G_e.cwiseProduct(J_e);
            }

            // C = (U_Q⁻¹)ᵀ · H · U_Qᵀ
            C = U_Q_inv.transpose() * H * U_Q.transpose();

            for (std::size_t i = 0; i < k; ++i)
                agg.T_total[i] += to_real(C(i, i));

            for (std::size_t i = 0; i < k; ++i)
                for (std::size_t j = 0; j < k; ++j)
                    if (i != j)
                        agg.N_total(i, j) += Q(i, j) * to_real(C(i, j));
        }
    }

    return agg;
}

} // namespace optimize
} // namespace phylocore

#endif // PHYLOCORE_OPTIMIZE_HOBOLTH_JENSEN_H
