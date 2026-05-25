#ifndef PHYLOCORE_STATE_TRANSITION_MODELS_CTMC_H
#define PHYLOCORE_STATE_TRANSITION_MODELS_CTMC_H

#include <complex>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include "../state_transition_model.h"

namespace phylocore {

/*!
 * @brief General continuous-time Markov chain (CTMC) state transition model.
 *
 * Latent states X_v ∈ {1,…,k} evolve along a tree via a CTMC with a general
 * (possibly non-reversible) k×k rate matrix Q. Per-edge transition is
 *   P^(e)_ij = [exp(b_e * Q)]_ij
 * computed in closed form from the eigendecomposition Q = U · diag(λ) · U⁻¹:
 *   P_αβ(t) = Σ_p U[α, p] · exp(λ_p · t) · U⁻¹[p, β].
 * Q is not assumed reversible, so eigenvalues/vectors are generally complex;
 * the real part of the sum is taken and log'd.
 *
 * Uses the default D = std::monostate (no per-node state).
 *
 * Usage:
 *   1. Set Q (k×k rate matrix; rows sum to zero, off-diagonal ≥ 0).
 *   2. Optionally set root_prior via set_root_prior(); otherwise build()
 *      installs the uniform distribution.
 *   3. Call build() to validate Q and compute the cached eigendecomposition.
 *
 * sites_latent_states is populated with a single entry {k, {}, 0}, matching
 * the paper's assumption of one shared Q across the (single) site. Callers
 * pass site = 0.
 *
 * Only the mandatory scalar primitive compute_log_transition_probability is
 * overridden. has_optimized_functions stays false; the traversal uses the
 * k²-scalar fallback. EM hooks (e_step/m_step) are intentionally not
 * overridden — parameter estimation is a separate, future change.
 */
class ctmc_model : public state_transition_model<> {
public:
    Eigen::MatrixXd Q;  // k×k rate matrix; rows sum to zero, off-diag ≥ 0.

    void build();

    double compute_log_transition_probability(
        std::monostate& data,
        size_t          site,
        double          branch_length,
        size_t          alpha,
        size_t          beta
    ) const override;

private:
    // Single shared cache: one Q applies across the whole tree.
    struct eigen_cache {
        Eigen::MatrixXcd U;       // k×k eigenvectors (columns)
        Eigen::VectorXcd lambda;  // k eigenvalues (generally complex)
        Eigen::MatrixXcd U_inv;   // k×k inverse of U
    };
    eigen_cache cache;
};

} // namespace phylocore

#endif // PHYLOCORE_STATE_TRANSITION_MODELS_CTMC_H
