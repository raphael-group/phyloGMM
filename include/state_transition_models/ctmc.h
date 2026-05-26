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

    // ── Uniform "of Q" accessors (model-agnostic; shared with reversible) ────
    //
    // For a general (possibly non-reversible) Q, the eigendecomposition is
    //     Q = U_Q · diag(λ_Q) · U_Q⁻¹
    // with U_Q, λ_Q, U_Q⁻¹ complex in general. ctmc_model maintains a single
    // shared cache (one Q across the tree), so the site argument is ignored.
    // These accessors mirror the ones on reversible_ctmc_model so that
    // templated callers (e.g. optimize::hobolth_jensen_stats) can deduce
    // the scalar type at compile time.
    const Eigen::MatrixXcd& eigenvectors_q(size_t /*site*/)         const { return cache.U;      }
    const Eigen::VectorXcd& eigenvalues_q(size_t /*site*/)          const { return cache.lambda; }
    const Eigen::MatrixXcd& inverse_eigenvectors_q(size_t /*site*/) const { return cache.U_inv;  }

    // Direct access to the underlying real rate matrix Q (used by callers
    // that need Q_ij directly, e.g. for posterior expected substitution
    // counts in hobolth_jensen_stats).
    const Eigen::MatrixXd& rate_matrix() const { return Q; }

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
