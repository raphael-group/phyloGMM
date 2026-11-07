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
 * Numerical note: the eigendecomposition route rebuilds structural zeros of
 * exp(tQ) by cancellation between dense U and U^-1. For a reducible Q (many
 * off-diagonals exactly zero) those zeros come back as +/-1e-15, and the
 * log-space guard then maps them inconsistently to log(tiny) vs -inf depending
 * on state ordering; when U is additionally ill-conditioned the whole matrix is
 * unreliable. Set use_pade_expm = true to use the robust exponential backend:
 * scaling-and-squaring (Pade) generally, and non-negative uniformization for a
 * reducible weight-1 CTMC. A reachability mask preserves structural zeros.
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
 * Both the scalar primitive compute_log_transition_probability and the
 * optimized matvec compute_log_pmatrix_vector_product are overridden;
 * build() sets has_optimized_functions = true so the traversal makes one
 * backend matrix-vector call instead of repeatedly invoking the scalar path.
 * EM hooks (e_step/m_step) are intentionally not overridden — parameter
 * estimation is a separate, future change.
 *
 * complete_data_weight optionally powers the complete CTMC path density.
 * Its default 1.0 preserves the ordinary exp(tQ) model exactly.
 */
class ctmc_model : public state_transition_model<> {
public:
    Eigen::MatrixXd Q;  // k×k rate matrix; rows sum to zero, off-diag ≥ 0.

    // Power applied to the complete CTMC path density. At the default 1.0,
    // branches use exp(tQ). For w != 1, traversal uses exp(tA), where
    // A_ii = w Q_ii and A_ij = Q_ij^w (i != j). This is the exact marginal
    // kernel obtained by integrating [p_Q(path)]^w over branch paths.
    double complete_data_weight = 1.0;

    // When true, exp(t * Q) uses the robust matrix-exponential path rather than
    // the cached eigendecomposition. The primary backend is scaling-and-
    // squaring (Pade). For a reducible, weight-1 CTMC it uses non-negative
    // uniformization plus scaling-and-squaring, because dense Pade arithmetic
    // can change the sign of tiny reachable probabilities. A graph-reachability
    // mask then restores exact structural zeros. The eigendecomposition is
    // still built either way: hobolth_jensen_stats() consumes it.
    bool use_pade_expm = false;

    void build();

    double compute_log_transition_probability(
        std::monostate& data,
        size_t          site,
        double          branch_length,
        size_t          alpha,
        size_t          beta
    ) const override;

    // Optimised forward/transpose matvec via the selected exponential backend.
    // result[i] = log Σ_j P(t)_{i,j} · exp(log_v[j])         (transpose=false)
    // result[i] = log Σ_j P(t)_{j,i} · exp(log_v[j])         (transpose=true)
    void compute_log_pmatrix_vector_product(
        std::monostate&             data,
        size_t                      site,
        double                      branch_length,
        const std::vector<double>&  log_vector,
        std::vector<double>&        result,
        bool                        transpose
    ) const override;

    // ── Uniform "of Q" accessors (model-agnostic; shared with reversible) ────
    //
    // These accessors expose the active branch operator: Q at weight 1, or A
    // defined above otherwise. Hobolth-Jensen uses the same operator as the
    // inside/outside traversal, which is required for weighted-path EM.
    // These accessors mirror the ones on reversible_ctmc_model so that
    // templated callers (e.g. optimize::hobolth_jensen_stats) can deduce
    // the scalar type at compile time.
    const Eigen::MatrixXcd& eigenvectors_q(size_t /*site*/)         const { return cache.U;      }
    const Eigen::VectorXcd& eigenvalues_q(size_t /*site*/)          const { return cache.lambda; }
    const Eigen::MatrixXcd& inverse_eigenvectors_q(size_t /*site*/) const { return cache.U_inv;  }

    // The eigendecomposition of the underlying rate generator Q is kept
    // separately because stationary root probabilities must be derived from
    // Q, never from the generally non-conservative weighted operator A.
    const Eigen::MatrixXcd& rate_eigenvectors_q() const { return rate_cache.U; }
    const Eigen::VectorXcd& rate_eigenvalues_q() const { return rate_cache.lambda; }
    const Eigen::MatrixXcd& inverse_rate_eigenvectors_q() const {
        return rate_cache.U_inv;
    }

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
    eigen_cache cache;       // active branch operator used by traversal
    eigen_cache rate_cache;  // unweighted generator Q

    // Real branch operator matching `cache` (Q when complete_data_weight == 1,
    // else the weighted path operator A). Used only by the robust path.
    Eigen::MatrixXd branch_operator;

    // Row-major transitive closure of the directed positive-rate graph.
    // transition_reachable[i*k+j] is true exactly when state j is reachable
    // from state i (including i == j). It is rebuilt whenever Q is rebuilt and
    // is used to restore structural zeros in the weight-1 robust path.
    std::vector<unsigned char> transition_reachable;

    // exp(t * branch_operator) via the selected robust backend. For weight 1,
    // structurally impossible transitions are masked exactly.
    Eigen::MatrixXd robust_transition(double t) const;

    // Cancellation-free exp(tQ) for a weight-1 CTMC. The scaled base
    // exponential is evaluated as a non-negative uniformization series and
    // then repeatedly squared.
    Eigen::MatrixXd uniformization_transition(double t) const;
};

} // namespace phylocore

#endif // PHYLOCORE_STATE_TRANSITION_MODELS_CTMC_H
