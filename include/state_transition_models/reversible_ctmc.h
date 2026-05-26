#ifndef PHYLOCORE_STATE_TRANSITION_MODELS_REVERSIBLE_CTMC_H
#define PHYLOCORE_STATE_TRANSITION_MODELS_REVERSIBLE_CTMC_H

#include <variant>
#include <vector>

#include <Eigen/Dense>

#include "../state_transition_model.h"

namespace phylocore {

/*!
 * @brief Reversible CTMC substitution model (the GTR parameterization).
 *
 * Q is constructed from a symmetric exchangeability matrix S and stationary
 * frequencies π via Q = S · diag(π); detailed balance holds by construction,
 * so eigenvalues are real and A = diag(√π)·Q·diag(1/√π) is symmetric.
 * Eigendecomposition uses Eigen::SelfAdjointEigenSolver for numerical stability.
 *
 * Uses the default D = std::monostate (no per-node state).
 *
 * Set rates[c], freqs[c] per site, then call build() to:
 *   - compute the eigendecomposition cache per site
 *   - populate sites_latent_states with each site's alphabet_size
 *   - set root_prior = freqs (stationary distribution at root)
 *   - sync state_transition_parameters = [rates[0], freqs[0], rates[1], freqs[1], ...]
 *
 * Only the mandatory scalar primitive compute_log_transition_probability
 * is overridden. has_optimized_functions stays false; the traversal uses the
 * k²-scalar fallback. Pair with identity_observation_model<> and observed_data
 * of type vector<vector<int>>.
 *
 * Per-site parameterization:
 *   - rates[c]: k*(k-1)/2 exchangeability rates (upper triangle of symmetric S,
 *               row-major: (0,1), (0,2), ..., (0,k-1), (1,2), ..., (k-2,k-1))
 *   - freqs[c]: k stationary frequencies, must sum to 1
 *
 * To update parameters after construction, modify rates/freqs and call build() again.
 */
class reversible_ctmc_model : public state_transition_model<> {
public:
    std::vector<std::vector<double>> rates;  // [site][k*(k-1)/2]
    std::vector<std::vector<double>> freqs;  // [site][k], must sum to 1

    void build();

    double compute_log_transition_probability(
        std::monostate& data,
        size_t          site,
        double          branch_length,
        size_t          alpha,
        size_t          beta
    ) const override;

    // ── Public accessors for the cached symmetric eigendecomposition ─────────
    //
    // For site c with stationary distribution π and symmetric-similarity-
    // transformed rate matrix A = diag(√π) · Q · diag(1/√π), these expose:
    //   eigenvectors(c) : orthogonal U such that A = U · diag(λ) · Uᵀ
    //                     (so U_Q = diag(1/√π) · U,
    //                      and U_Q⁻¹ = Uᵀ · diag(√π))
    //   eigenvalues(c)  : λ (real, since A is symmetric)
    //   sqrt_pi(c)      : √π
    //   inv_sqrt_pi(c)  : 1/√π
    //
    // These are needed by optimize::hobolth_jensen_stats to compute
    // expected sufficient statistics for a CTMC EM step without
    // re-running the eigensolver.
    const Eigen::MatrixXd& eigenvectors(size_t site)  const { return cache[site].U;            }
    const Eigen::VectorXd& eigenvalues(size_t site)   const { return cache[site].lambda;       }
    const Eigen::VectorXd& sqrt_pi(size_t site)       const { return cache[site].sqrt_pi;      }
    const Eigen::VectorXd& inv_sqrt_pi(size_t site)   const { return cache[site].inv_sqrt_pi;  }

private:
    struct site_cache {
        Eigen::MatrixXd U;            // eigenvectors of symmetric similarity transform
        Eigen::VectorXd lambda;       // eigenvalues
        Eigen::VectorXd sqrt_pi;
        Eigen::VectorXd inv_sqrt_pi;
    };
    std::vector<site_cache> cache;  // one per site, filled by build()
};

} // namespace phylocore

#endif // PHYLOCORE_STATE_TRANSITION_MODELS_REVERSIBLE_CTMC_H
