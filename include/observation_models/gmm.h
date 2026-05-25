#ifndef PHYLOCORE_OBSERVATION_MODELS_GMM_H
#define PHYLOCORE_OBSERVATION_MODELS_GMM_H

#include <variant>
#include <vector>

#include <Eigen/Dense>

#include "../observation_model.h"

namespace phylocore {

/*!
 * @brief Gaussian-mixture emission model (one Gaussian per latent state).
 *
 * Per-leaf emission: y_ℓ | X_ℓ = i  ~  N(μ_i, Σ_i),  μ_i ∈ ℝ^d,
 *                                                     Σ_i ∈ ℝ^(d×d) SPD.
 *
 * Uses the default D = std::monostate; the per-leaf observation type O is
 * Eigen::VectorXd (a single d-dim continuous observation per leaf).
 *
 * The paper's phylo-GMM treats observations as per-leaf, not per-site, so
 * num_sites = 1 in practice. The `site` parameter is accepted per the
 * observation_model interface but is unused inside the body.
 *
 * Usage:
 *   1. Set mu (k means of length d), Sigma (k SPD covariances, each d×d), d.
 *   2. Call build() to validate the Sigma[i] are positive definite and to
 *      cache their Cholesky factorizations and log-determinants.
 */
class gmm_observation_model
    : public observation_model<std::monostate, Eigen::VectorXd>
{
public:
    std::vector<Eigen::VectorXd> mu;     // k means, each of length d
    std::vector<Eigen::MatrixXd> Sigma;  // k SPD covariance matrices, each d×d
    size_t                       d = 0;  // observation dimensionality

    void build();

    double compute_log_emission_probability(
        std::monostate&         data,
        size_t                  alpha,
        size_t                  site,
        const Eigen::VectorXd&  observation
    ) const override;

private:
    std::vector<Eigen::LLT<Eigen::MatrixXd>> chol;     // Cholesky of each Sigma[i]
    std::vector<double>                      log_det;  // log|Sigma[i]| per component
    double                                   log_2pi_d = 0.0; // d * log(2π) precomputed
};

} // namespace phylocore

#endif // PHYLOCORE_OBSERVATION_MODELS_GMM_H
