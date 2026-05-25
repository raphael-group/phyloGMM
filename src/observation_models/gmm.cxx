#include "observation_models/gmm.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace phylocore {

// ── build ─────────────────────────────────────────────────────────────────────

void gmm_observation_model::build() {
    if (mu.size() != Sigma.size())
        throw std::runtime_error(
            "gmm_observation_model::build: mu (size " +
            std::to_string(mu.size()) + ") and Sigma (size " +
            std::to_string(Sigma.size()) + ") must have the same length");
    if (mu.empty())
        throw std::runtime_error(
            "gmm_observation_model::build: mu/Sigma must be non-empty");
    if (d == 0)
        throw std::runtime_error(
            "gmm_observation_model::build: observation dimensionality d is zero");

    const size_t k = mu.size();
    chol.clear();
    chol.reserve(k);
    log_det.assign(k, 0.0);
    log_2pi_d = static_cast<double>(d) * std::log(2.0 * M_PI);

    for (size_t i = 0; i < k; ++i) {
        if (static_cast<size_t>(mu[i].size()) != d)
            throw std::runtime_error(
                "gmm_observation_model::build: mu[" + std::to_string(i) +
                "] has length " + std::to_string(mu[i].size()) +
                ", expected " + std::to_string(d));
        if (static_cast<size_t>(Sigma[i].rows()) != d
            || static_cast<size_t>(Sigma[i].cols()) != d)
            throw std::runtime_error(
                "gmm_observation_model::build: Sigma[" + std::to_string(i) +
                "] has shape " + std::to_string(Sigma[i].rows()) + "x" +
                std::to_string(Sigma[i].cols()) + ", expected " +
                std::to_string(d) + "x" + std::to_string(d));

        Eigen::LLT<Eigen::MatrixXd> llt(Sigma[i]);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error(
                "gmm_observation_model::build: Sigma[" + std::to_string(i) +
                "] is not positive definite");

        // log|Σ| = 2 * Σ_j log L_jj, where L is the Cholesky factor.
        log_det[i] = 2.0 * llt.matrixL()
                                .toDenseMatrix()
                                .diagonal()
                                .array()
                                .log()
                                .sum();
        chol.push_back(std::move(llt));
    }

    // observation_parameters: flatten [mu[0], vec(Sigma[0]), mu[1], vec(Sigma[1]), ...]
    observation_parameters.clear();
    observation_parameters.reserve(k * (d + d * d));
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < d; ++j)
            observation_parameters.push_back(mu[i][j]);
        for (size_t r = 0; r < d; ++r)
            for (size_t c = 0; c < d; ++c)
                observation_parameters.push_back(Sigma[i](r, c));
    }
}

// ── Scalar emission primitive ─────────────────────────────────────────────────
//
// log N(y; μ, Σ) = -0.5 * ( d·log(2π) + log|Σ| + (y-μ)ᵀ Σ⁻¹ (y-μ) ).
// We solve Σ z = (y-μ) via the cached Cholesky; the quadratic form is
// (y-μ)ᵀ z. `site` is unused — phylo-GMM uses one observation per leaf for
// the whole tree, not per site.

double gmm_observation_model::compute_log_emission_probability(
    std::monostate&,
    size_t                  alpha,
    size_t                  /*site*/,
    const Eigen::VectorXd&  observation
) const {
    const Eigen::VectorXd diff = observation - mu[alpha];
    const Eigen::VectorXd z    = chol[alpha].solve(diff);
    const double quad = diff.dot(z);
    return -0.5 * (log_2pi_d + log_det[alpha] + quad);
}

} // namespace phylocore
