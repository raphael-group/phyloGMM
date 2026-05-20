#include "state_transition_models/gtr.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "constants.h"

namespace phylocore {

// ── build ─────────────────────────────────────────────────────────────────────

void gtr_model::build() {
    size_t num_sites = freqs.size();
    sites_latent_states.resize(num_sites);
    cache.resize(num_sites);

    for (size_t c = 0; c < num_sites; ++c) {
        size_t k = freqs[c].size();
        sites_latent_states[c].alphabet_size = k;

        // Symmetric exchangeability matrix S from upper-triangle rates
        Eigen::MatrixXd S = Eigen::MatrixXd::Zero(k, k);
        size_t r = 0;
        for (size_t i = 0; i < k; ++i)
            for (size_t j = i + 1; j < k; ++j)
                S(i, j) = S(j, i) = rates[c][r++];

        // Stationary frequencies
        Eigen::VectorXd pi(k);
        for (size_t i = 0; i < k; ++i) pi[i] = freqs[c][i];

        // Q = S · diag(π); off-diagonal Q_ij = S_ij · π_j; diagonal = -(row sum)
        Eigen::MatrixXd Q = S * pi.asDiagonal();
        for (size_t i = 0; i < k; ++i)
            Q(i, i) = -Q.row(i).sum();

        // Normalize so expected substitutions per unit time = 1
        double mu = -(pi.transpose() * Q.diagonal()).value();
        if (mu <= 0.0)
            throw std::runtime_error(
                "gtr_model: zero total rate at site " + std::to_string(c));
        Q /= mu;

        // Symmetric similarity transform: A = diag(√π) · Q · diag(1/√π)
        // A is symmetric, so use SelfAdjointEigenSolver for stability.
        Eigen::VectorXd sqrt_pi     = pi.array().sqrt();
        Eigen::VectorXd inv_sqrt_pi = sqrt_pi.array().inverse();
        Eigen::MatrixXd A = sqrt_pi.asDiagonal() * Q * inv_sqrt_pi.asDiagonal();

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error(
                "gtr_model: eigendecomp failed at site " + std::to_string(c));

        cache[c] = { solver.eigenvectors(), solver.eigenvalues(),
                     sqrt_pi, inv_sqrt_pi };
    }

    // Sync flat parameters: [rates[0], freqs[0], rates[1], freqs[1], ...]
    state_transition_parameters.clear();
    for (size_t c = 0; c < num_sites; ++c) {
        state_transition_parameters.insert(state_transition_parameters.end(),
            rates[c].begin(), rates[c].end());
        state_transition_parameters.insert(state_transition_parameters.end(),
            freqs[c].begin(), freqs[c].end());
    }

    // Root prior = stationary distribution
    root_prior = freqs;
}

// ── Scalar primitive ──────────────────────────────────────────────────────────
//
// P(t) = diag(1/√π) · U · diag(exp(λt)) · Uᵀ · diag(√π)
// =>
// P_αβ(t) = (1/√π_α) · √π_β · Σ_p U[α,p] · exp(λ_p · t) · U[β,p]
//
// (using Uᵀ[p, β] = U[β, p], which holds because A is symmetric → U is orthogonal.)

double gtr_model::compute_log_transition_probability(
    std::monostate&,
    size_t c,
    double branch_length,
    size_t alpha,
    size_t beta
) const {
    const auto& ch = cache[c];
    double sum = 0.0;
    for (Eigen::Index p = 0; p < ch.lambda.size(); ++p) {
        sum += ch.U(alpha, p)
             * std::exp(ch.lambda(p) * branch_length)
             * ch.U(beta, p);
    }
    sum *= ch.inv_sqrt_pi(alpha) * ch.sqrt_pi(beta);
    return (sum > 0.0) ? std::log(sum) : NEGATIVE_INFINITY;
}

} // namespace phylocore
