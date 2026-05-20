#include "state_transition_models/gtr.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace phylocore {

// ── build ─────────────────────────────────────────────────────────────────────

void gtr_model::build() {
    size_t num_chars = freqs.size();
    sites_latent_states.resize(num_chars);
    cache.resize(num_chars);

    for (size_t c = 0; c < num_chars; ++c) {
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

        // Rate matrix Q: off-diagonal Q_ij = S_ij * pi_j, diagonal = -row sum
        // (S diagonal is zero, so Q row sum before fixing diagonal = sum of off-diag)
        Eigen::MatrixXd Q = S * pi.asDiagonal();
        for (size_t i = 0; i < k; ++i)
            Q(i, i) = -Q.row(i).sum();

        // Normalize so expected substitutions per unit time = 1
        double mu = -(pi.transpose() * Q.diagonal()).value();
        if (mu <= 0.0)
            throw std::runtime_error(
                "GTR: zero total rate for character " + std::to_string(c));
        Q /= mu;

        // Symmetric transformation: A = diag(sqrt_pi) * Q * diag(inv_sqrt_pi)
        // A is real symmetric — use SelfAdjointEigenSolver for stability
        Eigen::VectorXd sqrt_pi     = pi.array().sqrt();
        Eigen::VectorXd inv_sqrt_pi = sqrt_pi.array().inverse();

        Eigen::MatrixXd A = sqrt_pi.asDiagonal() * Q * inv_sqrt_pi.asDiagonal();

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error(
                "GTR: eigendecomposition failed for character " + std::to_string(c));

        cache[c] = { solver.eigenvectors(), solver.eigenvalues(), sqrt_pi, inv_sqrt_pi };
    }

    // Sync to flat parameters vector: [rates[0], freqs[0], rates[1], freqs[1], ...]
    parameters.clear();
    for (size_t c = 0; c < num_chars; ++c) {
        parameters.insert(parameters.end(), rates[c].begin(), rates[c].end());
        parameters.insert(parameters.end(), freqs[c].begin(), freqs[c].end());
    }

    // Root prior = stationary distribution.
    root_prior = freqs;
}

// ── helpers ───────────────────────────────────────────────────────────────────

Eigen::VectorXd gtr_model::apply_pmatrix(
    const character_cache& ch, double blen,
    const std::vector<double>& log_v) const
{
    size_t k = log_v.size();
    Eigen::VectorXd v(k);
    for (size_t i = 0; i < k; ++i) v[i] = std::exp(log_v[i]);

    // P(t) = diag(inv_sqrt_pi) * U * diag(exp(λt)) * U^T * diag(sqrt_pi)
    return ch.inv_sqrt_pi.cwiseProduct(
        ch.U * ((ch.lambda.array() * blen).exp().matrix().cwiseProduct(
            ch.U.transpose() * ch.sqrt_pi.cwiseProduct(v))));
}

Eigen::VectorXd gtr_model::apply_pmatrix_transpose(
    const character_cache& ch, double blen,
    const std::vector<double>& log_v) const
{
    size_t k = log_v.size();
    Eigen::VectorXd v(k);
    for (size_t i = 0; i < k; ++i) v[i] = std::exp(log_v[i]);

    // P(t)^T = diag(sqrt_pi) * U * diag(exp(λt)) * U^T * diag(inv_sqrt_pi)
    return ch.sqrt_pi.cwiseProduct(
        ch.U * ((ch.lambda.array() * blen).exp().matrix().cwiseProduct(
            ch.U.transpose() * ch.inv_sqrt_pi.cwiseProduct(v))));
}

// ── required overrides ────────────────────────────────────────────────────────

void gtr_model::compute_log_pmatrix_vector_product(
    gtr_node_data&, size_t c, double blen,
    const std::vector<double>& log_v, std::vector<double>& result) const
{
    Eigen::VectorXd res = apply_pmatrix(cache[c], blen, log_v);
    for (size_t i = 0; i < sites_latent_states[c].alphabet_size; ++i)
        result[i] = (res[i] > 0.0) ? std::log(res[i]) : NEGATIVE_INFINITY;
}

void gtr_model::compute_log_pmatrix_transpose_vector_product(
    gtr_node_data&, size_t c, double blen,
    const std::vector<double>& log_v, std::vector<double>& result) const
{
    Eigen::VectorXd res = apply_pmatrix_transpose(cache[c], blen, log_v);
    for (size_t i = 0; i < sites_latent_states[c].alphabet_size; ++i)
        result[i] = (res[i] > 0.0) ? std::log(res[i]) : NEGATIVE_INFINITY;
}

} // namespace phylocore
