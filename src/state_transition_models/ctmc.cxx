#include "state_transition_models/ctmc.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

#include "constants.h"

namespace phylocore {

// ── build ─────────────────────────────────────────────────────────────────────

void ctmc_model::build() {
    if (Q.rows() == 0 || Q.cols() == 0)
        throw std::runtime_error("ctmc_model::build: Q is empty");
    if (Q.rows() != Q.cols())
        throw std::runtime_error(
            "ctmc_model::build: Q is not square (" +
            std::to_string(Q.rows()) + "x" + std::to_string(Q.cols()) + ")");

    const size_t k = static_cast<size_t>(Q.rows());

    // Validate Q: rows sum to zero, off-diagonal entries non-negative.
    constexpr double row_sum_tol = 1e-8;
    constexpr double off_diag_tol = 1e-10;
    for (size_t i = 0; i < k; ++i) {
        double row_sum = Q.row(i).sum();
        if (std::abs(row_sum) > row_sum_tol)
            throw std::runtime_error(
                "ctmc_model::build: row " + std::to_string(i) +
                " of Q sums to " + std::to_string(row_sum) + ", expected 0");
        for (size_t j = 0; j < k; ++j) {
            if (i != j && Q(i, j) < -off_diag_tol)
                throw std::runtime_error(
                    "ctmc_model::build: Q(" + std::to_string(i) + "," +
                    std::to_string(j) + ") = " + std::to_string(Q(i, j)) +
                    " is negative (off-diagonal entries must be >= 0)");
        }
    }

    // General eigendecomposition — eigenvalues may be complex since Q
    // is not assumed reversible.
    Eigen::EigenSolver<Eigen::MatrixXd> solver(Q, /* computeEigenvectors= */ true);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error(
            "ctmc_model::build: eigendecomposition of Q failed");

    cache.U      = solver.eigenvectors();
    cache.lambda = solver.eigenvalues();
    cache.U_inv  = cache.U.inverse();

    // Single shared site_space — the paper assumes one Q across the tree.
    sites_latent_states = {{ k, {}, 0 }};

    // Root prior: leave caller-set values alone; otherwise default to uniform.
    if (root_prior.empty()
        || root_prior.size() != 1
        || root_prior[0].size() != k)
    {
        root_prior = { std::vector<double>(k, 1.0 / static_cast<double>(k)) };
    }

    // Sync flat parameters: row-major flatten of Q.
    state_transition_parameters.clear();
    state_transition_parameters.reserve(k * k);
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < k; ++j)
            state_transition_parameters.push_back(Q(i, j));
}

// ── Scalar primitive ──────────────────────────────────────────────────────────
//
// Equation 22: P_αβ(t) = Σ_p U[α, p] · exp(λ_p · t) · U⁻¹[p, β].
// Q may have complex eigenvalues/vectors; the imaginary parts cancel in the
// sum, leaving a real probability. We take the real part to drop residual
// imaginary noise, then log it. Numerical artifacts can produce small
// negative values; we map those (and zero) to NEGATIVE_INFINITY.

double ctmc_model::compute_log_transition_probability(
    std::monostate&,
    size_t /*site*/,
    double branch_length,
    size_t alpha,
    size_t beta
) const {
    const Eigen::Index k = cache.lambda.size();
    std::complex<double> sum(0.0, 0.0);
    for (Eigen::Index p = 0; p < k; ++p) {
        sum += cache.U(static_cast<Eigen::Index>(alpha), p)
             * std::exp(cache.lambda(p) * branch_length)
             * cache.U_inv(p, static_cast<Eigen::Index>(beta));
    }
    double prob = sum.real();
    return (prob > 0.0) ? std::log(prob) : NEGATIVE_INFINITY;
}

} // namespace phylocore
