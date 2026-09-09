#include "state_transition_models/ctmc.h"

#include <unsupported/Eigen/MatrixFunctions>

#include <algorithm>
#include <cassert>
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

    if (!std::isfinite(complete_data_weight) || complete_data_weight <= 0.0)
        throw std::runtime_error(
            "ctmc_model::build: complete_data_weight must be finite and > 0");

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
    Eigen::EigenSolver<Eigen::MatrixXd> rate_solver(
        Q, /* computeEigenvectors= */ true);
    if (rate_solver.info() != Eigen::Success)
        throw std::runtime_error(
            "ctmc_model::build: eigendecomposition of Q failed");

    rate_cache.U      = rate_solver.eigenvectors();
    rate_cache.lambda = rate_solver.eigenvalues();
    rate_cache.U_inv  = rate_cache.U.inverse();

    // Preserve the exact legacy path, including the same eigendecomposition,
    // when w_C == 1. For other weights, the powered complete-path density has
    // branch kernel exp(tA), A_ii=w_C Q_ii, A_ij=Q_ij^w_C.
    if (complete_data_weight == 1.0) {
        cache = rate_cache;
        branch_operator = Q;
    } else {
        Eigen::MatrixXd A(Q.rows(), Q.cols());
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                A(static_cast<Eigen::Index>(i),
                  static_cast<Eigen::Index>(j)) =
                    (i == j)
                        ? complete_data_weight * Q(
                              static_cast<Eigen::Index>(i),
                              static_cast<Eigen::Index>(i))
                        : std::pow(std::max(
                              0.0, Q(static_cast<Eigen::Index>(i),
                                     static_cast<Eigen::Index>(j))),
                              complete_data_weight);
                if (!std::isfinite(A(static_cast<Eigen::Index>(i),
                                     static_cast<Eigen::Index>(j))))
                    throw std::runtime_error(
                        "ctmc_model::build: weighted path operator contains "
                        "a non-finite entry");
            }
        }
        Eigen::EigenSolver<Eigen::MatrixXd> weighted_solver(
            A, /* computeEigenvectors= */ true);
        if (weighted_solver.info() != Eigen::Success)
            throw std::runtime_error(
                "ctmc_model::build: eigendecomposition of weighted path "
                "operator failed");
        cache.U      = weighted_solver.eigenvectors();
        cache.lambda = weighted_solver.eigenvalues();
        cache.U_inv  = cache.U.inverse();
        branch_operator = A;
    }

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

    // Optimised matvec is implemented below.
    has_optimized_functions = true;
}

// ── Pade branch operator ──────────────────────────────────────────────────────
//
// exp(t * branch_operator) by scaling-and-squaring. Unlike the eigendecomposition
// route this never forms a dense U / U^-1, so entries of exp(tQ) that are
// structurally zero (state j unreachable from i) come out as exactly 0.0 rather
// than +/-1e-15. That keeps the log-space guard below deterministic and makes
// the likelihood invariant to how the states happen to be numbered.

Eigen::MatrixXd ctmc_model::pade_transition(double t) const {
    return (branch_operator * t).exp();
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
    if (use_pade_expm) {
        const double prob_p =
            pade_transition(branch_length)(static_cast<Eigen::Index>(alpha),
                                           static_cast<Eigen::Index>(beta));
        return (prob_p > 0.0) ? std::log(prob_p) : NEGATIVE_INFINITY;
    }

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

// ── Optimised matvec ──────────────────────────────────────────────────────────
//
// Forward:   P(t) · exp(log_v)        via   U · diag(exp(λt)) · U⁻¹ · v
// Transpose: P(t)ᵀ · exp(log_v)       via   U⁻¹ᵀ · diag(exp(λt)) · Uᵀ · v
//
// We subtract max(log_v) before exponentiating to avoid underflow on deep
// trees and add it back after the log — exploits P(t) · (c·v) = c · P(t)v.
// Residual imaginary parts cancel exactly in exact arithmetic; we take .real()
// and clamp non-positive entries to NEGATIVE_INFINITY.

void ctmc_model::compute_log_pmatrix_vector_product(
    std::monostate& /*data*/,
    size_t /*site*/,
    double branch_length,
    const std::vector<double>& log_vector,
    std::vector<double>& result,
    bool transpose
) const {
    const Eigen::Index k = cache.lambda.size();
    assert(log_vector.size() >= static_cast<std::size_t>(k));
    assert(result.size()     >= static_cast<std::size_t>(k));

    double max_log = *std::max_element(log_vector.begin(),
                                       log_vector.begin() + k);

    if (use_pade_expm) {
        Eigen::VectorXd vr(k);
        for (Eigen::Index i = 0; i < k; ++i)
            vr[i] = std::exp(log_vector[i] - max_log);
        const Eigen::MatrixXd P = pade_transition(branch_length);
        Eigen::VectorXd rp = transpose ? (P.transpose() * vr).eval()
                                       : (P * vr).eval();
        for (Eigen::Index i = 0; i < k; ++i)
            result[i] = (rp[i] > 0.0) ? std::log(rp[i]) + max_log
                                      : NEGATIVE_INFINITY;
        return;
    }

    Eigen::VectorXcd v(k);
    for (Eigen::Index i = 0; i < k; ++i)
        v[i] = std::exp(log_vector[i] - max_log);

    // First multiply: U_inv·v (forward) or U.transpose()·v (transpose).
    Eigen::VectorXcd w = transpose
                       ? (cache.U.transpose() * v).eval()
                       : (cache.U_inv * v).eval();

    // Apply diag(exp(λ_p · t)).
    for (Eigen::Index p = 0; p < k; ++p)
        w[p] *= std::exp(cache.lambda[p] * branch_length);

    // Second multiply: U·w (forward) or U_inv.transpose()·w (transpose).
    Eigen::VectorXd res = transpose
                        ? (cache.U_inv.transpose() * w).real().eval()
                        : (cache.U * w).real().eval();

    for (Eigen::Index i = 0; i < k; ++i)
        result[i] = (res[i] > 0.0) ? std::log(res[i]) + max_log
                                   : NEGATIVE_INFINITY;
}

} // namespace phylocore
