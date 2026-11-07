#include "state_transition_models/ctmc.h"

#include <unsupported/Eigen/MatrixFunctions>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <limits>
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

    // Transitive closure of the positive off-diagonal graph. Matrix-exponential
    // entries are structurally zero exactly when no directed path connects the
    // corresponding states. Eigen's dense Pade/LU implementation is accurate,
    // but an arbitrary state permutation can introduce signed O(epsilon) fill
    // into such entries. Record the graph structure once per rebuild so the
    // transition path can remove that fill deterministically.
    transition_reachable.assign(k * k, 0);
    for (size_t i = 0; i < k; ++i) {
        transition_reachable[i * k + i] = 1;
        for (size_t j = 0; j < k; ++j) {
            if (i != j && Q(static_cast<Eigen::Index>(i),
                            static_cast<Eigen::Index>(j)) > 0.0) {
                transition_reachable[i * k + j] = 1;
            }
        }
    }
    for (size_t via = 0; via < k; ++via) {
        for (size_t i = 0; i < k; ++i) {
            if (!transition_reachable[i * k + via]) continue;
            for (size_t j = 0; j < k; ++j) {
                if (transition_reachable[via * k + j])
                    transition_reachable[i * k + j] = 1;
            }
        }
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

// ── Robust branch operator ────────────────────────────────────────────────────
//
// For irreducible Q, exp(t * branch_operator) is evaluated by Eigen's Pade
// scaling-and-squaring implementation. For a reducible weight-1 CTMC, use the
// non-negative uniformization implementation below: even after structural-zero
// masking, dense Pade arithmetic can round a genuinely reachable tiny value
// negative on a long branch. Finally apply the reachability mask in either case.

Eigen::MatrixXd ctmc_model::robust_transition(double t) const {
    const size_t k = static_cast<size_t>(branch_operator.rows());
    assert(transition_reachable.size() == k * k);

    bool reducible = false;
    for (unsigned char reachable : transition_reachable) {
        if (!reachable) {
            reducible = true;
            break;
        }
    }

    bool uniformization_safe = complete_data_weight == 1.0
                            && t >= 0.0
                            && t <= std::numeric_limits<double>::max();
    for (size_t i = 0; uniformization_safe && i < k; ++i) {
        if (Q(static_cast<Eigen::Index>(i),
              static_cast<Eigen::Index>(i)) > 0.0) {
            uniformization_safe = false;
        }
        for (size_t j = 0; uniformization_safe && j < k; ++j) {
            if (i != j && Q(static_cast<Eigen::Index>(i),
                            static_cast<Eigen::Index>(j)) < 0.0) {
                uniformization_safe = false;
            }
        }
    }

    Eigen::MatrixXd P = (reducible && uniformization_safe)
                            ? uniformization_transition(t)
                            : (branch_operator * t).exp();
    if (complete_data_weight == 1.0) {
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                if (!transition_reachable[i * k + j]) {
                    P(static_cast<Eigen::Index>(i),
                      static_cast<Eigen::Index>(j)) = 0.0;
                }
            }
        }
    }
    return P;
}

Eigen::MatrixXd ctmc_model::uniformization_transition(double t) const {
    const Eigen::Index n = Q.rows();
    const size_t k = static_cast<size_t>(n);
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    if (t == 0.0) return I;

    double nu = 0.0;
    for (Eigen::Index i = 0; i < n; ++i)
        nu = std::max(nu, -Q(i, i));
    if (!(nu > 0.0)) return I;

    // Scale until the Poisson mean of the base exponential is at most 1/2.
    // This keeps the series short and starts it at exp(-mu), far from
    // underflow. Squaring later restores the original branch length.
    double scaled_t = t;
    int squarings = 0;
    constexpr int max_squarings = 4096;
    while (nu * scaled_t > 0.5 && squarings < max_squarings) {
        scaled_t *= 0.5;
        ++squarings;
    }
    if (squarings == max_squarings)
        return (Q * t).exp();
    const double mu = nu * scaled_t;
    const Eigen::MatrixXd R = I + Q / nu;

    Eigen::MatrixXd power = I;
    double weight = std::exp(-mu);
    Eigen::MatrixXd P = weight * I;

    // A shortest directed path has at most k-1 edges, so do not terminate the
    // series before every representable reachable entry has had a chance to
    // appear. With mu <= 1/2, subsequent Poisson weights decrease monotonically.
    const size_t min_order = (k == 0) ? 0 : k - 1;
    const size_t max_order = std::max<size_t>(min_order, 2048);
    for (size_t order = 1; order <= max_order; ++order) {
        power = (power * R).eval();
        weight *= mu / static_cast<double>(order);
        P.noalias() += weight * power;

        if (weight == 0.0) break;
        const double next_weight =
            weight * mu / static_cast<double>(order + 1);
        if (order >= min_order) {
            // Since mu <= 1/2, the remaining Poisson weights are bounded by
            // a geometric series whose first term is next_weight. Every entry
            // of every power of the stochastic matrix R is at most one, so
            // that scalar tail also bounds the omitted error in each entry.
            const double next_ratio =
                mu / static_cast<double>(order + 2);
            const double tail_bound = next_weight / (1.0 - next_ratio);

            // A global Poisson-tail test is not enough here: a transition that
            // first appears at order k-1 can be tiny even while the matrix has
            // O(1) entries. Require relative accuracy for the smallest
            // reachable entry accumulated so far. If a reachable entry is
            // still zero, keep going until it appears or the weights underflow.
            double min_positive = std::numeric_limits<double>::max();
            bool all_reachable_positive = true;
            for (size_t i = 0; i < k; ++i) {
                for (size_t j = 0; j < k; ++j) {
                    if (!transition_reachable[i * k + j]) continue;
                    const double value = P(static_cast<Eigen::Index>(i),
                                           static_cast<Eigen::Index>(j));
                    if (value > 0.0) {
                        min_positive = std::min(min_positive, value);
                    } else {
                        all_reachable_positive = false;
                    }
                }
            }
            if (all_reachable_positive &&
                tail_bound <= std::numeric_limits<double>::epsilon()
                              * min_positive) {
                break;
            }
        }
    }

    // A CTMC transition matrix is row-stochastic. Restore that exact invariant
    // before and during squaring: otherwise an O(epsilon) row-sum error in the
    // base series can be amplified by 2^s when Q contains widely separated
    // rates and scaling requires many squarings. Row scaling preserves every
    // structural zero because the series above is non-negative.
    const auto normalize_rows = [](Eigen::MatrixXd& matrix) {
        for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
            const double row_sum = matrix.row(row).sum();
            if (!(row_sum > 0.0) || !std::isfinite(row_sum))
                throw std::runtime_error(
                    "ctmc_model::uniformization_transition: non-finite or "
                    "non-positive transition row sum");
            matrix.row(row) /= row_sum;
        }
    };
    normalize_rows(P);
    for (int i = 0; i < squarings; ++i) {
        P = (P * P).eval();
        normalize_rows(P);
    }
    return P;
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
            robust_transition(branch_length)(static_cast<Eigen::Index>(alpha),
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
        const Eigen::MatrixXd P = robust_transition(branch_length);
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
