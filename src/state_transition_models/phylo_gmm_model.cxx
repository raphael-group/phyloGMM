#include "state_transition_models/phylo_gmm_model.h"
#include "phylogeny.h"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "constants.h"

namespace phylocore {

// ── Construction ──────────────────────────────────────────────────────────────

phylo_gmm_model::phylo_gmm_model(
    Eigen::MatrixXd              Q_,
    std::vector<Eigen::VectorXd> mu_,
    std::vector<Eigen::MatrixXd> Sigma_,
    Eigen::VectorXd              pi_,
    Eigen::MatrixXd              Y_)
    : k(Q_.rows()), d(Y_.cols()),
      Q(std::move(Q_)), mu(std::move(mu_)), Sigma(std::move(Sigma_)),
      pi(std::move(pi_)), Y(std::move(Y_))
{
    build();
}

// ── Convenience initializer ───────────────────────────────────────────────────

phylo_gmm_model phylo_gmm_model::init_kmeans(
    size_t k_, Eigen::MatrixXd Y_, unsigned seed)
{
    size_t n  = Y_.rows();
    size_t d_ = Y_.cols();

    std::mt19937 rng(seed);

    // k-means++ centroid initialization
    std::vector<Eigen::VectorXd> centroids;
    centroids.reserve(k_);
    {
        std::uniform_int_distribution<size_t> uid(0, n - 1);
        centroids.push_back(Y_.row(uid(rng)).transpose());
    }
    while (centroids.size() < k_) {
        Eigen::VectorXd dists(n);
        for (size_t j = 0; j < n; ++j) {
            double min_d2 = std::numeric_limits<double>::max();
            for (const auto& c : centroids) {
                double d2 = (Y_.row(j).transpose() - c).squaredNorm();
                if (d2 < min_d2) min_d2 = d2;
            }
            dists[j] = min_d2;
        }
        std::discrete_distribution<size_t> dd(dists.data(), dists.data() + n);
        centroids.push_back(Y_.row(dd(rng)).transpose());
    }

    // k-means iterations
    std::vector<int> assign(n, 0);
    for (int iter = 0; iter < 100; ++iter) {
        for (size_t j = 0; j < n; ++j) {
            double best = std::numeric_limits<double>::max();
            for (size_t ci = 0; ci < k_; ++ci) {
                double d2 = (Y_.row(j).transpose() - centroids[ci]).squaredNorm();
                if (d2 < best) { best = d2; assign[j] = (int)ci; }
            }
        }
        std::vector<Eigen::VectorXd> sums(k_, Eigen::VectorXd::Zero(d_));
        std::vector<int> counts(k_, 0);
        for (size_t j = 0; j < n; ++j) {
            sums[assign[j]] += Y_.row(j).transpose();
            counts[assign[j]]++;
        }
        for (size_t ci = 0; ci < k_; ++ci)
            if (counts[ci] > 0) centroids[ci] = sums[ci] / counts[ci];
    }

    // Build mu and Sigma from final assignments
    std::vector<Eigen::VectorXd> mu_(k_);
    std::vector<Eigen::MatrixXd> Sigma_(k_);
    std::vector<int> counts(k_, 0);
    for (size_t j = 0; j < n; ++j) counts[assign[j]]++;

    for (size_t ci = 0; ci < k_; ++ci) {
        mu_[ci] = centroids[ci];
        Eigen::MatrixXd S = Eigen::MatrixXd::Zero(d_, d_);
        for (size_t j = 0; j < n; ++j) {
            if (assign[j] == (int)ci) {
                Eigen::VectorXd diff = Y_.row(j).transpose() - mu_[ci];
                S += diff * diff.transpose();
            }
        }
        // Ridge regularization (1e-4 * I) ensures PD even when a cluster has
        // 0 or 1 point — common on small datasets.
        if (counts[ci] > 1)
            Sigma_[ci] = S / (double)counts[ci] + 1e-4 * Eigen::MatrixXd::Identity(d_, d_);
        else
            Sigma_[ci] = Eigen::MatrixXd::Identity(d_, d_);
    }

    // Q = -I + (1/k) * 11^T  — uniform off-diagonal rates, rows sum to zero
    Eigen::MatrixXd Q_ = Eigen::MatrixXd::Constant(k_, k_, 1.0 / k_);
    Q_.diagonal().setConstant(-(1.0 - 1.0 / k_));

    Eigen::VectorXd pi_ = Eigen::VectorXd::Constant(k_, 1.0 / k_);

    return phylo_gmm_model(
        std::move(Q_), std::move(mu_), std::move(Sigma_), std::move(pi_), std::move(Y_));
}

// ── build ─────────────────────────────────────────────────────────────────────

void phylo_gmm_model::build() {
    // Validate Q
    for (size_t i = 0; i < k; ++i) {
        double row_sum = Q.row(i).sum();
        if (std::abs(row_sum) > 1e-8)
            throw std::runtime_error(
                "phylo_gmm_model: row " + std::to_string(i) +
                " of Q sums to " + std::to_string(row_sum) + ", expected 0");
        for (size_t j = 0; j < k; ++j) {
            if (i != j && Q(i, j) < -1e-10)
                throw std::runtime_error(
                    "phylo_gmm_model: Q(" + std::to_string(i) + "," +
                    std::to_string(j) + ") = " + std::to_string(Q(i, j)) +
                    " is negative (off-diagonal entries must be >= 0)");
        }
    }

    // General eigendecomposition — eigenvalues may be complex
    Eigen::EigenSolver<Eigen::MatrixXd> solver(Q, /* computeEigenvectors= */ true);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("phylo_gmm_model: eigendecomposition of Q failed");

    ecache.lambda = solver.eigenvalues();
    ecache.U      = solver.eigenvectors();
    ecache.U_inv  = ecache.U.inverse();

    // Gaussian cache: Cholesky of each Sigma[i] and log-determinant
    gcache.chol.clear();
    gcache.chol.reserve(k);
    gcache.log_det.resize(k);
    gcache.log_2pi_d = (double)d * std::log(2.0 * M_PI);

    for (size_t i = 0; i < k; ++i) {
        Eigen::LLT<Eigen::MatrixXd> llt(Sigma[i]);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error(
                "phylo_gmm_model: Sigma[" + std::to_string(i) +
                "] is not positive definite");
        gcache.log_det[i] = 2.0 * llt.matrixL()
                                      .toDenseMatrix()
                                      .diagonal()
                                      .array()
                                      .log()
                                      .sum();
        gcache.chol.push_back(std::move(llt));
    }

    sites_latent_states = {{k, {}, 0}};

    // Sync the Eigen-typed root distribution (this->pi, the derived-class
    // member) into the base-class root_prior. Two parallel stores for now;
    // consolidating them is deferred (see Task #8).
    std::vector<double> prior_vec(k);
    for (size_t i = 0; i < k; ++i) prior_vec[i] = this->pi[i];
    root_prior = { std::move(prior_vec) };
}

// ── P(t) helpers ──────────────────────────────────────────────────────────────

Eigen::VectorXd phylo_gmm_model::apply_pmatrix(
    double t, const std::vector<double>& log_v) const
{
    // Convert from log-space to linear, promote to complex
    Eigen::VectorXcd v(k);
    for (size_t i = 0; i < k; ++i) v[i] = std::exp(log_v[i]);

    // P(t) v = U diag(exp(λt)) U⁻¹ v
    Eigen::VectorXcd w = ecache.U_inv * v;
    for (size_t p = 0; p < k; ++p)
        w[p] *= std::exp(ecache.lambda[p] * t);
    return (ecache.U * w).real();
}

Eigen::VectorXd phylo_gmm_model::apply_pmatrix_transpose(
    double t, const std::vector<double>& log_v) const
{
    Eigen::VectorXcd v(k);
    for (size_t i = 0; i < k; ++i) v[i] = std::exp(log_v[i]);

    // P(t)ᵀ = (U⁻¹)ᵀ diag(exp(λt)) Uᵀ
    // Note: .transpose() is the plain transpose, not conjugate transpose
    Eigen::VectorXcd w = ecache.U.transpose() * v;
    for (size_t p = 0; p < k; ++p)
        w[p] *= std::exp(ecache.lambda[p] * t);
    return (ecache.U_inv.transpose() * w).real();
}

// ── Required overrides ────────────────────────────────────────────────────────

void phylo_gmm_model::compute_log_pmatrix_vector_product(
    phylo_gmm_node_data&, size_t, double blen,
    const std::vector<double>& log_v, std::vector<double>& result) const
{
    Eigen::VectorXd res = apply_pmatrix(blen, log_v);
    for (size_t i = 0; i < k; ++i)
        result[i] = (res[i] > 0.0) ? std::log(res[i]) : NEGATIVE_INFINITY;
}

void phylo_gmm_model::compute_log_pmatrix_transpose_vector_product(
    phylo_gmm_node_data&, size_t, double blen,
    const std::vector<double>& log_v, std::vector<double>& result) const
{
    Eigen::VectorXd res = apply_pmatrix_transpose(blen, log_v);
    for (size_t i = 0; i < k; ++i)
        result[i] = (res[i] > 0.0) ? std::log(res[i]) : NEGATIVE_INFINITY;
}

// ── Hobolth-Jensen kernel ─────────────────────────────────────────────────────

std::complex<double> phylo_gmm_model::hobolth_jensen_kernel(
    std::complex<double> lambda_p,
    std::complex<double> lambda_q,
    double t) const
{
    // I_pq(t) = t * exp(t(λ_p+λ_q)/2) * sinhc(t(λ_p-λ_q)/2)   (eq 27)
    // sinhc(x) = sinh(x)/x, sinhc(0) = 1. Numerically stable for λ_p ≈ λ_q.
    std::complex<double> half_sum  = 0.5 * t * (lambda_p + lambda_q);
    std::complex<double> half_diff = 0.5 * t * (lambda_p - lambda_q);
    std::complex<double> sinhc = (std::abs(half_diff) < 1e-8)
        ? std::complex<double>(1.0)
        : std::sinh(half_diff) / half_diff;
    return (double)t * std::exp(half_sum) * sinhc;
}

// ── E-step ────────────────────────────────────────────────────────────────────

void phylo_gmm_model::e_step(
    tree& t,
    std::vector<phylo_gmm_node_data>&,
    const likelihood_buffer& inside_ll,
    const likelihood_buffer& outside_ll,
    const likelihood_buffer&,
    std::vector<double>& suffstats) const
{
    const size_t c        = 0;
    const size_t n_leaves = t.num_leaves;

    // Suffstats layout:
    //   [0,              k*n_leaves)         leaf responsibilities r[leaf*k + i]
    //   [k*n_leaves,     k*n_leaves+k)       root posterior γ_r(i)
    //   [k*n_leaves+k,   k*n_leaves+k+k*k)   E[N_ij | y]  [i*k + j]
    //   [k*n_leaves+k+k*k, ...)              E[T_i | y]   [i]
    const size_t off_r    = 0;
    const size_t off_root = k * n_leaves;
    const size_t off_N    = k * n_leaves + k;
    const size_t off_T    = k * n_leaves + k + k * k;
    suffstats.assign(k * n_leaves + k + k * k + k, 0.0);

    // log p(y) from root — valid for any node, but root is convenient
    double log_p_y;
    {
        std::vector<double> tmp(k);
        for (size_t i = 0; i < k; ++i)
            tmp[i] = inside_ll(c, t.root_id, i) + outside_ll(c, t.root_id, i);
        log_p_y = log_sum_exp(tmp.begin(), tmp.end());
    }

    // Normalize log-posteriors at a node and write probabilities to suffstats
    auto write_posterior = [&](size_t node_id, size_t out_offset) {
        std::vector<double> log_g(k);
        for (size_t i = 0; i < k; ++i)
            log_g[i] = inside_ll(c, node_id, i) + outside_ll(c, node_id, i);
        double norm = log_sum_exp(log_g.begin(), log_g.end());
        for (size_t i = 0; i < k; ++i)
            suffstats[out_offset + i] = std::exp(log_g[i] - norm);
    };

    // Leaf responsibilities r_ℓi = γ_ℓ(i)
    // Use leaf.data as the Y-row index (not node_id, which differs in parsed trees)
    for (int node_id : t.tree.nodes()) {
        if (t.tree.out_degree(node_id) != 0) continue;
        size_t leaf_data = t.tree[node_id].data;
        write_posterior((size_t)node_id, off_r + leaf_data * k);
    }

    // Root posterior γ_r(i) = P(X_root = i | y)
    write_posterior(t.root_id, off_root);

    // ── Hobolth-Jensen: accumulate H = Σ_e G_e ⊙ I_e ─────────────────────────
    //
    // For each edge e=(parent→v) with branch length t_e, define the k×k weight
    // matrix W_e[a,b] = R_parent(a) * L_v(b) / p(y).
    // Then G_e = U^T * W_e * (U_inv)^T and I_e[p,q] = kernel(λ_p, λ_q, t_e).
    // After accumulating H, form C = (U_inv)^T * H * U^T.
    //   E[T_i | y]   = Re(C[i,i])
    //   E[N_ij | y]  = Q_ij * Re(C[i,j])   (i ≠ j)
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(k, k);

    for (int node_id : t.tree.nodes()) {
        double blen = t.branch_lengths[node_id];

        // W_e[a,b]: parent prior for root edge, outside_ll of parent otherwise
        Eigen::MatrixXd W_e(k, k);
        if ((size_t)node_id == t.root_id) {
            for (size_t a = 0; a < k; ++a)
                for (size_t b = 0; b < k; ++b)
                    W_e(a, b) = std::exp(
                        std::log(pi[a]) + inside_ll(c, node_id, b) - log_p_y);
        } else {
            int parent_id = t.tree.predecessors(node_id)[0];
            for (size_t a = 0; a < k; ++a)
                for (size_t b = 0; b < k; ++b)
                    W_e(a, b) = std::exp(
                        outside_ll(c, parent_id, a) +
                        inside_ll(c, node_id, b) - log_p_y);
        }

        Eigen::MatrixXcd G_e = ecache.U.transpose()
                             * W_e.cast<std::complex<double>>()
                             * ecache.U_inv.transpose();

        Eigen::MatrixXcd I_e(k, k);
        for (size_t p = 0; p < k; ++p)
            for (size_t q = 0; q < k; ++q)
                I_e(p, q) = hobolth_jensen_kernel(
                    ecache.lambda[p], ecache.lambda[q], blen);

        H += G_e.cwiseProduct(I_e);
    }

    Eigen::MatrixXcd C = ecache.U_inv.transpose() * H * ecache.U.transpose();

    for (size_t i = 0; i < k; ++i)
        suffstats[off_T + i] = C(i, i).real();

    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < k; ++j)
            if (i != j)
                suffstats[off_N + i * k + j] = Q(i, j) * C(i, j).real();
}

// ── M-step ────────────────────────────────────────────────────────────────────

void phylo_gmm_model::m_step(
    tree&,
    std::vector<phylo_gmm_node_data>&,
    const std::vector<double>& suffstats)
{
    const size_t n_leaves = (size_t)Y.rows();

    const size_t off_r    = 0;
    const size_t off_root = k * n_leaves;
    const size_t off_N    = k * n_leaves + k;
    const size_t off_T    = k * n_leaves + k + k * k;

    // ── Gaussian updates (eqs 12-13) — skip unobserved leaves ────────────────
    auto leaf_observed = [&](size_t ell) {
        return is_observed.empty() || is_observed[ell];
    };

    std::vector<double> N_i(k, 0.0);
    for (size_t i = 0; i < k; ++i)
        for (size_t ell = 0; ell < n_leaves; ++ell) {
            if (!leaf_observed(ell)) continue;
            N_i[i] += suffstats[off_r + ell * k + i];
        }

    for (size_t i = 0; i < k; ++i) {
        Eigen::VectorXd new_mu = Eigen::VectorXd::Zero(d);
        for (size_t ell = 0; ell < n_leaves; ++ell) {
            if (!leaf_observed(ell)) continue;
            new_mu += suffstats[off_r + ell * k + i] * Y.row(ell).transpose();
        }
        mu[i] = new_mu / N_i[i];
    }

    for (size_t i = 0; i < k; ++i) {
        Eigen::MatrixXd new_Sigma = Eigen::MatrixXd::Zero(d, d);
        for (size_t ell = 0; ell < n_leaves; ++ell) {
            if (!leaf_observed(ell)) continue;
            Eigen::VectorXd diff = Y.row(ell).transpose() - mu[i];
            new_Sigma += suffstats[off_r + ell * k + i] * diff * diff.transpose();
        }
        Sigma[i] = new_Sigma / N_i[i] + 1e-6 * Eigen::MatrixXd::Identity(d, d);
    }

    // ── Root update (eq 14) ───────────────────────────────────────────────────
    for (size_t i = 0; i < k; ++i)
        pi[i] = suffstats[off_root + i];

    // ── Q update: Holmes-Rubin (eq 18) ────────────────────────────────────────
    for (size_t i = 0; i < k; ++i) {
        double T_i = suffstats[off_T + i];
        Q(i, i) = 0.0;  // zero diagonal before summing so row(i).sum() = off-diagonal sum
        for (size_t j = 0; j < k; ++j) {
            if (i == j) continue;
            Q(i, j) = suffstats[off_N + i * k + j] / T_i;
        }
        Q(i, i) = -Q.row(i).sum();
    }

    build();
}

} // namespace phylocore
