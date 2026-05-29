#include "state_transition_models/melt.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "constants.h"
#include "math_utilities.h"
#include "phylogeny.h"

namespace phylocore {

// ── build ─────────────────────────────────────────────────────────────────────

void melt_ctmc_model::build() {
    size_t K         = cell_freqs.size();
    size_t num_sites = tau.size();

    if (K < 2)
        throw std::runtime_error("melt_ctmc_model: need at least 2 cell types");
    if (cell_rates.size() != K * (K - 1) / 2)
        throw std::runtime_error("melt_ctmc_model: cell_rates length mismatch");
    if (nu <= 0.0 || mu <= 0.0)
        throw std::runtime_error("melt_ctmc_model: nu and mu must be positive");

    // ── Cell-type CTMC eigendecomposition ────────────────────────────────────

    // Build symmetric exchangeability matrix S from upper-triangle cell_rates
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(K, K);
    {
        size_t r = 0;
        for (size_t i = 0; i < K; ++i)
            for (size_t j = i + 1; j < K; ++j)
                S(i, j) = S(j, i) = cell_rates[r++];
    }

    Eigen::VectorXd pi(K);
    for (size_t i = 0; i < K; ++i) pi[i] = cell_freqs[i];

    // Q[i][j] = S[i][j] * pi[j]  (i != j);  Q[i][i] = -(row sum)
    Eigen::MatrixXd Q = S * pi.asDiagonal();
    for (size_t i = 0; i < K; ++i)
        Q(i, i) = -Q.row(i).sum();

    // Normalize to unit expected substitution rate
    double rate = -(pi.transpose() * Q.diagonal()).value();
    if (rate <= 0.0)
        throw std::runtime_error("melt_ctmc_model: zero total rate for cell-type CTMC");
    Q /= rate;

    // Symmetric similarity transform A = diag(sqrt_pi) * Q * diag(inv_sqrt_pi)
    Eigen::VectorXd sqrt_pi     = pi.array().sqrt();
    Eigen::VectorXd inv_sqrt_pi = sqrt_pi.array().inverse();
    Eigen::MatrixXd A           = sqrt_pi.asDiagonal() * Q * inv_sqrt_pi.asDiagonal();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("melt_ctmc_model: eigendecomposition failed for cell-type CTMC");

    cell_cache = { solver.eigenvectors(), solver.eigenvalues(), sqrt_pi, inv_sqrt_pi };

    // ── Alphabet sizes ────────────────────────────────────────────────────────

    sites_latent_states.resize(num_sites);
    for (size_t s = 0; s < num_sites; ++s)
        sites_latent_states[s].alphabet_size = K + 2;

    // ── Root prior ────────────────────────────────────────────────────────────
    //
    // root_prior[s][j] = tau[s] * pi_lin[j]   for j in {0, 1}
    //                  = (1 - tau[s]) * pi[k]  for j = 2 + k
    //
    // This ensures LL[s] = log( tau[s]*L_lin(s) + (1-tau[s])*L_cell(s) ).

    double r_lin = nu + mu;
    double pi_lin0 = mu / r_lin; // stationary P(unmethylated)
    double pi_lin1 = nu / r_lin; // stationary P(methylated)

    root_prior.resize(num_sites);
    for (size_t s = 0; s < num_sites; ++s) {
        double t_s  = tau[s];
        double mt_s = 1.0 - t_s;
        root_prior[s].resize(K + 2);
        root_prior[s][0] = t_s * pi_lin0;
        root_prior[s][1] = t_s * pi_lin1;
        for (size_t k = 0; k < K; ++k)
            root_prior[s][2 + k] = mt_s * cell_freqs[k];
    }

    has_optimized_functions = true;
}

// ── initialize_data ───────────────────────────────────────────────────────────

std::vector<melt_node_data> melt_ctmc_model::initialize_data(const tree& t) const {
    double r = nu + mu;
    std::vector<melt_node_data> result(t.num_nodes);
    for (size_t i = 0; i < t.num_nodes; ++i) {
        double blen = t.branch_lengths[i];
        double e    = std::exp(-r * blen);
        result[i].log_p_lin[0][0] = std::log((mu + nu * e) / r);
        result[i].log_p_lin[0][1] = std::log(nu * (1.0 - e) / r);
        result[i].log_p_lin[1][0] = std::log(mu * (1.0 - e) / r);
        result[i].log_p_lin[1][1] = std::log((nu + mu * e) / r);
    }
    return result;
}

// ── Cell-type matvec helpers ──────────────────────────────────────────────────

// P_cell(t) = diag(inv_sqrt_pi) * U * diag(exp(lambda*t)) * U^T * diag(sqrt_pi)
// Applied to v (linear space), where v[i] = exp(log_v[offset + i]).
Eigen::VectorXd melt_ctmc_model::apply_celltype_pmatrix(
    double blen,
    const std::vector<double>& log_v,
    size_t offset) const
{
    size_t K = cell_freqs.size();
    Eigen::VectorXd v(K);
    for (size_t i = 0; i < K; ++i) v[i] = std::exp(log_v[offset + i]);

    // a = diag(sqrt_pi) * v
    Eigen::VectorXd a = cell_cache.sqrt_pi.cwiseProduct(v);
    // b = U^T * a
    Eigen::VectorXd b = cell_cache.U.transpose() * a;
    // c = diag(exp(lambda * blen)) * b
    Eigen::VectorXd c = (cell_cache.lambda.array() * blen).exp().matrix().cwiseProduct(b);
    // d = U * c
    Eigen::VectorXd d = cell_cache.U * c;
    // w = diag(inv_sqrt_pi) * d
    return cell_cache.inv_sqrt_pi.cwiseProduct(d);
}

// P_cell(t)^T = diag(sqrt_pi) * U * diag(exp(lambda*t)) * U^T * diag(inv_sqrt_pi)
Eigen::VectorXd melt_ctmc_model::apply_celltype_pmatrix_transpose(
    double blen,
    const std::vector<double>& log_v,
    size_t offset) const
{
    size_t K = cell_freqs.size();
    Eigen::VectorXd v(K);
    for (size_t i = 0; i < K; ++i) v[i] = std::exp(log_v[offset + i]);

    // a = diag(inv_sqrt_pi) * v
    Eigen::VectorXd a = cell_cache.inv_sqrt_pi.cwiseProduct(v);
    // b = U^T * a
    Eigen::VectorXd b = cell_cache.U.transpose() * a;
    // c = diag(exp(lambda * blen)) * b
    Eigen::VectorXd c = (cell_cache.lambda.array() * blen).exp().matrix().cwiseProduct(b);
    // d = U * c
    Eigen::VectorXd d = cell_cache.U * c;
    // w = diag(sqrt_pi) * d
    return cell_cache.sqrt_pi.cwiseProduct(d);
}

// ── compute_log_transition_probability ───────────────────────────────────────
//
// The K+2 x K+2 transition matrix is block-diagonal.
//
// Case 1: alpha, beta in {0, 1}  (lineage -> lineage)
//   Read from d.log_p_lin[alpha][beta].
//
// Case 2: alpha, beta >= 2  (cell-type -> cell-type)
//   Let i = alpha - 2, j = beta - 2.
//   P_cell(t)[i][j] = (1/sqrt_pi[i]) * sum_p( U[i,p]*exp(lambda_p*t)*U[j,p] ) * sqrt_pi[j]
//
// Case 3: cross-block  (one in {0,1}, other >= 2)
//   Return NEGATIVE_INFINITY.

double melt_ctmc_model::compute_log_transition_probability(
    melt_node_data& d,
    size_t          /*site*/,
    double          branch_length,
    size_t          alpha,
    size_t          beta
) const {
    // Case 1: lineage -> lineage
    if (alpha < 2 && beta < 2)
        return d.log_p_lin[alpha][beta];

    // Case 3: cross-block
    if (alpha < 2 || beta < 2)
        return NEGATIVE_INFINITY;

    // Case 2: cell-type -> cell-type
    size_t i = alpha - 2;
    size_t j = beta  - 2;

    // P_cell(t)[i][j] = inv_sqrt_pi[i] * sum_p(U[i,p]*exp(lambda_p*t)*U[j,p]) * sqrt_pi[j]
    double sum = 0.0;
    for (Eigen::Index p = 0; p < cell_cache.lambda.size(); ++p)
        sum += cell_cache.U(i, p)
             * std::exp(cell_cache.lambda(p) * branch_length)
             * cell_cache.U(j, p);
    sum *= cell_cache.inv_sqrt_pi(i) * cell_cache.sqrt_pi(j);
    return (sum > 0.0) ? std::log(sum) : NEGATIVE_INFINITY;
}

// ── compute_log_pmatrix_vector_product ────────────────────────────────────────
//
// Block-diagonal application of P(t) or P(t)^T to log_vector.
//
// Lineage block (indices 0, 1):
//   Forward:   result[i] = log sum_j P_lin[i][j] * exp(log_v[j])
//   Transpose: result[i] = log sum_j P_lin[j][i] * exp(log_v[j])
//              (i.e., read d.log_p_lin[j][i] instead of [i][j])
//
// Cell-type block (indices 2..K+1):
//   Forward:   via apply_celltype_pmatrix
//   Transpose: via apply_celltype_pmatrix_transpose

void melt_ctmc_model::compute_log_pmatrix_vector_product(
    melt_node_data&            d,
    size_t                     /*site*/,
    double                     branch_length,
    const std::vector<double>& log_vector,
    std::vector<double>&       result,
    bool                       transpose
) const {
    size_t K = cell_freqs.size();

    // ── Lineage block ─────────────────────────────────────────────────────────
    if (!transpose) {
        for (int i = 0; i < 2; ++i) {
            std::array<double, 2> terms = {
                d.log_p_lin[i][0] + log_vector[0],
                d.log_p_lin[i][1] + log_vector[1]
            };
            result[i] = log_sum_exp(terms.begin(), terms.end());
        }
    } else {
        // P^T[i][j] = P[j][i]
        for (int i = 0; i < 2; ++i) {
            std::array<double, 2> terms = {
                d.log_p_lin[0][i] + log_vector[0],
                d.log_p_lin[1][i] + log_vector[1]
            };
            result[i] = log_sum_exp(terms.begin(), terms.end());
        }
    }

    // ── Cell-type block ───────────────────────────────────────────────────────
    Eigen::VectorXd cell_res = transpose
        ? apply_celltype_pmatrix_transpose(branch_length, log_vector, 2)
        : apply_celltype_pmatrix(branch_length, log_vector, 2);

    for (size_t j = 0; j < K; ++j)
        result[j + 2] = (cell_res[j] > 0.0) ? std::log(cell_res[j]) : NEGATIVE_INFINITY;
}

} // namespace phylocore
