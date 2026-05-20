#include "state_transition_models/melt.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "constants.h"
#include "math_utilities.h"
#include "phylogeny.h"

namespace phylocore {

// ── build ─────────────────────────────────────────────────────────────────────

void melt_model::build() {
    size_t K          = cell_freqs.size();
    size_t num_sites  = tau.size();

    // Every site has K+2 states: 2 lineage + K cell-type
    alphabet_sizes.assign(num_sites, K + 2);

    // Build symmetric exchangeability matrix S from upper-triangle rates
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(K, K);
    size_t r = 0;
    for (size_t i = 0; i < K; ++i)
        for (size_t j = i + 1; j < K; ++j)
            S(i, j) = S(j, i) = cell_rates[r++];

    Eigen::VectorXd pi(K);
    for (size_t i = 0; i < K; ++i) pi[i] = cell_freqs[i];

    // Q_cell[i][j] = S[i][j] * pi[j]  (i ≠ j),  Q_cell[i][i] = -row sum
    Eigen::MatrixXd Q = S * pi.asDiagonal();
    for (size_t i = 0; i < K; ++i)
        Q(i, i) = -Q.row(i).sum();

    // Normalize so that the expected number of substitutions per unit time = 1
    double rate = -(pi.transpose() * Q.diagonal()).value();
    if (rate <= 0.0)
        throw std::runtime_error("MELT: zero total rate for cell-type CTMC");
    Q /= rate;

    // Symmetric transformation A = diag(sqrt_pi) * Q * diag(inv_sqrt_pi);
    // A is real symmetric, so SelfAdjointEigenSolver is numerically stable
    Eigen::VectorXd sqrt_pi     = pi.array().sqrt();
    Eigen::VectorXd inv_sqrt_pi = sqrt_pi.array().inverse();
    Eigen::MatrixXd A           = sqrt_pi.asDiagonal() * Q * inv_sqrt_pi.asDiagonal();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("MELT: eigendecomposition failed for cell-type CTMC");

    cell_cache = { solver.eigenvectors(), solver.eigenvalues(), sqrt_pi, inv_sqrt_pi };

    parameters = { nu, mu };
}

// ── initialize_data ───────────────────────────────────────────────────────────

std::vector<melt_node_data> melt_model::initialize_data(const tree& t) const {
    double nu_ = parameters[0];
    double mu_ = parameters[1];
    double r   = nu_ + mu_;

    std::vector<melt_node_data> result(t.num_nodes);
    for (size_t i = 0; i < t.num_nodes; ++i) {
        double blen = t.branch_lengths[i];
        double e    = std::exp(-r * blen);
        // P_lineage[parent][child]:
        //   P[0][0] = (mu + nu*e)/r    P[0][1] = nu*(1-e)/r
        //   P[1][0] = mu*(1-e)/r       P[1][1] = (nu + mu*e)/r
        result[i].log_p_lin[0][0] = std::log((mu_ + nu_ * e) / r);
        result[i].log_p_lin[0][1] = std::log(nu_ * (1.0 - e) / r);
        result[i].log_p_lin[1][0] = std::log(mu_ * (1.0 - e) / r);
        result[i].log_p_lin[1][1] = std::log((nu_ + mu_ * e) / r);
    }
    return result;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// P_cell(t) * exp(log_v[offset:offset+K])  (linear space output)
// Uses P(t) = diag(inv_sqrt_pi) * U * diag(exp(λt)) * U^T * diag(sqrt_pi)
Eigen::VectorXd melt_model::apply_celltype_pmatrix(
    double blen, const std::vector<double>& log_v, size_t offset) const
{
    size_t K = cell_freqs.size();
    Eigen::VectorXd v(K);
    for (size_t i = 0; i < K; ++i) v[i] = std::exp(log_v[offset + i]);

    return cell_cache.inv_sqrt_pi.cwiseProduct(
        cell_cache.U * ((cell_cache.lambda.array() * blen).exp().matrix().cwiseProduct(
            cell_cache.U.transpose() * cell_cache.sqrt_pi.cwiseProduct(v))));
}

// P_cell(t)^T * exp(log_v[offset:offset+K])  (linear space output)
// Uses P(t)^T = diag(sqrt_pi) * U * diag(exp(λt)) * U^T * diag(inv_sqrt_pi)
Eigen::VectorXd melt_model::apply_celltype_pmatrix_transpose(
    double blen, const std::vector<double>& log_v, size_t offset) const
{
    size_t K = cell_freqs.size();
    Eigen::VectorXd v(K);
    for (size_t i = 0; i < K; ++i) v[i] = std::exp(log_v[offset + i]);

    return cell_cache.sqrt_pi.cwiseProduct(
        cell_cache.U * ((cell_cache.lambda.array() * blen).exp().matrix().cwiseProduct(
            cell_cache.U.transpose() * cell_cache.inv_sqrt_pi.cwiseProduct(v))));
}

// ── Transition matrix-vector products ────────────────────────────────────────

/*
 * Computes log(P(t) * exp(log_vector)) for the K+2 state space.
 *
 * log_vector layout:
 *   [0..1]     lineage methylation states (unmethylated, methylated)
 *   [2..K+1]   cell-type states (cell types 0,...,K-1)
 *
 * The block-diagonal structure of P means the two components decouple:
 *   lineage block : 2×2 P_lineage, applied analytically from cached log_p_lin
 *   cell-type block: K×K P_cell,   applied via eigendecomposition
 */
void melt_model::compute_log_pmatrix_vector_product(
    melt_node_data& d,
    size_t /*character*/,
    double branch_length,
    const std::vector<double>& log_vector,
    std::vector<double>& result) const
{
    size_t K = cell_freqs.size();

    // Lineage block: result[i] = log(sum_j P_lin[i][j] * exp(log_v[j]))
    for (int i = 0; i < 2; ++i) {
        double tmp[2] = {
            d.log_p_lin[i][0] + log_vector[0],
            d.log_p_lin[i][1] + log_vector[1]
        };
        result[i] = log_sum_exp(tmp, tmp + 2);
    }

    // Cell-type block: result[2+j] = log((P_cell * v_cell)[j])
    Eigen::VectorXd cell_res = apply_celltype_pmatrix(branch_length, log_vector, 2);
    for (size_t j = 0; j < K; ++j)
        result[j + 2] = (cell_res[j] > 0.0) ? std::log(cell_res[j]) : NEGATIVE_INFINITY;
}

void melt_model::compute_log_pmatrix_transpose_vector_product(
    melt_node_data& d,
    size_t /*character*/,
    double branch_length,
    const std::vector<double>& log_vector,
    std::vector<double>& result) const
{
    size_t K = cell_freqs.size();

    // Lineage block: (P_lin^T)[i][j] = P_lin[j][i], so
    // result[i] = log(sum_j P_lin[j][i] * exp(log_v[j]))
    for (int i = 0; i < 2; ++i) {
        double tmp[2] = {
            d.log_p_lin[0][i] + log_vector[0],
            d.log_p_lin[1][i] + log_vector[1]
        };
        result[i] = log_sum_exp(tmp, tmp + 2);
    }

    // Cell-type block: result[2+j] = log((P_cell^T * v_cell)[j])
    Eigen::VectorXd cell_res = apply_celltype_pmatrix_transpose(branch_length, log_vector, 2);
    for (size_t j = 0; j < K; ++j)
        result[j + 2] = (cell_res[j] > 0.0) ? std::log(cell_res[j]) : NEGATIVE_INFINITY;
}

// ── Leaf and root distributions ───────────────────────────────────────────────

/*
 * At each leaf the observation is a (methylation, cell_type) pair.
 *
 * Lineage states 0, 1:
 *   The observed methylation is a point mass — result[obs] = 0, result[1-obs] = -inf.
 *   If methylation is missing (-1), both lineage states have log-prob 0.
 *
 * Cell-type states 2..K+1:
 *   Only state 2+k (the observed cell type k) can be non-zero. It is zero
 *   when the observed methylation matches theta[site][k], and -inf otherwise.
 *   If methylation is missing, the cell-type state is always 0.
 *   All other cell-type states remain -inf (cell type is observed at leaves).
 */
void melt_model::compute_taxa_log_inside_likelihood(
    melt_node_data& /*d*/,
    size_t character,
    size_t taxa_id,
    std::vector<double>& result) const
{
    std::fill(result.begin(), result.end(), NEGATIVE_INFINITY);

    int obs    = methylation_matrix[taxa_id][character]; // -1, 0, or 1
    size_t k   = static_cast<size_t>(cell_type_labels[taxa_id]);

    // Lineage states: point mass at observed methylation
    if (obs == -1) {
        result[0] = result[1] = 0.0;
    } else {
        result[obs] = 0.0;
    }

    // Cell-type state 2+k: 0 iff methylation matches theta[site][k] (or is missing)
    if (obs == -1 || obs == theta[character][k])
        result[2 + k] = 0.0;
}

/*
 * Root distribution is the product of the stationary distribution and τ
 * (for the lineage component) or (1 − τ) (for the cell-type component).
 *
 * Lineage stationary: π_lin[0] = μ/(ν+μ),  π_lin[1] = ν/(ν+μ)
 * Cell-type stationary: π_cell[j] = cell_freqs[j]
 *
 * Together this ensures:
 *   LL[s] = log( τ_s * L_lineage(s) + (1-τ_s) * L_celltype(s) )
 */
void melt_model::compute_root_distribution(
    melt_node_data& /*d*/,
    size_t character,
    std::vector<double>& result) const
{
    double nu_    = parameters[0];
    double mu_    = parameters[1];
    double r      = nu_ + mu_;
    double tau_c  = tau[character];
    double one_minus_tau = 1.0 - tau_c;

    // Lineage block
    result[0] = (tau_c > 0.0) ? std::log(tau_c * mu_ / r) : NEGATIVE_INFINITY;
    result[1] = (tau_c > 0.0) ? std::log(tau_c * nu_ / r) : NEGATIVE_INFINITY;

    // Cell-type block
    for (size_t j = 0; j < cell_freqs.size(); ++j) {
        result[2 + j] = (one_minus_tau > 0.0)
            ? std::log(one_minus_tau * cell_freqs[j])
            : NEGATIVE_INFINITY;
    }
}

} // namespace phylocore
