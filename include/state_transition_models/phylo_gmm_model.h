#ifndef PHYLOCORE_MODELS_PHYLO_GMM_MODEL_H
#define PHYLOCORE_MODELS_PHYLO_GMM_MODEL_H

#include <complex>
#include <vector>

#include <Eigen/Dense>

#include "../state_transition_model.h"

namespace phylocore {

struct phylo_gmm_node_data {};

/*!
 * @brief Phylogenetic Gaussian Mixture Model.
 *
 * Latent states 1..k evolve along the tree via a general (possibly
 * non-reversible) CTMC with rate matrix Q. At each leaf ℓ, the latent
 * state i emits a d-dimensional Gaussian observation:
 *
 *   y_ℓ | X_ℓ = i  ~  N(mu[i], Sigma[i])
 *
 * The model jointly estimates Q, {mu[i]}, {Sigma[i]}, and the root
 * distribution pi via EM (Felsenstein inside-outside for E-step,
 * Holmes-Rubin / closed-form Gaussian updates for M-step).
 *
 * Usage:
 *   1. Construct with explicit parameters, or use the convenience
 *      initializer init_kmeans() to seed from data.
 *   2. Call build() to compute the eigendecomposition of Q.
 *   3. Pass to em_optimize().
 *
 * After each M-step, build() is called automatically to refresh the cache.
 */
class phylo_gmm_model : public state_transition_model<phylo_gmm_node_data> {
public:
    // ── Model parameters ──────────────────────────────────────────────────────

    size_t k; // number of latent states
    size_t d; // observation dimension

    Eigen::MatrixXd Q;                    // k×k CTMC rate matrix
    std::vector<Eigen::VectorXd> mu;      // k means, each of length d
    std::vector<Eigen::MatrixXd> Sigma;   // k covariance matrices, each d×d
    Eigen::VectorXd pi;                   // k-vector, root distribution (probabilities)

    // Leaf observations: row ℓ is y_ℓ ∈ R^d, indexed by global leaf ID.
    // Must be set before build() and kept consistent with the tree.
    Eigen::MatrixXd Y; // n_leaves × d

    // is_observed[ℓ] = true iff leaf ℓ has an actual observation in Y.
    // Unobserved leaves contribute uniform emission (log-likelihood 0 for all states).
    // Length must equal Y.rows(). If empty, all leaves are treated as observed.
    std::vector<bool> is_observed;

    // ── Construction ─────────────────────────────────────────────────────────

    /*!
     * @brief Construct with explicit parameters.
     *
     * @param Q_     k×k rate matrix (rows sum to zero, off-diagonal ≥ 0)
     * @param mu_    k means, each of length d
     * @param Sigma_ k covariance matrices, each d×d symmetric positive definite
     * @param pi_    root distribution (length k, sums to 1)
     * @param Y_     n_leaves × d observation matrix, rows indexed by global leaf ID
     */
    phylo_gmm_model(
        Eigen::MatrixXd              Q_,
        std::vector<Eigen::VectorXd> mu_,
        std::vector<Eigen::MatrixXd> Sigma_,
        Eigen::VectorXd              pi_,
        Eigen::MatrixXd              Y_);

    /*!
     * @brief Convenience initializer: seed parameters from data via k-means.
     *
     * Runs k-means++ on Y to obtain initial mu and Sigma (per-cluster sample
     * covariances). Initializes Q = -I + (1/k) * 11^T (uniform off-diagonal
     * rates) and pi as the uniform distribution.
     *
     * @param k_  Number of latent states
     * @param Y_  n_leaves × d observation matrix
     * @param seed Random seed for k-means (default 42)
     */
    static phylo_gmm_model init_kmeans(
        size_t k_,
        Eigen::MatrixXd Y_,
        unsigned seed = 42);

    /*!
     * @brief Compute and cache the eigendecomposition of Q.
     *
     * Must be called after construction and after any direct modification of Q.
     * Called automatically by m_step(). Sets sites_latent_states to a single-element {{k, {}, 0}}.
     */
    void build();

    // ── Required state_transition_model interface ────────────────────────────

    void compute_log_pmatrix_vector_product(
        phylo_gmm_node_data& data,
        size_t character,
        double branch_length,
        const std::vector<double>& log_vector,
        std::vector<double>& result) const override;

    void compute_log_pmatrix_transpose_vector_product(
        phylo_gmm_node_data& data,
        size_t character,
        double branch_length,
        const std::vector<double>& log_vector,
        std::vector<double>& result) const override;

    // ── EM interface ──────────────────────────────────────────────────────────

    /*!
     * @brief E-step: compute sufficient statistics from inside-outside buffers.
     *
     * Sufficient statistics layout in suffstats (all flattened row-major):
     *   [0,        k*d)       — weighted sum Σ_ℓ r_ℓi * y_ℓ  for each i (k blocks of d)
     *   [k*d,      2*k*d)     — effective counts N_i = Σ_ℓ r_ℓi  (k scalars, stored as d-replicated for layout convenience; actual values at [k*d + i*d])
     *   [2*k*d,    2*k*d+k)   — root posterior γ_r(i)  (k scalars)
     *   [2*k*d+k,  2*k*d+k+k*k)  — E[N_ij | y] flattened k×k
     *   [2*k*d+k+k*k, 2*k*d+k+2*k*k) — E[T_i | y] (k scalars, stored as k*k for uniform layout; actual at [2*k*d+k+k*k + i*k])
     *
     * Note: the layout above is an initial proposal — feel free to revise once
     * the M-step implementation makes the exact access pattern clear.
     */
    void e_step(
        tree& t,
        std::vector<phylo_gmm_node_data>& node_data,
        const likelihood_buffer& inside_ll,
        const likelihood_buffer& outside_ll,
        const likelihood_buffer& edge_inside_ll,
        std::vector<double>& suffstats) const override;

    /*!
     * @brief M-step: closed-form updates for mu, Sigma, pi, and Q.
     *
     * Updates mu and Sigma via weighted sample statistics (eqs 12-13),
     * pi via root posterior (eq 14), and Q via Holmes-Rubin (eq 18).
     * Calls build() to refresh the eigendecomposition cache.
     */
    void m_step(
        tree& t,
        std::vector<phylo_gmm_node_data>& node_data,
        const std::vector<double>& suffstats) override;

private:
    struct eigen_cache {
        Eigen::MatrixXcd U;       // k×k matrix of eigenvectors (columns)
        Eigen::MatrixXcd U_inv;   // k×k inverse of U
        Eigen::VectorXcd lambda;  // k eigenvalues
    };

    struct gaussian_cache {
        std::vector<Eigen::LLT<Eigen::MatrixXd>> chol; // Cholesky of each Sigma[i]
        Eigen::VectorXd log_det;                        // log|Sigma[i]| for i=0..k-1
        double log_2pi_d;                               // d * log(2π), precomputed
    };

    eigen_cache    ecache;
    gaussian_cache gcache;

    // P(t) * v in linear space, result is real
    Eigen::VectorXd apply_pmatrix(
        double branch_length,
        const std::vector<double>& log_v) const;

    // P(t)^T * v in linear space, result is real
    Eigen::VectorXd apply_pmatrix_transpose(
        double branch_length,
        const std::vector<double>& log_v) const;

    // Hobolth-Jensen kernel I_pq(t): numerically stable divided difference
    // of exp at eigenvalues lambda[p] and lambda[q] (eq 27 of paper).
    std::complex<double> hobolth_jensen_kernel(
        std::complex<double> lambda_p,
        std::complex<double> lambda_q,
        double t) const;
};

} // namespace phylocore

#endif // PHYLOCORE_MODELS_PHYLO_GMM_MODEL_H
