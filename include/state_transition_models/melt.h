#ifndef PHYLOCORE_STATE_TRANSITION_MODELS_MELT_H
#define PHYLOCORE_STATE_TRANSITION_MODELS_MELT_H

#include <vector>

#include <Eigen/Dense>

#include "../constants.h"
#include "../state_transition_model.h"

namespace phylocore {

// Forward declaration.
struct tree;

/*
 * Precomputed per-node values for the MeLT lineage component.
 *
 * The 2-state asymmetric reversible CTMC (states: unmethylated=0, methylated=1)
 * has closed-form transition probabilities for a given branch length t:
 *
 *   r = nu + mu,   e = exp(-r * t)
 *   P[0][0] = (mu + nu*e) / r     P[0][1] = nu*(1-e) / r
 *   P[1][0] = mu*(1-e)  / r       P[1][1] = (nu + mu*e) / r
 *
 * log_p_lin[i][j] = log P[i][j], precomputed per-node in initialize_data().
 */
struct melt_node_data {
    double log_p_lin[2][2];
};

/*!
 * @brief MeLT (Methylation Lineage-Tracing) state transition model.
 *
 * Models a binary methylation matrix (cells x sites) together with a
 * cell-type label per cell. Each site has a soft assignment tau[s] in [0,1]
 * that interpolates between two regimes:
 *
 *   tau[s] = 1  Lineage-informative: methylation evolves via a 2-state
 *               asymmetric reversible CTMC with rates nu (0->1) and mu (1->0).
 *
 *   tau[s] = 0  Cell-type-informative: cell-type identity evolves via a
 *               K-state reversible (GTR-style) CTMC. The canonical methylation
 *               at each leaf is determined deterministically by the cell type
 *               via a separate theta table (stored in the observation model).
 *
 * The state space at every site is K+2:
 *   states 0, 1        -- lineage methylation (unmethylated, methylated)
 *   states 2, ..., K+1 -- cell-type identity (cell types 0, ..., K-1)
 *
 * The transition matrix P(t) is block-diagonal: lineage and cell-type
 * components never couple during tree propagation. The mixing weight tau[s]
 * enters only at the root distribution, which yields:
 *
 *   LL[s] = log( tau[s] * L_lineage(s) + (1 - tau[s]) * L_celltype(s) )
 *
 * Lineage parameters:
 *   nu  -- methylation rate   (0 -> 1)
 *   mu  -- demethylation rate (1 -> 0)
 *
 * Cell-type CTMC parameters:
 *   cell_rates -- K*(K-1)/2 exchangeabilities (upper triangle, row-major)
 *   cell_freqs -- K stationary frequencies (must sum to 1)
 *
 * Usage:
 *   1. Set nu, mu, cell_rates, cell_freqs, tau.
 *   2. Call build() to eigendecompose the cell-type CTMC, set alphabet_sizes,
 *      and populate root_prior.
 *   3. Call initialize_data(t) to obtain per-node lineage caches.
 *   4. Pass the model and node_data to the standard phylocore likelihood
 *      traversals together with a melt_observation_model.
 */
class melt_ctmc_model : public state_transition_model<melt_node_data> {
public:
    // ── Lineage CTMC parameters ───────────────────────────────────────────────

    double nu = 0.1; // methylation rate   0 -> 1
    double mu = 0.1; // demethylation rate 1 -> 0

    // ── Cell-type CTMC parameters (K-state, GTR-style) ────────────────────────

    // Upper-triangle exchangeabilities, row-major: (0,1),(0,2),...,(K-2,K-1)
    std::vector<double> cell_rates; // length K*(K-1)/2

    // Stationary frequencies, must sum to 1
    std::vector<double> cell_freqs; // length K

    // ── Site-level assignment ─────────────────────────────────────────────────

    std::vector<double> tau; // [site] in [0,1]; fraction lineage-informative

    // ── Construction ─────────────────────────────────────────────────────────

    /*!
     * @brief Eigendecompose the cell-type CTMC, set alphabet_sizes, and
     *        populate root_prior from nu, mu, cell_freqs, and tau.
     *
     * Must be called after any change to nu, mu, cell_rates, cell_freqs, or tau,
     * and before initialize_data() or any likelihood traversal.
     */
    void build();

    /*!
     * @brief Precompute per-node lineage transition log-probabilities.
     *
     * For each node computes the 2x2 log-P matrix of the lineage CTMC at
     * that node's branch length. Call once after build() and after any change
     * to tree branch lengths.
     */
    std::vector<melt_node_data> initialize_data(const tree& t) const;

    // ── Required interface ────────────────────────────────────────────────────

    /*!
     * @brief log P(alpha -> beta | branch_length, site).
     *
     * The state space is block-diagonal:
     *
     *   alpha, beta in {0,1}   -- lineage block, read from d.log_p_lin
     *   alpha, beta >= 2       -- cell-type block, computed from eigendecomposition
     *   cross-block            -- NEGATIVE_INFINITY (impossible)
     */
    double compute_log_transition_probability(
        melt_node_data& d,
        size_t          site,
        double          branch_length,
        size_t          alpha,
        size_t          beta
    ) const override;

    /*!
     * @brief Block-diagonal matvec: log(P(t) * exp(log_vector)) or its transpose.
     *
     * log_vector layout:
     *   [0..1]    lineage methylation states (unmethylated, methylated)
     *   [2..K+1]  cell-type states (cell types 0,...,K-1)
     */
    void compute_log_pmatrix_vector_product(
        melt_node_data&            d,
        size_t                     site,
        double                     branch_length,
        const std::vector<double>& log_vector,
        std::vector<double>&       result,
        bool                       transpose
    ) const override;

private:
    // Eigendecomposition of the normalized cell-type CTMC.
    // Mirrors the pattern in reversible_ctmc_model::site_cache.
    struct celltype_cache {
        Eigen::MatrixXd U;           // K x K eigenvectors (columns)
        Eigen::VectorXd lambda;      // K eigenvalues (all <= 0)
        Eigen::VectorXd sqrt_pi;     // element-wise sqrt of cell_freqs
        Eigen::VectorXd inv_sqrt_pi; // 1 / sqrt_pi
    } cell_cache;

    // Apply P_cell(t) * exp(log_v[offset : offset+K]) in linear space.
    // Returns a K-vector.
    Eigen::VectorXd apply_celltype_pmatrix(
        double blen,
        const std::vector<double>& log_v,
        size_t offset) const;

    // Apply P_cell(t)^T * exp(log_v[offset : offset+K]) in linear space.
    Eigen::VectorXd apply_celltype_pmatrix_transpose(
        double blen,
        const std::vector<double>& log_v,
        size_t offset) const;
};

} // namespace phylocore

#endif // PHYLOCORE_STATE_TRANSITION_MODELS_MELT_H
