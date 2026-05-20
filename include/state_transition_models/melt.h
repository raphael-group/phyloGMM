#ifndef PHYLOCORE_MODELS_MELT_H
#define PHYLOCORE_MODELS_MELT_H

#include <vector>

#include <Eigen/Dense>

#include "../constants.h"
#include "../phylogenetic_model.h"

namespace phylocore {

/*
 * Precomputed per-node values for the MELT lineage component.
 *
 * The 2-state asymmetric CTMC (states: unmethylated=0, methylated=1) has
 * closed-form transition probabilities for a given branch length t:
 *
 *   P[i][j] = P(child = j | parent = i, t)
 *
 *   r = nu + mu,   e = exp(-r * t)
 *   P[0][0] = (mu + nu*e) / r     P[0][1] = nu*(1-e) / r
 *   P[1][0] = mu*(1-e)  / r       P[1][1] = (nu + mu*e) / r
 *
 * log_p_lin[i][j] = log P[i][j], precomputed in initialize_data().
 * The K-state cell-type transition matrix is computed on-the-fly per call
 * using the model-level eigendecomposition, so it is not cached here.
 */
struct melt_node_data {
    double log_p_lin[2][2];
};

/*!
 * @brief MELT (Methylation Lineage-Tracing) substitution model.
 *
 * Models a binary methylation matrix (cells × sites) where each site has a
 * relaxed assignment τ_s ∈ [0, 1] that interpolates between two regimes:
 *
 *   τ_s = 1  Lineage-informative: methylation evolves via a 2-state
 *             asymmetric CTMC with shared rates nu (0→1) and mu (1→0).
 *
 *   τ_s = 0  Cell-type-informative: methylation is a deterministic
 *             function of cell type. Cell types evolve along the tree via a
 *             K-state symmetric CTMC (GTR-style, shared across all sites).
 *             For site s and cell type k, theta[s][k] ∈ {0, 1} gives the
 *             expected methylation state.
 *
 * The state space at each site is K+2:
 *   states 0, 1        — lineage methylation (unmethylated, methylated)
 *   states 2, ..., K+1 — cell-type identity (cell types 0, ..., K-1)
 *
 * The P matrix is block-diagonal (lineage and cell-type components do not
 * couple during tree propagation). The mixing weight τ_s enters only at the
 * root distribution, which weights the lineage stationary distribution by τ_s
 * and the cell-type stationary distribution by (1 - τ_s). This yields:
 *
 *   LL[s] = log( τ_s * L_lineage(s) + (1 - τ_s) * L_celltype(s) )
 *
 * Parameters (stored in the inherited parameters vector):
 *   parameters[0] = nu   (lineage methylation rate, 0 → 1)
 *   parameters[1] = mu   (lineage demethylation rate, 1 → 0)
 *
 * Usage:
 *   1. Set methylation_matrix, cell_type_labels, tau, theta, cell_rates,
 *      cell_freqs, nu, mu.
 *   2. Call build() to eigendecompose the cell-type CTMC and set alphabet_sizes.
 *   3. Call initialize_data(t) to get per-node caches.
 *   4. Pass the model and node_data to the standard phylocore likelihood traversals.
 *
 * To update nu, mu, cell_rates, cell_freqs, or tau, modify them and call
 * build() + initialize_data() again before the next likelihood computation.
 */
class melt_model : public phylogenetic_model<melt_node_data> {
public:
    // ── Data ─────────────────────────────────────────────────────────────────

    std::vector<std::vector<int>> methylation_matrix; // [leaf_id][site] ∈ {-1, 0, 1}
    std::vector<int>              cell_type_labels;   // [leaf_id] ∈ {0,...,K-1}

    // ── Site-level parameters ─────────────────────────────────────────────────

    std::vector<double>           tau;   // [site] ∈ [0, 1]; fraction lineage-informative
    std::vector<std::vector<int>> theta; // [site][cell_type] ∈ {0, 1}; methylation emission

    // ── Cell-type CTMC parameters (K-state, GTR-style) ────────────────────────

    // Upper-triangle exchangeabilities, row-major: (0,1),(0,2),...,(K-2,K-1)
    std::vector<double> cell_rates; // length K*(K-1)/2
    // Stationary frequencies, must sum to 1
    std::vector<double> cell_freqs; // length K

    // ── Lineage CTMC parameters ───────────────────────────────────────────────

    double nu = 0.1; // methylation rate     (0 → 1)
    double mu = 0.1; // demethylation rate   (1 → 0)

    // ── Construction ─────────────────────────────────────────────────────────

    /*!
     * @brief Eigendecompose the cell-type CTMC and set alphabet_sizes.
     *
     * Builds Q_cell from cell_rates and cell_freqs (same GTR construction as
     * gtr_model), normalizes to unit expected substitution rate, and caches
     * the symmetric eigendecomposition for fast P(t) evaluation.
     *
     * Also sets alphabet_sizes[s] = K + 2 for all sites s, and syncs nu and
     * mu into the inherited parameters vector.
     *
     * Must be called after any change to nu, mu, cell_rates, or cell_freqs,
     * and before initialize_data() or any likelihood traversal.
     */
    void build();

    /*!
     * @brief Precompute per-node lineage transition log-probabilities.
     *
     * For each node, computes the 2×2 log-probability matrix of the lineage
     * CTMC at that node's branch length. Call this once after build() and
     * after any change to tree branch lengths.
     *
     * @param t Tree whose branch_lengths are used.
     */
    std::vector<melt_node_data> initialize_data(const tree& t) const;

    // ── Required interface ────────────────────────────────────────────────────

    void compute_log_pmatrix_vector_product(
        melt_node_data& d,
        size_t character,
        double branch_length,
        const std::vector<double>& log_vector,
        std::vector<double>& result) const override;

    void compute_log_pmatrix_transpose_vector_product(
        melt_node_data& d,
        size_t character,
        double branch_length,
        const std::vector<double>& log_vector,
        std::vector<double>& result) const override;

    void compute_taxa_log_inside_likelihood(
        melt_node_data& d,
        size_t character,
        size_t taxa_id,
        std::vector<double>& result) const override;

    void compute_root_distribution(
        melt_node_data& d,
        size_t character,
        std::vector<double>& result) const override;

private:
    // Eigendecomposition of the cell-type CTMC, built by build().
    // Mirrors the structure in gtr_model::character_cache.
    struct celltype_cache {
        Eigen::MatrixXd U;           // K×K eigenvectors (columns)
        Eigen::VectorXd lambda;      // K eigenvalues (all ≤ 0)
        Eigen::VectorXd sqrt_pi;     // sqrt of stationary frequencies
        Eigen::VectorXd inv_sqrt_pi; // 1 / sqrt_pi
    } cell_cache;

    // P_cell(t) * exp(log_v[offset:offset+K]) in linear space
    Eigen::VectorXd apply_celltype_pmatrix(
        double blen,
        const std::vector<double>& log_v,
        size_t offset) const;

    // P_cell(t)^T * exp(log_v[offset:offset+K]) in linear space
    Eigen::VectorXd apply_celltype_pmatrix_transpose(
        double blen,
        const std::vector<double>& log_v,
        size_t offset) const;
};

} // namespace phylocore

#endif // PHYLOCORE_MODELS_MELT_H
