#ifndef PHYLOCORE_MODELS_GTR_H
#define PHYLOCORE_MODELS_GTR_H

#include <vector>

#include <Eigen/Dense>

#include "../constants.h"
#include "../phylogenetic_model.h"

namespace phylocore {

struct gtr_node_data {};

/*!
 * @brief General Time Reversible (GTR) substitution model — k-state reversible CTMC.
 *
 * Each character has its own rate matrix parameterized by:
 *   - rates[c]: k*(k-1)/2 exchangeability rates (upper triangle of symmetric S,
 *               row-major: (0,1), (0,2), ..., (0,k-1), (1,2), ..., (k-2,k-1))
 *   - freqs[c]: k stationary frequencies summing to 1
 *
 * Usage:
 *   1. Set alphabet_sizes, rates, freqs, character_matrix
 *   2. Call build() to compute eigendecompositions and normalize Q
 *   3. Pass to compute_inside_log_likelihood, optimize_branch_lengths, etc.
 *
 * To update parameters after construction, modify rates/freqs and call build() again.
 */
class gtr_model : public phylogenetic_model<gtr_node_data> {
public:
    std::vector<std::vector<double>> rates;            // [character][k*(k-1)/2]
    std::vector<std::vector<double>> freqs;            // [character][k], must sum to 1
    std::vector<std::vector<int>>    character_matrix; // [leaf_id][character], -1 = missing

    /*!
     * @brief Build eigendecompositions from rates and freqs.
     *
     * Sets alphabet_sizes from freqs[c].size(). Normalizes each Q so that the
     * expected number of substitutions per unit time equals 1. Syncs rates and
     * freqs into the inherited flat parameters vector.
     */
    void build();

    void compute_log_pmatrix_vector_product(
        gtr_node_data& data,
        size_t character,
        double branch_length,
        const std::vector<double>& log_vector,
        std::vector<double>& result) const override;

    void compute_log_pmatrix_transpose_vector_product(
        gtr_node_data& data,
        size_t character,
        double branch_length,
        const std::vector<double>& log_vector,
        std::vector<double>& result) const override;

    void compute_taxa_log_inside_likelihood(
        gtr_node_data& data,
        size_t character,
        size_t taxa_id,
        std::vector<double>& result) const override;

    void compute_root_distribution(
        gtr_node_data& data,
        size_t character,
        std::vector<double>& result) const override;

private:
    struct character_cache {
        Eigen::MatrixXd U;           // eigenvectors (columns), k×k
        Eigen::VectorXd lambda;      // eigenvalues (ascending, all ≤ 0)
        Eigen::VectorXd sqrt_pi;     // sqrt of stationary frequencies
        Eigen::VectorXd inv_sqrt_pi; // 1 / sqrt of stationary frequencies
    };

    std::vector<character_cache> cache; // one per character, filled by build()

    // P(t) * exp(log_v) in linear space, using precomputed eigendecomposition
    Eigen::VectorXd apply_pmatrix(
        const character_cache& ch, double blen,
        const std::vector<double>& log_v) const;

    // P(t)^T * exp(log_v) in linear space
    Eigen::VectorXd apply_pmatrix_transpose(
        const character_cache& ch, double blen,
        const std::vector<double>& log_v) const;
};

} // namespace phylocore

#endif // PHYLOCORE_MODELS_GTR_H
