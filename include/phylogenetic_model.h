#ifndef PHYLOCORE_PHYLOGENETIC_MODEL_H
#define PHYLOCORE_PHYLOGENETIC_MODEL_H

#include <vector>

namespace phylocore {

/*!
 * @brief Abstract base class for substitution models used in phylogenetic likelihood.
 *
 * @tparam D Model-specific per-node data type.
 */
template <typename D>
class phylogenetic_model {
public:
    std::vector<size_t> alphabet_sizes; // one entry per character
    std::vector<double> parameters;     // non-branch-length model parameters

    virtual ~phylogenetic_model() = default;

    /*!
     * @brief Computes log(P^c * v) for character c and branch length b,
     *        where P^c is the transition probability matrix.
     */
    virtual void compute_log_pmatrix_vector_product(
        D& data,
        size_t character,
        double branch_length,
        const std::vector<double>& log_vector,
        std::vector<double>& result
    ) const = 0;

    /*!
     * @brief Computes log(v^T * P^c) for character c and branch length b.
     */
    virtual void compute_log_pmatrix_transpose_vector_product(
        D& data,
        size_t character,
        double branch_length,
        const std::vector<double>& log_vector,
        std::vector<double>& result
    ) const = 0;

    /*!
     * @brief Computes the log inside likelihood at a leaf for a given character.
     *
     * @param taxa_id Leaf index (0 to num_taxa - 1).
     */
    virtual void compute_taxa_log_inside_likelihood(
        D& data,
        size_t character,
        size_t taxa_id,
        std::vector<double>& result
    ) const = 0;

    /*!
     * @brief Computes the log root distribution for a given character.
     */
    virtual void compute_root_distribution(
        D& data,
        size_t character,
        std::vector<double>& result
    ) const = 0;
};

} // namespace phylocore

#endif // PHYLOCORE_PHYLOGENETIC_MODEL_H
