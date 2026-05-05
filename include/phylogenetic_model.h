#ifndef PHYLOCORE_PHYLOGENETIC_MODEL_H
#define PHYLOCORE_PHYLOGENETIC_MODEL_H

#include <vector>

namespace phylocore {

// Forward declaration — full definition is in phylogeny.h.
// Avoids a circular dependency since phylogeny.h includes this header.
class likelihood_buffer;

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

    // ── EM branch length optimization (optional) ──────────────────────────────
    //
    // Models that support EM-based branch length optimization override both
    // methods below. The default no-ops cause em_optimize_branch_lengths to
    // leave branch lengths unchanged, so models that don't implement EM can
    // still link against phylocore without issue.

    /*!
     * @brief E-step: compute sufficient statistics for the branch leading into
     *        node_id, for a single character.
     *
     * Called once per node per character after the inside-outside pass. The
     * returned vector is model-defined — pack whatever the M-step needs.
     * Return an empty vector to signal that this node/character should be skipped.
     */
    virtual std::vector<double> e_step_branch(
        D& data,
        size_t node_id,
        size_t character,
        const likelihood_buffer& inside_ll,
        const likelihood_buffer& outside_ll,
        const likelihood_buffer& edge_inside_ll
    ) const { return {}; }

    /*!
     * @brief M-step: given per-character sufficient statistics, return the
     *        optimal branch length for node_id.
     *
     * suffstats[c] is the vector returned by e_step_branch for character c.
     * Return current_branch_length to leave the branch unchanged.
     */
    virtual double m_step_branch(
        size_t node_id,
        const std::vector<std::vector<double>>& suffstats,
        double current_branch_length
    ) const { return current_branch_length; }
};

} // namespace phylocore

#endif // PHYLOCORE_PHYLOGENETIC_MODEL_H
