#ifndef PHYLOCORE_PHYLOGENETIC_MODEL_H
#define PHYLOCORE_PHYLOGENETIC_MODEL_H

#include <vector>

namespace phylocore {

// Forward declarations — full definitions are in phylogeny.h.
// Avoids a circular dependency since phylogeny.h includes this header.
class likelihood_buffer;
struct tree;

/*!
 * @brief Abstract base class for substitution models used in phylogenetic likelihood.
 *
 * To define a new model, inherit from this class with your per-node data type D
 * and implement the four pure virtual methods. You then get likelihood computation,
 * branch length optimization (Brent's), and topology search for free.
 *
 * To support EM-based parameter optimization, additionally override e_step and
 * m_step. The M-step may update branch lengths, model parameters, or both.
 *
 * @tparam D Model-specific per-node data type. Use an empty struct for homogeneous
 *           models where no per-node state is needed.
 */
template <typename D>
class phylogenetic_model {
public:
    std::vector<size_t> alphabet_sizes; // one entry per character
    std::vector<double> parameters;     // non-branch-length model parameters

    virtual ~phylogenetic_model() = default;

    // ── Required interface ────────────────────────────────────────────────────

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

    // ── Optional EM interface ─────────────────────────────────────────────────
    //
    // Override both methods to enable em_optimize(). The model owns the
    // sufficient statistics format and decides what gets updated in the M-step
    // — branch lengths, model parameters, or both.
    //
    // Default no-ops mean em_optimize() converges immediately for models that
    // don't override these (use optimize_branch_lengths() instead).

    /*!
     * @brief E-step: compute sufficient statistics from the inside-outside pass.
     *
     * Called once per EM iteration after all three likelihood buffers are filled.
     * Pack whatever the M-step needs into suffstats — the format is model-defined.
     */
    virtual void e_step(
        tree& t,
        std::vector<D>& node_data,
        const likelihood_buffer& inside_ll,
        const likelihood_buffer& outside_ll,
        const likelihood_buffer& edge_inside_ll,
        std::vector<double>& suffstats
    ) const {}

    /*!
     * @brief M-step: update parameters given sufficient statistics.
     *
     * May update t.branch_lengths, model parameters, node_data, or any
     * combination. The model has full control.
     */
    virtual void m_step(
        tree& t,
        std::vector<D>& node_data,
        const std::vector<double>& suffstats
    ) {}
};

} // namespace phylocore

#endif // PHYLOCORE_PHYLOGENETIC_MODEL_H
