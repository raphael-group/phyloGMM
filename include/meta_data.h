#ifndef PHYLOCORE_META_DATA_H
#define PHYLOCORE_META_DATA_H

#include <string>
#include <vector>

namespace phylocore {

/*!
 * @brief Metadata about a parsed dataset: site / alphabet info and taxa names.
 *
 * Observation data itself (per-leaf observations) lives outside this struct,
 * in a vector<O> handed to compute_inside_log_likelihood. meta_data carries
 * only the descriptive metadata that callers may want for I/O or validation.
 */
struct meta_data {
    size_t                   num_sites;
    // Uniform stride used by likelihood_buffer for O(1) flat indexing; equals
    // max(alphabet_sizes). Some sites are padded — those padded slots are
    // never read or written.
    size_t                   max_alphabet_size;
    // Real per-site alphabet sizes; may be heterogeneous across sites.
    std::vector<size_t>      alphabet_sizes;
    std::vector<std::string> taxa_names;

    // Optional per-leaf hard labels indexed by taxa_id (= leaf node.data).
    // Empty → no clamping anywhere. When non-empty:
    //   hard_labels[taxa_id] == -1   → leaf is unclamped (normal emission)
    //   hard_labels[taxa_id] >=  0   → inside_buffer at this leaf is forced to
    //                                  log P(X_leaf = j) = 0  if j == label
    //                                                    = -inf otherwise
    // The clamp is applied during compute_inside_log_likelihood at the leaf
    // branch. Downstream passes (edge_inside, outside, Hobolth-Jensen suff
    // stats, node/leaf posteriors) propagate it automatically.
    std::vector<int>         hard_labels;
};

} // namespace phylocore

#endif // PHYLOCORE_META_DATA_H
