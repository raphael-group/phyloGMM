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
};

} // namespace phylocore

#endif // PHYLOCORE_META_DATA_H
