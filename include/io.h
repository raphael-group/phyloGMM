#ifndef PHYLOCORE_IO_H
#define PHYLOCORE_IO_H

#include <string>
#include <vector>

#include "phylogeny.h"
#include "meta_data.h"

namespace phylocore {

/*!
 * @brief Parse a file containing one Newick string per non-empty line.
 */
std::vector<tree> parse_newick_trees(const std::string& fname);

/*!
 * @brief Serialize a tree to a Newick string.
 */
std::string write_newick_tree(const tree& t);

/*!
 * @brief Parse a character matrix CSV (rows = taxa, columns = sites).
 *
 * Returns (taxa_names, matrix) where matrix[i][j] is the state for taxon i,
 * site j. Missing/unknown entries ("?") are encoded as −1.
 */
std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
parse_character_matrix(const std::string& filename);

/*!
 * @brief Bundled result of process_input_data.
 *
 * observed_data is indexed by taxa_id (matches tree leaf vertex.data after
 * processing); metadata carries per-site alphabet info and taxa_names.
 */
template <typename O>
struct process_input_result {
    std::vector<O> observed_data;
    meta_data      metadata;
};

/*!
 * @brief Parse and align observation data against one or more trees.
 *
 * Modifies trees in place: for each tree, leaf node.data is set to the
 * global row index in observed_data; internal node.data is set to a
 * globally-offset index. Leaf sets across trees must be disjoint.
 *
 * Specialized per Observation type O.
 *   - O = std::vector<int>: parses a CSV character matrix (one int per site
 *     per taxon).
 *
 * For a single-tree run, pass a length-1 trees vector.
 */
template <typename O>
process_input_result<O> process_input_data(
    std::vector<tree>& trees,
    const std::string& observations_file
);

} // namespace phylocore

#endif // PHYLOCORE_IO_H
