#ifndef PHYLOCORE_IO_H
#define PHYLOCORE_IO_H

#include <string>
#include <vector>

#include "phylogeny.h"
#include "phylogeny_data.h"

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
 * @brief Parse a character matrix CSV (rows = taxa, columns = characters).
 *
 * Returns (taxa_names, matrix) where matrix[i][j] is the state for taxon i,
 * character j. Missing/unknown entries ("?") are encoded as −1.
 */
std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
parse_character_matrix(const std::string& filename);

/*!
 * @brief Align a character matrix to one or more trees and build phylogeny_data.
 *
 * For each tree, leaf node.data is set to the global row index in the returned
 * character_matrix; internal node.data is set to a globally-offset index.
 * Leaf sets across trees must be disjoint.
 *
 * For a single-tree run, pass a length-1 vector.
 */
phylogeny_data process_phylogeny_data(
    std::vector<tree>& trees,
    const std::string& character_matrix_file
);

} // namespace phylocore

#endif // PHYLOCORE_IO_H
