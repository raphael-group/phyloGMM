#ifndef PHYLOCORE_PHYLOGENY_DATA_H
#define PHYLOCORE_PHYLOGENY_DATA_H

#include <string>
#include <vector>

namespace phylocore {

/*!
 * @brief Core data container linking a character matrix to a set of trees.
 *
 * character_matrix[leaf_id][character] holds integer states (−1 = missing).
 * Downstream consumers extend this struct via inheritance to add fields such as
 * observation_matrix, mutation_priors, and state_map.
 */
struct phylogeny_data {
    size_t num_characters;
    size_t max_alphabet_size;
    std::vector<std::string>      taxa_names;
    std::vector<std::vector<int>> character_matrix; // [leaf_id][character]
};

} // namespace phylocore

#endif // PHYLOCORE_PHYLOGENY_DATA_H
