#ifndef PHYLOCORE_STATE_SPACE_H
#define PHYLOCORE_STATE_SPACE_H

#include <string>
#include <vector>

namespace phylocore {

/*!
 * @brief Describes the discrete state space at a single site.
 *
 * alphabet_characters[i] is the human-readable label for state i.
 * alphabet_size equals alphabet_characters.size().
 * missing_index is the position in alphabet_characters that represents
 * missing data.
 */
struct state_space {
    size_t                   alphabet_size;
    std::vector<std::string> alphabet_characters;
    size_t                   missing_index;
};

} // namespace phylocore

#endif // PHYLOCORE_STATE_SPACE_H
