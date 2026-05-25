#ifndef PHYLOCORE_STATE_SPACE_H
#define PHYLOCORE_STATE_SPACE_H

#include <string>
#include <vector>

namespace phylocore {

/*!
 * @brief Describes the state space at a single site.
 *
 * Supports two cases:
 *
 *   Discrete (is_continuous == false; the default):
 *     - alphabet_characters[i] is the human-readable label for state i.
 *     - alphabet_size equals alphabet_characters.size().
 *     - missing_index is the position in alphabet_characters that represents
 *       missing data.
 *
 *   Continuous (is_continuous == true):
 *     - dimension is the dimensionality of the observation space (ℝ^dimension).
 *     - alphabet_size, alphabet_characters, missing_index are unused.
 *
 * Code that consumes a state_space should check is_continuous before reading
 * the case-specific fields. Defaults yield a discrete state_space with empty
 * alphabet — existing aggregate initializations (e.g., state_space{k, {}, 0})
 * remain valid; the new fields default appropriately.
 */
struct state_space {
    // ── Discrete case ────────────────────────────────────────────────────────
    size_t                   alphabet_size       = 0;
    std::vector<std::string> alphabet_characters;
    size_t                   missing_index       = 0;

    // ── Continuous case ──────────────────────────────────────────────────────
    bool                     is_continuous       = false;
    size_t                   dimension           = 0;
};

} // namespace phylocore

#endif // PHYLOCORE_STATE_SPACE_H
