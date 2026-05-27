#ifndef PHYLOCORE_OBSERVATION_MODELS_PMM_CATEGORICAL_H
#define PHYLOCORE_OBSERVATION_MODELS_PMM_CATEGORICAL_H

#include <vector>

#include "../constants.h"
#include "../observation_model.h"
#include "../state_transition_models/laml_data.h"

namespace phylocore {

/*!
 * @brief Categorical-with-silencing observation model for LAML / PMM.
 *
 * Pairs with pmm_ctmc_model. The per-leaf observation O is the per-character
 * observed integer label using the same convention as the latent state space:
 *   obs[c] = -1   → missing / silenced (sentinel)
 *   obs[c] =  0   → observed unedited
 *   obs[c] >= 1   → observed mutation in alphabet category (obs[c] + 1)
 *                   (because the latent alphabet places mutated categories
 *                    starting at index 2; see pmm_ctmc.h)
 *
 * Emission probabilities, with phi = per-leaf dropout / silencing rate
 * (cached via laml_data.log_phi, laml_data.log_one_minus_phi):
 *
 *   P(y = missing  | X = silenced  ) = 1
 *   P(y = missing  | X = anything else) = phi   (silently dropped out)
 *
 *   P(y = α (≥0)   | X = α + 1     ) = 1 - phi  (observed correctly, no dropout)
 *                                                (α=0 means unedited;
 *                                                 alphabet index 1)
 *   P(y = α (≥0)   | X != α + 1    ) = 0        (impossible)
 *
 * Mirrors `laml_model::compute_taxa_log_inside_likelihood` in
 *   /Users/gc3045/Desktop/git/LAML-Pro/src/models/laml.cxx
 * for the "character-matrix" branch.
 *
 * @tparam D Per-node data; must expose log_phi and log_one_minus_phi
 *           (matches laml_data).
 * @tparam O Per-leaf observation type. Defaults to std::vector<int>.
 */
template <typename D = laml_data, typename O = std::vector<int>>
class pmm_categorical_observation_model
    : public observation_model<D, O>
{
public:
    using observation_model<D, O>::sites_latent_states;
    using observation_model<D, O>::sites_observation_states;

    /*!
     * @brief Build the observation model.
     *
     * @param latent_alphabets Per-character latent alphabet sizes; must match
     *                         the paired pmm_ctmc_model's sites_latent_states.
     *                         Index 0 is silenced, index 1 is unedited,
     *                         indices 2..k-1 are mutated categories.
     */
    explicit pmm_categorical_observation_model(
        const std::vector<size_t>& latent_alphabets)
    {
        sites_latent_states.resize(latent_alphabets.size());
        sites_observation_states.resize(latent_alphabets.size());
        for (size_t c = 0; c < latent_alphabets.size(); ++c) {
            sites_latent_states[c].alphabet_size = latent_alphabets[c];
            sites_latent_states[c].missing_index = 0;
            // Observation alphabet = latent alphabet minus the silenced index
            // (the missing observation is encoded as -1, not as an alphabet
            // entry). One slot for "unedited" + (k - 2) mutated categories.
            sites_observation_states[c].alphabet_size = latent_alphabets[c] - 1;
            sites_observation_states[c].missing_index = 0;
        }
    }

    /*!
     * @brief log P(observation at this leaf | latent state = alpha).
     *
     * @param data        Per-node cache; uses data.log_phi and data.log_one_minus_phi.
     * @param alpha       Latent state index in [0, alphabet_size).
     * @param site        Character index.
     * @param observation Per-leaf observation bundle; observation[site] is read.
     */
    double compute_log_emission_probability(
        D&       data,
        size_t   alpha,
        size_t   site,
        const O& observation
    ) const override {
        const int y = observation[site];
        const size_t k = sites_latent_states[site].alphabet_size;
        if (alpha >= k) return NEGATIVE_INFINITY;

        if (y == -1) {
            // Missing observation:
            //   silenced latent (alpha = 0)        → probability 1   (log 0)
            //   any other latent                   → probability phi (log_phi)
            return (alpha == 0) ? 0.0 : data.log_phi;
        }

        // Non-missing observation: the latent state must match y + 1
        // (y = 0 → unedited at alphabet index 1; y = α ≥ 1 → mutated category
        // at alphabet index α + 1). Probability (1 - phi); else 0.
        const size_t target = static_cast<size_t>(y) + 1u;
        return (alpha == target) ? data.log_one_minus_phi : NEGATIVE_INFINITY;
    }
};

} // namespace phylocore

#endif // PHYLOCORE_OBSERVATION_MODELS_PMM_CATEGORICAL_H
