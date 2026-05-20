#ifndef PHYLOCORE_OBSERVATION_MODELS_IDENTITY_H
#define PHYLOCORE_OBSERVATION_MODELS_IDENTITY_H

#include <variant>
#include <vector>

#include "../constants.h"
#include "../observation_model.h"

namespace phylocore {

/*!
 * @brief Observation model for "the observation IS the latent state."
 *
 * The per-leaf observation is a vector<int> indexed by site:
 *   - observation[site] == -1: missing — returns 0 (uniform across all alphas)
 *   - observation[site] >= 0:  returns 0 if alpha matches, NEGATIVE_INFINITY otherwise
 *
 * Templated on D so it can pair with any state_transition_model<D>. D is
 * unused inside the body — emission is independent of per-node state.
 */
template <typename D = std::monostate>
class identity_observation_model : public observation_model<D, std::vector<int>> {
public:
    double compute_log_emission_probability(
        D& /*data*/,
        size_t alpha,
        size_t site,
        const std::vector<int>& observation
    ) const override {
        int obs = observation[site];
        if (obs == -1) return 0.0;
        return (alpha == static_cast<size_t>(obs)) ? 0.0 : NEGATIVE_INFINITY;
    }
};

} // namespace phylocore

#endif // PHYLOCORE_OBSERVATION_MODELS_IDENTITY_H
