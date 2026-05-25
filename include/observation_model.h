#ifndef PHYLOCORE_OBSERVATION_MODEL_H
#define PHYLOCORE_OBSERVATION_MODEL_H

#include <variant>
#include <vector>

#include "state_space.h"

namespace phylocore {

/*!
 * @brief Abstract base class for emission models.
 *
 * Computes log P(observation at a leaf | latent state = alpha).
 * The caller (traversal) extracts the per-leaf observation from observed_data
 * and passes it in directly; obs_models do not own observation storage.
 *
 * @tparam D Per-node data type (matches state_transition_model<D>'s D).
 * @tparam O Per-leaf observation type. For discrete models, often vector<int>
 *           (per-site states for one leaf). For continuous models, e.g.
 *           Eigen::VectorXd (one d-dim vector per leaf).
 */
template <typename D = std::monostate, typename O = std::vector<int>>
class observation_model {
public:
    // Mirror of state_transition_model::sites_latent_states. Caller sets this
    // in sync with the paired state_transition_model (after stm.build()). Lets
    // the obs_model know what `alpha` ranges over per site without holding a
    // reference to the state_transition_model.
    std::vector<state_space> sites_latent_states;
    std::vector<double>      observation_parameters;

    virtual ~observation_model() = default;

    /*!
     * @brief log P(observation | latent state = alpha) at this leaf.
     *
     * @param data        Per-node data, resolved by the caller from node_data[node_id].
     * @param alpha       Index of the latent state.
     * @param site        Site index.
     * @param observation Per-leaf observation bundle; the model reads the site-relevant
     *                    piece from it (e.g., observation[site] for discrete models).
     */
    virtual double compute_log_emission_probability(
        D&        data,
        size_t    alpha,
        size_t    site,
        const O&  observation
    ) const = 0;
};

} // namespace phylocore

#endif // PHYLOCORE_OBSERVATION_MODEL_H
