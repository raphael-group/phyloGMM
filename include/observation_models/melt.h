#ifndef PHYLOCORE_OBSERVATION_MODELS_MELT_H
#define PHYLOCORE_OBSERVATION_MODELS_MELT_H

#include <stdexcept>
#include <string>
#include <vector>

#include "../constants.h"
#include "../observation_model.h"
#include "../state_transition_models/melt.h"

namespace phylocore {

/*!
 * @brief Per-leaf observation for the MeLT model.
 *
 * methylation[site] in {-1, 0, 1}  (-1 = missing data)
 * cell_type         in {0, ..., K-1}
 */
struct melt_leaf_obs {
    std::vector<int> methylation; // [site]
    int              cell_type;
};

/*!
 * @brief Asymmetric bit-flip observation model for MeLT.
 *
 * The same bit-flip model applies to both lineage-informative and
 * cell-type-informative sites. The "canonical" methylation that the
 * bit-flip corrupts differs by latent state:
 *
 *   Lineage states (latent in {0, 1}):
 *     canonical = latent state index (0 = unmethylated, 1 = methylated)
 *
 *   Cell-type states (latent = 2 + k):
 *     Cell type is observed as a point mass -- if k != obs.cell_type the
 *     emission is impossible (NEGATIVE_INFINITY). If k == obs.cell_type,
 *     canonical = theta[site][k].
 *
 * Bit-flip probabilities:
 *   P(obs=1 | canonical=0) = alpha   (false-positive rate)
 *   P(obs=0 | canonical=0) = 1-alpha
 *   P(obs=0 | canonical=1) = beta    (false-negative rate)
 *   P(obs=1 | canonical=1) = 1-beta
 *
 * Missing observations (methylation[site] == -1) return log-prob 0 for all
 * latent states in the appropriate block.
 *
 * Parameters:
 *   alpha -- false-positive rate,  in (0, 1)
 *   beta  -- false-negative rate,  in (0, 1)
 *   theta -- [site][cell_type] in {0,1}: canonical methylation per cell type
 *
 * Usage:
 *   1. Set alpha, beta, theta, and sites_latent_states (synced from the
 *      paired melt_ctmc_model after its build() call).
 *   2. Call build() to validate parameters and cache log values.
 *   3. Pass to compute_inside_log_likelihood together with the CTMC model.
 */
class melt_observation_model
    : public observation_model<melt_node_data, melt_leaf_obs> {
public:
    // ── Parameters ───────────────────────────────────────────────────────────

    double alpha = 0.01; // P(obs=1 | canonical=0)  false-positive rate
    double beta  = 0.01; // P(obs=0 | canonical=1)  false-negative rate

    // theta[site][cell_type] in {0,1}: canonical methylation for cell type k
    // at site s (used when the leaf's cell-type state matches the latent state).
    std::vector<std::vector<int>> theta;

    // ── Construction ─────────────────────────────────────────────────────────

    /*!
     * @brief Validate alpha and beta and cache their log values.
     *
     * Call after setting alpha, beta, theta, and sites_latent_states.
     */
    void build();

    // ── Required interface ────────────────────────────────────────────────────

    double compute_log_emission_probability(
        melt_node_data&      data,
        size_t               latent_state,
        size_t               site,
        const melt_leaf_obs& obs
    ) const override;

private:
    double log_alpha;
    double log_1m_alpha;
    double log_beta;
    double log_1m_beta;

    // Returns log P(obs_meth | canonical) using the bit-flip model.
    // Returns 0.0 when obs_meth == -1 (missing).
    double log_bitflip(int obs_meth, int canonical) const;
};

} // namespace phylocore

#endif // PHYLOCORE_OBSERVATION_MODELS_MELT_H
