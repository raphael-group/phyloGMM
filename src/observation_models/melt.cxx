#include "observation_models/melt.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "constants.h"

namespace phylocore {

// ── build ─────────────────────────────────────────────────────────────────────

void melt_observation_model::build() {
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::runtime_error(
            "melt_observation_model: alpha must be in (0, 1), got " +
            std::to_string(alpha));
    if (beta <= 0.0 || beta >= 1.0)
        throw std::runtime_error(
            "melt_observation_model: beta must be in (0, 1), got " +
            std::to_string(beta));

    log_alpha    = std::log(alpha);
    log_1m_alpha = std::log(1.0 - alpha);
    log_beta     = std::log(beta);
    log_1m_beta  = std::log(1.0 - beta);
}

// ── log_bitflip ───────────────────────────────────────────────────────────────
//
// Returns log P(obs_meth | canonical) under the asymmetric bit-flip model:
//
//   canonical = 0:  P(obs=0) = 1-alpha,  P(obs=1) = alpha
//   canonical = 1:  P(obs=0) = beta,     P(obs=1) = 1-beta
//
// Missing observation (obs_meth == -1) marginalizes to log-prob 0.

double melt_observation_model::log_bitflip(int obs_meth, int canonical) const {
    if (obs_meth == -1) return 0.0;
    if (canonical == 0)
        return obs_meth == 1 ? log_alpha : log_1m_alpha;
    else
        return obs_meth == 0 ? log_beta  : log_1m_beta;
}

// ── compute_log_emission_probability ─────────────────────────────────────────
//
// Latent state layout (K+2 states):
//   0, 1       -- lineage methylation (unmethylated, methylated)
//   2, ..., K+1 -- cell-type identity (cell types 0, ..., K-1)
//
// Lineage states (latent_state in {0, 1}):
//   canonical = latent_state; apply bit-flip on obs.methylation[site].
//
// Cell-type states (latent_state = 2 + j):
//   Cell type is observed as a point mass:
//     - j != obs.cell_type  =>  NEGATIVE_INFINITY (wrong cell type)
//     - j == obs.cell_type  =>  canonical = theta[site][j]; apply bit-flip.

double melt_observation_model::compute_log_emission_probability(
    melt_node_data&      /*data*/,
    size_t               latent_state,
    size_t               site,
    const melt_leaf_obs& obs
) const {
    int m = obs.methylation[site]; // -1, 0, or 1

    // Lineage states
    if (latent_state < 2)
        return log_bitflip(m, static_cast<int>(latent_state));

    // Cell-type states
    size_t j = latent_state - 2;
    if (static_cast<int>(j) != obs.cell_type)
        return NEGATIVE_INFINITY;

    int canonical = theta[site][j];
    return log_bitflip(m, canonical);
}

} // namespace phylocore
