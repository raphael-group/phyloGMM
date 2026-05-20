#ifndef PHYLOCORE_STATE_TRANSITION_MODELS_GTR_H
#define PHYLOCORE_STATE_TRANSITION_MODELS_GTR_H

#include <vector>

#include <Eigen/Dense>

#include "../state_transition_model.h"

namespace phylocore {

struct gtr_node_data {};  // GTR has no per-node state

/*!
 * @brief General Time Reversible (GTR) substitution model.
 *
 * Set rates[c], freqs[c] per site, then call build() to:
 *   - compute the eigendecomposition cache per site
 *   - populate sites_latent_states with each site's alphabet_size
 *   - set root_prior = freqs (stationary distribution at root)
 *   - sync state_transition_parameters = [rates[0], freqs[0], rates[1], freqs[1], ...]
 *
 * Only the mandatory scalar primitive compute_log_transition_probability
 * is overridden. has_optimized_functions stays false; the traversal uses the
 * k²-scalar fallback. Pair with identity_observation_model<gtr_node_data>
 * and observed_data of type vector<vector<int>>.
 *
 * Per-site parameterization:
 *   - rates[c]: k*(k-1)/2 exchangeability rates (upper triangle of symmetric S,
 *               row-major: (0,1), (0,2), ..., (0,k-1), (1,2), ..., (k-2,k-1))
 *   - freqs[c]: k stationary frequencies, must sum to 1
 *
 * To update parameters after construction, modify rates/freqs and call build() again.
 */
class gtr_model : public state_transition_model<gtr_node_data> {
public:
    std::vector<std::vector<double>> rates;  // [site][k*(k-1)/2]
    std::vector<std::vector<double>> freqs;  // [site][k], must sum to 1

    void build();

    double compute_log_transition_probability(
        gtr_node_data& data,
        size_t         site,
        double         branch_length,
        size_t         alpha,
        size_t         beta
    ) const override;

private:
    struct site_cache {
        Eigen::MatrixXd U;            // eigenvectors of symmetric similarity transform
        Eigen::VectorXd lambda;       // eigenvalues
        Eigen::VectorXd sqrt_pi;
        Eigen::VectorXd inv_sqrt_pi;
    };
    std::vector<site_cache> cache;  // one per site, filled by build()
};

} // namespace phylocore

#endif // PHYLOCORE_STATE_TRANSITION_MODELS_GTR_H
