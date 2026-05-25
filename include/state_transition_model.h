#ifndef PHYLOCORE_STATE_TRANSITION_MODEL_H
#define PHYLOCORE_STATE_TRANSITION_MODEL_H

#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "state_space.h"

namespace phylocore {

// Forward declarations — full definitions are in phylogeny.h.
// Avoids a circular dependency since phylogeny.h includes this header.
class likelihood_buffer;
struct tree;

/*!
 * @brief Abstract base class for substitution models used in phylogenetic likelihood.
 *
 * To define a new model, inherit from this class with your per-node data type D
 * and implement the single pure virtual compute_log_transition_probability —
 * the model's mathematical primitive. Populate sites_latent_states and
 * root_prior in build()/constructor. You then get likelihood computation,
 * branch length optimization (Brent's), and topology search for free.
 *
 * For speed-conscious models, additionally override the matvec methods and set
 * has_optimized_functions = true so the traversal uses them instead of looping
 * over the scalar primitive.
 *
 * Parameter estimation (EM, branch optimization, etc.) lives separately under
 * `include/parameter_estimation/` as free-function overloads, not virtuals on
 * this base. See docs/parameter_estimation.md.
 *
 * @tparam D Model-specific per-node data type. Use an empty struct for homogeneous
 *           models where no per-node state is needed.
 */
template <typename D = std::monostate>
class state_transition_model {
public:
    std::vector<state_space>          sites_latent_states; // one state_space per site
    std::vector<std::vector<double>> root_prior;          // per-site root prior (raw probabilities); root_prior[c][j] = P(root state j at site c)
    std::vector<double>              state_transition_parameters;          // non-branch-length numeric model parameters

    // If true, the model has provided an optimized implementation of
    // compute_log_transition_probability_vectorized. The traversal uses it
    // instead of looping over the scalar primitive. Set in the concrete model's
    // build()/constructor when the vectorized method is overridden and
    // mathematically consistent with the scalar.
    bool has_optimized_functions = false;

    virtual ~state_transition_model() = default;

    /*!
     * @brief Set the per-site root prior.
     *
     * @param prior outer size must equal sites_latent_states.size();
     *              each inner vector's size must equal that site's alphabet_size.
     * @throws std::invalid_argument on dimension mismatch.
     */
    void set_root_prior(std::vector<std::vector<double>> prior) {
        if (prior.size() != sites_latent_states.size())
            throw std::invalid_argument(
                "set_root_prior: outer size " + std::to_string(prior.size()) +
                " does not match num_sites " + std::to_string(sites_latent_states.size()));
        for (size_t c = 0; c < prior.size(); ++c) {
            if (prior[c].size() != sites_latent_states[c].alphabet_size)
                throw std::invalid_argument(
                    "set_root_prior: site " + std::to_string(c) +
                    " inner size " + std::to_string(prior[c].size()) +
                    " does not match alphabet_size " +
                    std::to_string(sites_latent_states[c].alphabet_size));
        }
        root_prior = std::move(prior);
    }

    // ── Required primitive ────────────────────────────────────────────────────

    /*!
     * @brief log P(alpha → beta along a branch of length t) at site s.
     *
     * The model's mathematical primitive — the only transition method a new
     * model must implement. Return NEGATIVE_INFINITY for impossible transitions.
     */
    virtual double compute_log_transition_probability(
        D& data,
        size_t site,
        double branch_length,
        size_t alpha,
        size_t beta
    ) const = 0;

    // ── Optional matvec optimization ──────────────────────────────────────────
    //
    // Only called by the traversal when has_optimized_functions == true.
    // Override (and set has_optimized_functions = true) to provide a fast
    // path. The override MUST be mathematically consistent with
    // compute_log_transition_probability.

    /*!
     * @brief Apply log P^c(t) or its transpose to a log-vector.
     *
     * If transpose is false: result[i] = log Σⱼ P^c(t)_ij · exp(log_v[j])
     * If transpose is true:  result[i] = log Σⱼ P^c(t)_ji · exp(log_v[j])
     */
    virtual void compute_log_pmatrix_vector_product(
        D& data,
        size_t site,
        double branch_length,
        const std::vector<double>& log_vector,
        std::vector<double>& result,
        bool transpose
    ) const {}

};

} // namespace phylocore

#endif // PHYLOCORE_STATE_TRANSITION_MODEL_H
