#ifndef PHYLOCORE_EM_OPTIMIZATION_H
#define PHYLOCORE_EM_OPTIMIZATION_H

#include <algorithm>
#include <vector>

#include "constants.h"
#include "observation_model.h"
#include "state_transition_model.h"
#include "phylogeny.h"

namespace phylocore {

/*!
 * @brief Optimize model parameters via Expectation-Maximization.
 *
 * Each iteration:
 *   E-step — runs the full inside-outside pass, then calls model.e_step() to
 *            collect sufficient statistics. The format of suffstats is entirely
 *            model-defined.
 *   M-step — calls model.m_step(), which may update branch lengths, model
 *            parameters, node_data, or any combination.
 *
 * Convergence is checked on total log-likelihood after each iteration.
 *
 * Models that do not override e_step / m_step leave all parameters unchanged
 * and em_optimize returns immediately after one iteration; use
 * optimize_branch_lengths() for those models instead.
 *
 * @param t         Tree (may be modified by m_step)
 * @param model     Model providing e_step and m_step
 * @param node_data Per-node model data (may be modified by e_step / m_step)
 * @param tol       Stop when log-likelihood improvement falls below this
 * @param max_iter  Maximum EM iterations
 * @return          Log-likelihood at the final parameters
 */
template <typename D>
double em_optimize(
    tree&                        t,
    state_transition_model<D>&       state_transition_model,
    const observation_model<D>&  obs_model,
    std::vector<D>&              node_data,
    double                       tol      = 1e-6,
    int                          max_iter = 100)
{
    size_t num_chars = state_transition_model.sites_latent_states.size();
    size_t max_alpha = std::max_element(
        state_transition_model.sites_latent_states.begin(),
        state_transition_model.sites_latent_states.end(),
        [](const state_space& a, const state_space& b) {
            return a.alphabet_size < b.alphabet_size;
        })->alphabet_size;

    likelihood_buffer inside_ll    (num_chars, max_alpha, t.num_nodes);
    likelihood_buffer edge_inside_ll(num_chars, max_alpha, t.num_nodes);
    likelihood_buffer outside_ll   (num_chars, max_alpha, t.num_nodes);

    auto total_llh = [&]() {
        auto per_char = compute_inside_log_likelihood(state_transition_model, obs_model, t, inside_ll, node_data);
        double s = 0.0;
        for (double v : per_char) s += v;
        return s;
    };

    double llh = total_llh();

    for (int iter = 0; iter < max_iter; ++iter) {
        compute_inside_log_likelihood    (state_transition_model, obs_model, t, inside_ll,       node_data);
        compute_edge_inside_log_likelihood(state_transition_model, t, inside_ll,  edge_inside_ll, node_data);
        compute_outside_log_likelihood   (state_transition_model, t, edge_inside_ll, outside_ll,  node_data);

        std::vector<double> suffstats;
        state_transition_model.e_step(t, node_data, inside_ll, outside_ll, edge_inside_ll, suffstats);
        state_transition_model.m_step(t, node_data, suffstats);

        double new_llh = total_llh();
        if (new_llh - llh < tol) { llh = new_llh; break; }
        llh = new_llh;
    }

    return llh;
}

} // namespace phylocore

#endif // PHYLOCORE_EM_OPTIMIZATION_H
