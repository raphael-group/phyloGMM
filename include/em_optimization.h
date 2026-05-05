#ifndef PHYLOCORE_EM_OPTIMIZATION_H
#define PHYLOCORE_EM_OPTIMIZATION_H

#include <algorithm>
#include <vector>

#include "constants.h"
#include "phylogenetic_model.h"
#include "phylogeny.h"

namespace phylocore {

/*!
 * @brief Optimize branch lengths via Expectation-Maximization.
 *
 * Each iteration:
 *   E-step — runs the inside-outside pass and calls model.e_step_branch()
 *            for every (node, character) pair to collect sufficient statistics.
 *   M-step — calls model.m_step_branch() for every node to update branch lengths,
 *            then clamps each length to [BRANCH_LENGTH_LB, BRANCH_LENGTH_UB].
 *
 * Convergence is checked on total log-likelihood after each iteration.
 *
 * Models that do not override e_step_branch / m_step_branch return empty
 * suffstats / unchanged branch lengths, so this function degrades gracefully
 * to a no-op for such models (use optimize_branch_lengths instead).
 *
 * @param t         Tree whose branch lengths are optimized in place
 * @param model     Model providing e_step_branch and m_step_branch
 * @param node_data Per-node model data (may be updated by the model during E/M steps)
 * @param tol       Stop when log-likelihood improvement per iteration falls below this
 * @param max_iter  Maximum number of EM iterations
 * @return          Log-likelihood at the final branch lengths
 */
template <typename D>
double em_optimize_branch_lengths(
    tree&                    t,
    phylogenetic_model<D>&   model,
    std::vector<D>&          node_data,
    double                   tol      = 1e-6,
    int                      max_iter = 100)
{
    size_t num_chars = model.alphabet_sizes.size();
    size_t max_alpha = *std::max_element(
        model.alphabet_sizes.begin(), model.alphabet_sizes.end());

    likelihood_buffer inside_ll    (num_chars, max_alpha, t.num_nodes);
    likelihood_buffer edge_inside_ll(num_chars, max_alpha, t.num_nodes);
    likelihood_buffer outside_ll   (num_chars, max_alpha, t.num_nodes);

    auto total_llh = [&]() {
        auto per_char = compute_inside_log_likelihood(model, t, inside_ll, node_data);
        double s = 0.0;
        for (double v : per_char) s += v;
        return s;
    };

    double llh = total_llh();

    for (int iter = 0; iter < max_iter; ++iter) {
        // E-step: full inside-outside pass
        compute_inside_log_likelihood    (model, t, inside_ll,      node_data);
        compute_edge_inside_log_likelihood(model, t, inside_ll, edge_inside_ll, node_data);
        compute_outside_log_likelihood   (model, t, edge_inside_ll, outside_ll, node_data);

        // Collect sufficient statistics and update each branch
        for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
            std::vector<std::vector<double>> suffstats(num_chars);
            for (size_t c = 0; c < num_chars; ++c) {
                suffstats[c] = model.e_step_branch(
                    node_data[node_id], node_id, c,
                    inside_ll, outside_ll, edge_inside_ll);
            }

            double new_blen = model.m_step_branch(
                node_id, suffstats, t.branch_lengths[node_id]);

            t.branch_lengths[node_id] = std::clamp(new_blen, BRANCH_LENGTH_LB, BRANCH_LENGTH_UB);
        }

        double new_llh = total_llh();
        if (new_llh - llh < tol) { llh = new_llh; break; }
        llh = new_llh;
    }

    return llh;
}

} // namespace phylocore

#endif // PHYLOCORE_EM_OPTIMIZATION_H
