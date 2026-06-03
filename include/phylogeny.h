#ifndef PHYLOCORE_PHYLOGENY_H
#define PHYLOCORE_PHYLOGENY_H

#include <memory>
#include <string>
#include <vector>

#include "constants.h"
#include "digraph.h"
#include "math_utilities.h"
#include "meta_data.h"
#include "observation_model.h"
#include "state_transition_model.h"

namespace phylocore {

/*!
 * @brief A rooted phylogenetic tree.
 *
 * Leaves occupy IDs 0 to num_leaves-1; internal nodes from num_leaves onward.
 * branch_lengths[v] is the length of the edge leading into node v.
 */
struct tree {
    size_t num_leaves;
    size_t num_nodes;
    size_t root_id;
    digraph<size_t> tree;
    std::vector<double> branch_lengths;
    std::vector<std::string> node_names;
};

/*!
 * @brief Flat 3-D buffer (characters × nodes × states) for log-likelihood values.
 *
 * Used to store inside and outside log-probabilities during likelihood traversals.
 */
class likelihood_buffer {
    std::vector<double> buffer;

public:
    size_t num_nodes;
    size_t num_sites;
    size_t max_alphabet_size;

    likelihood_buffer(size_t num_sites, size_t max_alphabet_size, size_t num_nodes,
                      double fill_value = -1e9)
        : num_nodes(num_nodes),
          num_sites(num_sites),
          max_alphabet_size(max_alphabet_size) {
        buffer.resize(num_nodes * num_sites * max_alphabet_size, fill_value);
    }

    double& operator()(size_t character, size_t node, size_t state) {
        return buffer[node * (num_sites * max_alphabet_size)
                      + character * max_alphabet_size + state];
    }

    const double& operator()(size_t character, size_t node, size_t state) const {
        return buffer[node * (num_sites * max_alphabet_size)
                      + character * max_alphabet_size + state];
    }
};

// ─── Helper: apply log P(t) or log P(t)^T to a vector ───────────────────────
//
// If the model has set has_optimized_functions = true, dispatches to its
// optimized matvec (forward or transpose). Otherwise falls back to the
// scalar primitive in a k² loop. log_terms is a caller-provided scratch
// buffer (must have size >= alphabet_size at this site).

template <typename D>
inline void apply_log_transition(
    const state_transition_model<D>& state_transition_model,
    D& data,
    size_t site,
    double branch_length,
    const std::vector<double>& log_v,
    std::vector<double>& result,
    std::vector<double>& log_terms,
    bool transpose)
{
    if (state_transition_model.has_optimized_functions) {
        state_transition_model.compute_log_pmatrix_vector_product(
            data, site, branch_length, log_v, result, transpose);
        return;
    }
    size_t k = state_transition_model.sites_latent_states[site].alphabet_size;
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < k; ++j) {
            size_t a = transpose ? j : i;
            size_t b = transpose ? i : j;
            log_terms[j] = state_transition_model.compute_log_transition_probability(
                data, site, branch_length, a, b) + log_v[j];
        }
        result[i] = log_sum_exp(log_terms.begin(), log_terms.begin() + k);
    }
}

// ─── Likelihood traversals ────────────────────────────────────────────────────

template <typename D, typename O>
std::vector<double> compute_inside_log_likelihood(
    const state_transition_model<D>&  state_transition_model,
    const observation_model<D, O>&    obs_model,
    const std::vector<O>&             observed_data,
    const meta_data&                  metadata,
    const tree& t,
    likelihood_buffer& b,
    std::vector<D>& node_data
) {
    size_t num_sites = state_transition_model.sites_latent_states.size();
    size_t max_alpha = std::max_element(
        state_transition_model.sites_latent_states.begin(),
        state_transition_model.sites_latent_states.end(),
        [](const state_space& a, const state_space& b) {
            return a.alphabet_size < b.alphabet_size;
        })->alphabet_size;
    std::vector<double> tmp1(max_alpha), tmp2(max_alpha), log_terms(max_alpha);

    for (int node_id : t.tree.postorder_traversal(t.root_id)) {
        for (size_t c = 0; c < num_sites; ++c) {
            size_t alpha = state_transition_model.sites_latent_states[c].alphabet_size;
            if (t.tree.out_degree(node_id) == 0) {
                size_t taxa_id = t.tree[node_id].data;
                const O& obs = observed_data[taxa_id];
                for (size_t j = 0; j < alpha; ++j)
                    b(c, node_id, j) = obs_model.compute_log_emission_probability(
                        node_data[node_id], j, c, obs);
                // Hard-label clamp (no-op when metadata.hard_labels is empty).
                // When set, force the leaf's inside likelihood to a delta at
                // hard_labels[taxa_id]. The conditional distribution propagates
                // to internal nodes via the postorder pass below.
                if (!metadata.hard_labels.empty()
                    && taxa_id < metadata.hard_labels.size())
                {
                    const int j_star = metadata.hard_labels[taxa_id];
                    if (j_star >= 0) {
                        for (size_t j = 0; j < alpha; ++j) {
                            b(c, node_id, j) =
                                (static_cast<int>(j) == j_star)
                                    ? 0.0 : NEGATIVE_INFINITY;
                        }
                    }
                }
                continue;
            }
            for (size_t j = 0; j < alpha; ++j) b(c, node_id, j) = 0.0;
            for (int u_id : t.tree.successors(node_id)) {
                for (size_t j = 0; j < alpha; ++j) tmp2[j] = b(c, u_id, j);
                apply_log_transition(
                    state_transition_model, node_data[u_id], c, t.branch_lengths[u_id],
                    tmp2, tmp1, log_terms, /*transpose=*/false);
                for (size_t j = 0; j < alpha; ++j) b(c, node_id, j) += tmp1[j];
            }
        }
    }

    std::vector<double> llh(num_sites, 0.0);
    for (size_t c = 0; c < num_sites; ++c) {
        size_t alpha = state_transition_model.sites_latent_states[c].alphabet_size;
        std::vector<double> root_llh(alpha), buf1(alpha);
        for (size_t j = 0; j < alpha; ++j) root_llh[j] = b(c, t.root_id, j);
        apply_log_transition(
            state_transition_model, node_data[t.root_id], c, t.branch_lengths[t.root_id],
            root_llh, buf1, log_terms, /*transpose=*/false);
        for (size_t j = 0; j < alpha; ++j)
            buf1[j] += (state_transition_model.root_prior[c][j] > 0.0)
                ? std::log(state_transition_model.root_prior[c][j])
                : NEGATIVE_INFINITY;
        llh[c] = log_sum_exp(buf1.begin(), buf1.end());
    }
    return llh;
}

template <typename D>
void compute_edge_inside_log_likelihood(
    const state_transition_model<D>& state_transition_model,
    const tree& t,
    const likelihood_buffer& inside_ll,
    likelihood_buffer& b,
    std::vector<D>& node_data
) {
    size_t num_sites = state_transition_model.sites_latent_states.size();
    size_t max_alpha = std::max_element(
        state_transition_model.sites_latent_states.begin(),
        state_transition_model.sites_latent_states.end(),
        [](const state_space& a, const state_space& b) {
            return a.alphabet_size < b.alphabet_size;
        })->alphabet_size;
    std::vector<double> tmp1(max_alpha), tmp2(max_alpha), log_terms(max_alpha);

    for (int node_id : t.tree.nodes()) {
        for (size_t c = 0; c < num_sites; ++c) {
            size_t alpha = state_transition_model.sites_latent_states[c].alphabet_size;
            for (size_t j = 0; j < alpha; ++j) tmp2[j] = inside_ll(c, node_id, j);
            apply_log_transition(
                state_transition_model, node_data[node_id], c, t.branch_lengths[node_id],
                tmp2, tmp1, log_terms, /*transpose=*/false);
            for (size_t j = 0; j < alpha; ++j) b(c, node_id, j) = tmp1[j];
        }
    }
}

template <typename D>
void compute_outside_log_likelihood(
    const state_transition_model<D>& state_transition_model,
    const tree& t,
    const likelihood_buffer& edge_inside_ll,
    likelihood_buffer& b,
    std::vector<D>& node_data
) {
    size_t num_sites = state_transition_model.sites_latent_states.size();
    size_t max_alpha = std::max_element(
        state_transition_model.sites_latent_states.begin(),
        state_transition_model.sites_latent_states.end(),
        [](const state_space& a, const state_space& b) {
            return a.alphabet_size < b.alphabet_size;
        })->alphabet_size;
    std::vector<double> tmp1(max_alpha), tmp2(max_alpha), log_terms(max_alpha);

    for (size_t c = 0; c < num_sites; ++c) {
        size_t alpha = state_transition_model.sites_latent_states[c].alphabet_size;
        for (size_t j = 0; j < alpha; ++j)
            tmp1[j] = (state_transition_model.root_prior[c][j] > 0.0)
                ? std::log(state_transition_model.root_prior[c][j])
                : NEGATIVE_INFINITY;
        apply_log_transition(
            state_transition_model, node_data[t.root_id], c, t.branch_lengths[t.root_id],
            tmp1, tmp2, log_terms, /*transpose=*/true);
        for (size_t j = 0; j < alpha; ++j) b(c, t.root_id, j) = tmp2[j];
    }

    for (size_t u_id : t.tree.preorder_traversal(t.root_id)) {
        if (t.tree.in_degree(u_id) == 0) continue;
        size_t w_id = t.tree.predecessors(u_id)[0];

        for (size_t c = 0; c < num_sites; ++c) {
            size_t alpha = state_transition_model.sites_latent_states[c].alphabet_size;
            // outside(u) ∝ sum_{state_w} T(state_u | state_w) · outside(w, state_w)
            //                · product_{sib of u} edge_inside(sib, state_w)
            // Start with outside(w), then in log-space add every sibling's
            // edge_inside contribution. Works for any arity at w (unary,
            // binary, multifurcating). The previous implementation hard-coded
            // exactly two children and read successors(w)[1] when w had only
            // one — undefined behaviour on unary internal nodes (notably the
            // unary roots emitted by some simulators and ete3 NHX trees).
            for (size_t j = 0; j < alpha; ++j) tmp1[j] = b(c, w_id, j);
            for (int sib_id : t.tree.successors(w_id)) {
                if (static_cast<size_t>(sib_id) == u_id) continue;
                for (size_t j = 0; j < alpha; ++j)
                    tmp1[j] += edge_inside_ll(c, sib_id, j);
            }
            apply_log_transition(
                state_transition_model, node_data[u_id], c, t.branch_lengths[u_id],
                tmp1, tmp2, log_terms, /*transpose=*/true);
            for (size_t j = 0; j < alpha; ++j) b(c, u_id, j) = tmp2[j];
        }
    }
}

} // namespace phylocore

#endif // PHYLOCORE_PHYLOGENY_H
