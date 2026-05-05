#ifndef PHYLOCORE_PHYLOGENY_H
#define PHYLOCORE_PHYLOGENY_H

#include <memory>
#include <string>
#include <vector>

#include "digraph.h"
#include "math_utilities.h"
#include "phylogenetic_model.h"

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
    size_t num_characters;
    size_t max_alphabet_size;

    likelihood_buffer(size_t num_characters, size_t max_alphabet_size, size_t num_nodes,
                      double fill_value = -1e9)
        : num_nodes(num_nodes),
          num_characters(num_characters),
          max_alphabet_size(max_alphabet_size) {
        buffer.resize(num_nodes * num_characters * max_alphabet_size, fill_value);
    }

    double& operator()(size_t character, size_t node, size_t state) {
        return buffer[node * (num_characters * max_alphabet_size)
                      + character * max_alphabet_size + state];
    }

    const double& operator()(size_t character, size_t node, size_t state) const {
        return buffer[node * (num_characters * max_alphabet_size)
                      + character * max_alphabet_size + state];
    }
};

// ─── Likelihood traversals ────────────────────────────────────────────────────

template <typename D>
std::vector<double> compute_inside_log_likelihood(
    const phylogenetic_model<D>& model,
    const tree& t,
    likelihood_buffer& b,
    std::vector<D>& node_data
) {
    size_t num_characters  = model.alphabet_sizes.size();
    size_t max_alpha       = *std::max_element(model.alphabet_sizes.begin(),
                                               model.alphabet_sizes.end());
    std::vector<double> tmp1(max_alpha), tmp2(max_alpha);

    for (int node_id : t.tree.postorder_traversal(t.root_id)) {
        for (size_t c = 0; c < num_characters; ++c) {
            size_t alpha = model.alphabet_sizes[c];
            if (t.tree.out_degree(node_id) == 0) {
                model.compute_taxa_log_inside_likelihood(
                    node_data[node_id], c, t.tree[node_id].data, tmp1);
                for (size_t j = 0; j < alpha; ++j) b(c, node_id, j) = tmp1[j];
                continue;
            }
            for (size_t j = 0; j < alpha; ++j) b(c, node_id, j) = 0.0;
            for (int u_id : t.tree.successors(node_id)) {
                for (size_t j = 0; j < alpha; ++j) tmp2[j] = b(c, u_id, j);
                model.compute_log_pmatrix_vector_product(
                    node_data[u_id], c, t.branch_lengths[u_id], tmp2, tmp1);
                for (size_t j = 0; j < alpha; ++j) b(c, node_id, j) += tmp1[j];
            }
        }
    }

    std::vector<double> llh(num_characters, 0.0);
    for (size_t c = 0; c < num_characters; ++c) {
        size_t alpha = model.alphabet_sizes[c];
        std::vector<double> root_llh(alpha), buf1(alpha), buf2(alpha);
        for (size_t j = 0; j < alpha; ++j) root_llh[j] = b(c, t.root_id, j);
        model.compute_log_pmatrix_vector_product(
            node_data[t.root_id], c, t.branch_lengths[t.root_id], root_llh, buf1);
        model.compute_root_distribution(node_data[t.root_id], c, buf2);
        for (size_t j = 0; j < alpha; ++j) buf1[j] += buf2[j];
        llh[c] = log_sum_exp(buf1.begin(), buf1.end());
    }
    return llh;
}

template <typename D>
void compute_edge_inside_log_likelihood(
    const phylogenetic_model<D>& model,
    const tree& t,
    const likelihood_buffer& inside_ll,
    likelihood_buffer& b,
    std::vector<D>& node_data
) {
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alpha      = *std::max_element(model.alphabet_sizes.begin(),
                                              model.alphabet_sizes.end());
    std::vector<double> tmp1(max_alpha), tmp2(max_alpha);

    for (int node_id : t.tree.nodes()) {
        for (size_t c = 0; c < num_characters; ++c) {
            size_t alpha = model.alphabet_sizes[c];
            for (size_t j = 0; j < alpha; ++j) tmp2[j] = inside_ll(c, node_id, j);
            model.compute_log_pmatrix_vector_product(
                node_data[node_id], c, t.branch_lengths[node_id], tmp2, tmp1);
            for (size_t j = 0; j < alpha; ++j) b(c, node_id, j) = tmp1[j];
        }
    }
}

template <typename D>
void compute_outside_log_likelihood(
    const phylogenetic_model<D>& model,
    const tree& t,
    const likelihood_buffer& edge_inside_ll,
    likelihood_buffer& b,
    std::vector<D>& node_data
) {
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alpha      = *std::max_element(model.alphabet_sizes.begin(),
                                              model.alphabet_sizes.end());
    std::vector<double> tmp1(max_alpha), tmp2(max_alpha);

    for (size_t c = 0; c < num_characters; ++c) {
        size_t alpha = model.alphabet_sizes[c];
        model.compute_root_distribution(node_data[t.root_id], c, tmp1);
        model.compute_log_pmatrix_transpose_vector_product(
            node_data[t.root_id], c, t.branch_lengths[t.root_id], tmp1, tmp2);
        for (size_t j = 0; j < alpha; ++j) b(c, t.root_id, j) = tmp2[j];
    }

    for (size_t u_id : t.tree.preorder_traversal(t.root_id)) {
        if (t.tree.in_degree(u_id) == 0) continue;
        size_t w_id = t.tree.predecessors(u_id)[0];
        size_t v_id = t.tree.successors(w_id)[0];
        if (v_id == u_id) v_id = t.tree.successors(w_id)[1];

        for (size_t c = 0; c < num_characters; ++c) {
            size_t alpha = model.alphabet_sizes[c];
            for (size_t j = 0; j < alpha; ++j)
                tmp1[j] = edge_inside_ll(c, v_id, j) + b(c, w_id, j);
            model.compute_log_pmatrix_transpose_vector_product(
                node_data[u_id], c, t.branch_lengths[u_id], tmp1, tmp2);
            for (size_t j = 0; j < alpha; ++j) b(c, u_id, j) = tmp2[j];
        }
    }
}

} // namespace phylocore

#endif // PHYLOCORE_PHYLOGENY_H
