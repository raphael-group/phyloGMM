// Synthetic data simulator for the phyloGMM model.
//
// phyloGMM: a discrete-state CTMC {X_v}_{v in V} evolves on a rooted tree.
//   - Root state drawn from pi.
//   - Each edge applies P^(e)(t_e) = exp(t_e * Q), where Q is a GENERAL
//     (non-reversible) CTMC rate matrix.
//   - At each leaf l, y_l | X_l = i ~ Normal(mu_i, Sigma_i) with full d x d
//     symmetric positive definite Sigma_i.
//
// The simulator forward-samples (X_v, y_l) from a known ground_truth so EM
// drivers can attempt to recover (Q, {mu_i}, {Sigma_i}, pi). Header-only;
// pulled in by tests/phylo_gmm/simulate.cxx and downstream EM tests.

#ifndef PHYLOCORE_TESTS_PHYLO_GMM_SIMULATOR_H
#define PHYLOCORE_TESTS_PHYLO_GMM_SIMULATOR_H

#include <algorithm>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "digraph.h"
#include "meta_data.h"
#include "phylogeny.h"

namespace phylocore::tests::phylo_gmm {

// Ground-truth phyloGMM parameters shared across all simulated trees.
struct ground_truth {
    Eigen::MatrixXd              Q;      // k x k general CTMC rate matrix
    std::vector<Eigen::VectorXd> mu;     // size k, each d-dim
    std::vector<Eigen::MatrixXd> Sigma;  // size k, each d x d symmetric PSD
    Eigen::VectorXd              pi;     // k-dim, sums to 1
};

// Output of a (possibly multi-tree) simulation run.
//
// observed_data is indexed by GLOBAL taxa_id; across multiple trees the leaf
// taxa_ids are disjoint and contiguous, matching the convention used by
// io.cxx's process_input_data (each leaf is its own row in observed_data).
struct simulated_dataset {
    std::vector<phylocore::tree>     trees;
    std::vector<std::vector<size_t>> latent_per_tree;   // [tree_i][node_id] -> X_v in [0, k)
    std::vector<Eigen::VectorXd>     observed_data;     // indexed by global taxa_id
    phylocore::meta_data             metadata;
};

// ── Internal helpers ────────────────────────────────────────────────────────

namespace detail {

inline void validate_ground_truth(const ground_truth& gt) {
    const auto k = static_cast<Eigen::Index>(gt.pi.size());
    if (k <= 0) throw std::runtime_error("ground_truth: pi is empty");
    if (gt.Q.rows() != k || gt.Q.cols() != k)
        throw std::runtime_error("ground_truth: Q must be k x k");
    if (static_cast<Eigen::Index>(gt.mu.size()) != k)
        throw std::runtime_error("ground_truth: mu must have k entries");
    if (static_cast<Eigen::Index>(gt.Sigma.size()) != k)
        throw std::runtime_error("ground_truth: Sigma must have k entries");
    const Eigen::Index d = gt.mu[0].size();
    for (Eigen::Index i = 0; i < k; ++i) {
        if (gt.mu[i].size() != d)
            throw std::runtime_error("ground_truth: mu[i] dimension mismatch");
        if (gt.Sigma[i].rows() != d || gt.Sigma[i].cols() != d)
            throw std::runtime_error("ground_truth: Sigma[i] dimension mismatch");
    }
}

// Draw an index in [0, k) from a categorical distribution given by probs.
// Robust to small negative noise / sums slightly off 1.
inline size_t sample_categorical(const Eigen::VectorXd& probs,
                                 std::mt19937& rng) {
    const Eigen::Index k = probs.size();
    double total = 0.0;
    for (Eigen::Index i = 0; i < k; ++i) {
        total += std::max(0.0, probs[i]);
    }
    if (total <= 0.0) {
        throw std::runtime_error(
            "sample_categorical: probability vector sums to zero");
    }
    std::uniform_real_distribution<double> u(0.0, total);
    const double r = u(rng);
    double acc = 0.0;
    for (Eigen::Index i = 0; i < k; ++i) {
        acc += std::max(0.0, probs[i]);
        if (r <= acc) return static_cast<size_t>(i);
    }
    return static_cast<size_t>(k - 1);
}

}  // namespace detail

// ── Random tree construction ────────────────────────────────────────────────
//
// Builds a rooted binary tree with `num_leaves` leaves by repeatedly joining
// two random unattached subtree-roots into a new internal node. Leaf node_ids
// are 0..num_leaves-1 (matching phylocore convention); internal node_ids
// continue from num_leaves; the LAST node added is the root.
//
// `start_taxa_id` is added to each leaf's taxa_id, so multi-tree callers can
// assign DISJOINT taxa ids across trees. Internal node `.data` is set to
// num_leaves + internal_offset; it is never consulted by the simulator but
// kept consistent with the phylocore convention.
//
// Branch lengths are drawn from Exponential(1 / mean_branch_length).

inline phylocore::tree simulate_random_binary_tree(size_t num_leaves,
                                                   std::mt19937& rng,
                                                   double mean_branch_length = 0.5,
                                                   size_t start_taxa_id = 0) {
    if (num_leaves < 2) {
        throw std::runtime_error(
            "simulate_random_binary_tree: need num_leaves >= 2");
    }

    const size_t num_internal = num_leaves - 1;  // rooted binary
    const size_t num_nodes = num_leaves + num_internal;

    phylocore::digraph<size_t> g;
    std::vector<double>      branch_lengths(num_nodes, 0.0);
    std::vector<std::string> node_names(num_nodes);

    // Add leaves first (node_ids 0..num_leaves-1).
    for (size_t i = 0; i < num_leaves; ++i) {
        const int id = g.add_vertex(start_taxa_id + i);
        node_names[id] = "L" + std::to_string(start_taxa_id + i);
    }

    std::exponential_distribution<double> br_dist(1.0 / mean_branch_length);

    // Active set: roots of currently-unattached subtrees.
    std::vector<int> active;
    active.reserve(num_leaves);
    for (size_t i = 0; i < num_leaves; ++i) active.push_back(static_cast<int>(i));

    // Repeatedly join two random subtrees under a new internal node.
    size_t internal_offset = 0;
    while (active.size() > 1) {
        std::uniform_int_distribution<size_t> pick(0, active.size() - 1);
        const size_t ia = pick(rng);
        int a = active[ia];
        active[ia] = active.back();
        active.pop_back();

        std::uniform_int_distribution<size_t> pick2(0, active.size() - 1);
        const size_t ib = pick2(rng);
        int b = active[ib];
        active[ib] = active.back();
        active.pop_back();

        // Per phylocore convention internal node.data is just an index.
        const int parent = g.add_vertex(num_leaves + internal_offset);
        node_names[parent] = "N" + std::to_string(internal_offset);
        g.add_edge(parent, a);
        g.add_edge(parent, b);

        branch_lengths[a] = br_dist(rng);
        branch_lengths[b] = br_dist(rng);
        ++internal_offset;

        active.push_back(parent);
    }

    const size_t root_id = static_cast<size_t>(active[0]);
    // phyloGMM draws the topological root directly from pi; it has no incoming
    // root stem. Keep the otherwise edge-indexed length slot at exactly zero.
    branch_lengths[root_id] = 0.0;

    phylocore::tree t{num_leaves, num_nodes, root_id, std::move(g),
                      std::move(branch_lengths), std::move(node_names)};
    return t;
}

// ── Latent-state forward simulation ────────────────────────────────────────
//
// Preorder traversal from root. Root state drawn from pi; each child v of
// parent u has state drawn from row X_u of P^(e)(t_e) = exp(t_e * Q).

inline std::vector<size_t> simulate_latent_states(const phylocore::tree& t,
                                                   const ground_truth& gt,
                                                   std::mt19937& rng) {
    detail::validate_ground_truth(gt);
    std::vector<size_t> X(t.num_nodes, 0);

    // Sample root.
    X[t.root_id] = detail::sample_categorical(gt.pi, rng);

    // Preorder over the tree; for each non-root node, transition from parent.
    for (int u_id : t.tree.preorder_traversal(static_cast<int>(t.root_id))) {
        const auto& preds = t.tree.predecessors(u_id);
        if (preds.empty()) continue;  // root, already sampled
        const int parent = preds[0];
        const double tau = t.branch_lengths[u_id];

        // P(tau) = exp(tau * Q). Eigen's MatrixExponential lives under
        // <unsupported/Eigen/MatrixFunctions>.
        const Eigen::MatrixXd P = (tau * gt.Q).exp();

        const size_t xp = X[parent];
        Eigen::VectorXd row = P.row(static_cast<Eigen::Index>(xp)).transpose();
        X[u_id] = detail::sample_categorical(row, rng);
    }
    return X;
}

// ── Emission ───────────────────────────────────────────────────────────────
//
// y = mu_i + L * z, where L = chol(Sigma_i) (lower triangular) and
// z ~ N(0, I_d). Sigma_i must be symmetric positive definite.

inline Eigen::VectorXd simulate_emission(size_t latent_state,
                                          const ground_truth& gt,
                                          std::mt19937& rng) {
    if (latent_state >= gt.mu.size()) {
        throw std::runtime_error("simulate_emission: latent_state out of range");
    }
    const Eigen::VectorXd& mu = gt.mu[latent_state];
    const Eigen::MatrixXd& Sigma = gt.Sigma[latent_state];

    Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error(
            "simulate_emission: Sigma is not positive definite for state "
            + std::to_string(latent_state));
    }
    const Eigen::MatrixXd L = llt.matrixL();

    const Eigen::Index d = mu.size();
    Eigen::VectorXd z(d);
    std::normal_distribution<double> nrm(0.0, 1.0);
    for (Eigen::Index i = 0; i < d; ++i) z[i] = nrm(rng);
    return mu + L * z;
}

// ── Multi-tree convenience ────────────────────────────────────────────────

inline simulated_dataset simulate_multi_tree(
    const ground_truth& gt,
    const std::vector<size_t>& leaves_per_tree,
    std::mt19937& rng,
    double mean_branch_length = 0.5) {

    detail::validate_ground_truth(gt);
    if (leaves_per_tree.empty()) {
        throw std::runtime_error(
            "simulate_multi_tree: leaves_per_tree must be non-empty");
    }

    simulated_dataset out;
    out.trees.reserve(leaves_per_tree.size());
    out.latent_per_tree.reserve(leaves_per_tree.size());

    // Disjoint taxa-id scheme: each tree owns a contiguous block.
    size_t total_leaves = 0;
    for (size_t n : leaves_per_tree) total_leaves += n;
    out.observed_data.resize(total_leaves);

    const Eigen::Index d = gt.mu[0].size();

    size_t start_taxa_id = 0;
    for (size_t n : leaves_per_tree) {
        phylocore::tree t = simulate_random_binary_tree(
            n, rng, mean_branch_length, start_taxa_id);
        std::vector<size_t> X = simulate_latent_states(t, gt, rng);

        // Draw emissions for leaves only; place them at global taxa_id.
        for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
            if (t.tree.out_degree(static_cast<int>(node_id)) != 0) continue;
            const size_t taxa_id = t.tree[static_cast<int>(node_id)].data;
            out.observed_data[taxa_id] = simulate_emission(X[node_id], gt, rng);
        }

        out.trees.push_back(std::move(t));
        out.latent_per_tree.push_back(std::move(X));
        start_taxa_id += n;
    }

    // Metadata: phyloGMM observations are continuous d-vectors per taxon.
    // We populate the alphabet bookkeeping with k (the discrete latent
    // alphabet) since num_sites / alphabet_sizes are about the LATENT side;
    // observation dimensionality d is carried implicitly by observed_data.
    const size_t k = static_cast<size_t>(gt.pi.size());
    out.metadata.num_sites = 1;
    out.metadata.max_alphabet_size = k;
    out.metadata.alphabet_sizes = {k};
    out.metadata.taxa_names.reserve(total_leaves);
    for (size_t i = 0; i < total_leaves; ++i) {
        out.metadata.taxa_names.push_back("L" + std::to_string(i));
    }
    (void)d;  // dimensionality is implicit in observed_data[i].size()
    return out;
}

// ── Single-tree convenience ────────────────────────────────────────────────

inline simulated_dataset simulate(const ground_truth& gt,
                                   size_t num_leaves,
                                   std::mt19937& rng,
                                   double mean_branch_length = 0.5) {
    return simulate_multi_tree(gt, {num_leaves}, rng, mean_branch_length);
}

}  // namespace phylocore::tests::phylo_gmm

#endif  // PHYLOCORE_TESTS_PHYLO_GMM_SIMULATOR_H
