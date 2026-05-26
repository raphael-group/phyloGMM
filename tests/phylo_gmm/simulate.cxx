// simulate_phylo_gmm: CLI driver for the phyloGMM synthetic data simulator.
//
// Defines a small hardcoded ground truth (k = 3 latent states, d = 2
// observation dimensions) with well-separated cluster means and anisotropic
// covariances, then forward-samples a multi-tree dataset and dumps a summary.
//
// Usage: simulate_phylo_gmm [seed]

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "phylogeny.h"

#include "../../tests/phylo_gmm/simulator.h"

using phylocore::tests::phylo_gmm::ground_truth;
using phylocore::tests::phylo_gmm::simulate_multi_tree;
using phylocore::tests::phylo_gmm::simulated_dataset;

namespace {

ground_truth make_ground_truth() {
    ground_truth gt;

    // pi: root state distribution (k = 3).
    gt.pi = Eigen::VectorXd(3);
    gt.pi << 0.3, 0.4, 0.3;

    // Q: general (non-reversible) 3x3 rate matrix. Off-diagonals biased upward
    // (i -> i+1 is faster than the reverse) so the chain is NOT reversible.
    // Diagonal = -row-sum.
    //
    //          0      1      2
    //  0:   -0.8    0.5    0.3
    //  1:    0.2   -0.9    0.7
    //  2:    0.4    0.1   -0.5
    gt.Q = Eigen::MatrixXd(3, 3);
    gt.Q << -0.8, 0.5, 0.3,
             0.2, -0.9, 0.7,
             0.4, 0.1, -0.5;

    // mu_i: well-separated cluster centers in R^2.
    Eigen::VectorXd m0(2); m0 << 0.0,  0.0;
    Eigen::VectorXd m1(2); m1 << 3.0,  3.0;
    Eigen::VectorXd m2(2); m2 << -3.0, 3.0;
    gt.mu = {m0, m1, m2};

    // Sigma_i: anisotropic full 2x2 SPD covariances.
    Eigen::MatrixXd S0(2, 2); S0 << 0.5, 0.0,
                                     0.0, 0.5;     // isotropic, tight
    Eigen::MatrixXd S1(2, 2); S1 << 1.0, 0.3,
                                     0.3, 0.5;     // positive correlation
    Eigen::MatrixXd S2(2, 2); S2 << 0.8, -0.2,
                                    -0.2, 0.6;     // negative correlation
    gt.Sigma = {S0, S1, S2};

    return gt;
}

void print_ground_truth(const ground_truth& gt) {
    std::cout << "── Ground truth ─────────────────────────────────────────\n";
    std::cout << "pi  = " << gt.pi.transpose() << "\n";
    std::cout << "Q   =\n" << gt.Q << "\n";
    for (size_t i = 0; i < gt.mu.size(); ++i) {
        std::cout << "mu_" << i << "    = " << gt.mu[i].transpose() << "\n";
        std::cout << "Sigma_" << i << " =\n" << gt.Sigma[i] << "\n";
    }
}

void print_dataset_summary(const simulated_dataset& ds, const ground_truth& gt) {
    std::cout << "\n── Dataset summary ──────────────────────────────────────\n";
    std::cout << "num_trees      = " << ds.trees.size() << "\n";
    std::cout << "total_taxa     = " << ds.observed_data.size() << "\n";

    for (size_t ti = 0; ti < ds.trees.size(); ++ti) {
        const auto& t = ds.trees[ti];
        std::cout << "  tree " << ti
                  << ": num_leaves=" << t.num_leaves
                  << ", num_nodes=" << t.num_nodes
                  << ", root_id="   << t.root_id  << "\n";
    }

    // Confirm taxa-id disjointness: leaves' taxa_ids should be exactly
    // [0, total_taxa) with no duplicates.
    std::vector<int> seen(ds.observed_data.size(), 0);
    bool ok = true;
    for (const auto& t : ds.trees) {
        for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
            if (t.tree.out_degree(static_cast<int>(node_id)) != 0) continue;
            size_t taxa_id = t.tree[static_cast<int>(node_id)].data;
            if (taxa_id >= seen.size()) { ok = false; break; }
            seen[taxa_id]++;
        }
    }
    int dups = 0, miss = 0;
    for (int v : seen) { if (v > 1) ++dups; if (v == 0) ++miss; }
    std::cout << "disjoint_taxa  = "
              << (ok && dups == 0 && miss == 0 ? "yes" : "no")
              << " (dups=" << dups << ", missing=" << miss << ")\n";

    // Sample y_l per latent cluster (a few entries from each).
    std::cout << "\nSample observations grouped by latent state:\n";
    const size_t k = gt.mu.size();
    std::vector<int> per_state_count(k, 0);
    const int per_state_target = 3;

    for (size_t ti = 0; ti < ds.trees.size(); ++ti) {
        const auto& t = ds.trees[ti];
        const auto& X = ds.latent_per_tree[ti];
        for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
            if (t.tree.out_degree(static_cast<int>(node_id)) != 0) continue;
            size_t s = X[node_id];
            if (per_state_count[s] >= per_state_target) continue;
            size_t taxa_id = t.tree[static_cast<int>(node_id)].data;
            std::cout << "  state=" << s
                      << " taxa_id=" << taxa_id
                      << " y=(" << ds.observed_data[taxa_id].transpose() << ")"
                      << " mu_" << s << "=(" << gt.mu[s].transpose() << ")\n";
            per_state_count[s]++;
        }
    }

    // Empirical per-state means to sanity-check separability.
    std::cout << "\nEmpirical per-state means (should track mu_i):\n";
    std::vector<Eigen::VectorXd> empirical_sum(
        k, Eigen::VectorXd::Zero(gt.mu[0].size()));
    std::vector<int> counts(k, 0);
    for (size_t ti = 0; ti < ds.trees.size(); ++ti) {
        const auto& t = ds.trees[ti];
        const auto& X = ds.latent_per_tree[ti];
        for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
            if (t.tree.out_degree(static_cast<int>(node_id)) != 0) continue;
            size_t taxa_id = t.tree[static_cast<int>(node_id)].data;
            size_t s = X[node_id];
            empirical_sum[s] += ds.observed_data[taxa_id];
            counts[s]++;
        }
    }
    for (size_t s = 0; s < k; ++s) {
        if (counts[s] == 0) {
            std::cout << "  state " << s << ": n=0 (no leaves)\n";
            continue;
        }
        Eigen::VectorXd em = empirical_sum[s] / static_cast<double>(counts[s]);
        std::cout << "  state " << s << ": n=" << counts[s]
                  << " empirical_mean=(" << em.transpose() << ")"
                  << " mu_i=(" << gt.mu[s].transpose() << ")\n";
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    unsigned int seed = 42;
    if (argc > 1) seed = static_cast<unsigned int>(std::atoi(argv[1]));

    std::mt19937 rng(seed);

    ground_truth gt = make_ground_truth();
    print_ground_truth(gt);

    // Multi-tree: 3 trees of 40 leaves each.
    std::vector<size_t> leaves_per_tree{40, 40, 40};
    simulated_dataset ds = simulate_multi_tree(
        gt, leaves_per_tree, rng, /*mean_branch_length=*/0.5);

    print_dataset_summary(ds, gt);

    std::cout << "\nseed=" << seed << "\n";
    return 0;
}
