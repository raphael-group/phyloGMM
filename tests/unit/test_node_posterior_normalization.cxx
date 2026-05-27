// Unit test: node_posterior is a proper probability distribution.
//
// Σ_i exp(node_posterior(tree_idx, node_id, site_c)[i]) == 1
//
// This must hold for every node in the tree, leaves AND the root AND internal
// nodes (in this minimal fixture the root is the only internal node). It is
// the bread-and-butter normalization invariant: posterior over a single node's
// latent state given all observed data is, by construction, a probability mass
// function summing to 1.
//
// Catches:
//   - log_p_y normalization bugs (the regression that motivated this harness)
//   - inside/outside buffer indexing bugs (off-by-one across sites/nodes)
//   - root_prior dimension mistakes that bleed through to outside pass
//
// We exercise the same 2-site fixture as test_log_p_y_accessor so that
// per-site marginalization is non-trivial (the two sites have distinct
// marginals).

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <variant>
#include <vector>

#include "jc_fixture.h"
#include "phylogeny.h"

namespace {

// Build the 2-site fixture (duplicated from test_log_p_y_accessor — keeping
// each test file self-contained avoids cross-file fixture coupling).
phylocore::tests::jc_fixture make_two_site_fixture(
    int s0_site0, int s1_site0,
    int s0_site1, int s1_site1,
    double branch)
{
    using namespace phylocore;
    using namespace phylocore::tests;

    jc_fixture f;
    f.t = std::make_unique<tree>();
    f.t->num_leaves = 2;
    f.t->num_nodes  = 3;
    int leaf0 = f.t->tree.add_vertex(0);
    int leaf1 = f.t->tree.add_vertex(1);
    int root  = f.t->tree.add_vertex(0);
    f.t->root_id = static_cast<std::size_t>(root);
    f.t->tree.add_edge(root, leaf0);
    f.t->tree.add_edge(root, leaf1);
    f.t->branch_lengths = { branch, branch, 0.0 };
    f.t->node_names     = { "L0", "L1", "ROOT" };

    f.stm = std::make_unique<reversible_ctmc_model>();
    f.stm->rates.assign(2, std::vector<double>(6, 1.0));
    f.stm->freqs.assign(2, std::vector<double>(4, 0.25));
    f.stm->build();

    f.obs = std::make_unique<identity_observation_model<>>();
    f.obs->sites_latent_states = f.stm->sites_latent_states;

    f.observed_data = std::make_unique<std::vector<std::vector<int>>>();
    f.observed_data->push_back({s0_site0, s0_site1});
    f.observed_data->push_back({s1_site0, s1_site1});

    f.metadata = std::make_unique<meta_data>();
    f.metadata->num_sites         = 2;
    f.metadata->max_alphabet_size = 4;
    f.metadata->alphabet_sizes    = {4, 4};
    f.metadata->taxa_names        = {"L0", "L1"};

    f.node_data = std::make_unique<std::vector<std::monostate>>(3);

    f.problem = std::make_unique<jc_problem>();
    f.problem->stm           = f.stm.get();
    f.problem->obs           = f.obs.get();
    f.problem->observed_data = f.observed_data.get();
    f.problem->metadata      = f.metadata.get();
    f.problem->trees         = { f.t.get() };
    f.problem->node_data     = { f.node_data.get() };
    return f;
}

void assert_posterior_normalized(
    const std::vector<double>& log_gamma,
    const std::string& what)
{
    double sum = 0.0;
    for (double lg : log_gamma) sum += std::exp(lg);
    phylocore::tests::expect_near(sum, 1.0, 1e-10, what);
}

} // namespace

int main() {
    using namespace phylocore;
    using namespace phylocore::tests;

    // Mismatched + matched sites so the posterior at the root is non-trivial.
    auto f = make_two_site_fixture(/*site0:*/ 0, 1, /*site1:*/ 2, 2, /*branch=*/0.3);

    // ── Required: run all three traversals so posterior accessors work ──────
    f.problem->log_likelihood();
    for (std::size_t i = 0; i < f.problem->trees.size(); ++i) {
        compute_edge_inside_log_likelihood(
            *f.problem->stm, *f.problem->trees[i],
            f.problem->inside_buffer(i),
            f.problem->edge_inside_buffer(i),
            *f.problem->node_data[i]);
        compute_outside_log_likelihood(
            *f.problem->stm, *f.problem->trees[i],
            f.problem->edge_inside_buffer(i),
            f.problem->outside_buffer(i),
            *f.problem->node_data[i]);
    }

    // ── Check normalization at every (node, site) pair ──────────────────────
    const std::size_t num_sites = f.stm->sites_latent_states.size();
    for (std::size_t node_id = 0; node_id < f.t->num_nodes; ++node_id) {
        for (std::size_t c = 0; c < num_sites; ++c) {
            auto log_gamma = f.problem->node_posterior(0, node_id, c);
            assert_posterior_normalized(
                log_gamma,
                "node_posterior(0, node=" + std::to_string(node_id) +
                ", site=" + std::to_string(c) + ") sums to 1");
        }
    }

    // ── Specifically test the root, which is the internal node here. ────────
    for (std::size_t c = 0; c < num_sites; ++c) {
        auto root_g = f.problem->root_posterior(0, c);
        assert_posterior_normalized(
            root_g, "root_posterior(0, " + std::to_string(c) + ") sums to 1");
    }

    // ── Leaves where the observation is fully informative: posterior should
    //     be a delta on the observed state. Tighter test on top of the
    //     normalization check.
    {
        auto log_gamma_leaf0_site0 = f.problem->node_posterior(0, /*leaf 0*/0, /*site 0*/0);
        const int observed_state = (*f.problem->observed_data)[0][0];  // = 0 (A)
        expect_near(std::exp(log_gamma_leaf0_site0[observed_state]), 1.0, 1e-10,
                    "leaf posterior concentrates on observed state");
        for (std::size_t i = 0; i < log_gamma_leaf0_site0.size(); ++i) {
            if (static_cast<int>(i) == observed_state) continue;
            expect_near(std::exp(log_gamma_leaf0_site0[i]), 0.0, 1e-10,
                        "leaf posterior is zero on un-observed state");
        }
    }

    std::printf("test_node_posterior_normalization: PASS\n");
    return 0;
}
