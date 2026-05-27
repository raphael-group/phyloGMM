// Unit test: joint_optimization_problem::log_p_y(tree_idx, site_c) returns
// the PER-SITE marginal log p(y_c | tree), NOT the per-tree sum across sites.
//
// This is the regression test for the bug that crashed run_gtr_em: prior to
// the fix, log_p_y was implemented as a per-tree scalar (the total), and any
// downstream consumer that fed it back into a posterior accessor would
// silently compute the wrong normalizer for multi-site problems.
//
// Strategy: build a 2-site fixture where the two sites have DIFFERENT
// observation patterns and therefore distinct per-site marginals. Verify
// that:
//   1. log_p_y(0, 0) != log_p_y(0, 1)                  (sites differ)
//   2. log_p_y(0, c) matches the closed-form JC analytic answer for site c
//   3. log_total_likelihood(0) == log_p_y(0, 0) + log_p_y(0, 1)
//      (the per-site values sum to the per-tree total — which is what
//      log_total_likelihood returns).
//
// If log_p_y were still the (buggy) per-tree sum, both checks (1) and (2)
// would fail, and check (3) would degenerate to total = 2·total.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <variant>
#include <vector>

#include "jc_fixture.h"

namespace {

double jc_analytic_log_likelihood(int s0, int s1, double t) {
    const double e    = std::exp(-4.0 * t / 3.0);
    const double diag = 0.25 + 0.75 * e;
    const double off  = 0.25 - 0.25 * e;
    double L = 0.0;
    for (int i = 0; i < 4; ++i) {
        const double p_i_s0 = (i == s0) ? diag : off;
        const double p_i_s1 = (i == s1) ? diag : off;
        L += 0.25 * p_i_s0 * p_i_s1;
    }
    return std::log(L);
}

// Build a 2-site JC fixture on the same 2-leaf topology as jc_fixture::make.
// We can't reuse jc_fixture::make directly because it hard-codes a single
// site; here we expand to two sites with distinct per-leaf observations.
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

} // namespace

int main() {
    using namespace phylocore::tests;

    const double t   = 0.1;
    const double TOL = 1e-10;

    // Site 0: both leaves A,A (matching).  Site 1: A,C (mismatched).
    auto f = make_two_site_fixture(/*site0:*/ 0, 0, /*site1:*/ 0, 1, t);

    const double total = f.problem->log_likelihood();

    const double L0_actual   = f.problem->log_p_y(0, 0);
    const double L1_actual   = f.problem->log_p_y(0, 1);
    const double L0_expected = jc_analytic_log_likelihood(0, 0, t);
    const double L1_expected = jc_analytic_log_likelihood(0, 1, t);

    // (1) Sites have distinct per-site marginals. (Strong signal that
    // log_p_y indexes per-site, not per-tree.)
    expect_true(L0_actual != L1_actual,
                "log_p_y(0,0) and log_p_y(0,1) should differ across sites");

    // (2) Each matches its analytic answer.
    expect_near(L0_actual, L0_expected, TOL,
                "log_p_y(0, 0) matches JC analytic for (A,A)");
    expect_near(L1_actual, L1_expected, TOL,
                "log_p_y(0, 1) matches JC analytic for (A,C)");

    // (3) Per-tree total equals sum of per-site marginals.
    const double total_from_per_site = L0_actual + L1_actual;
    expect_near(total, total_from_per_site, TOL,
                "log_likelihood() == Σ_c log_p_y(0, c)");
    expect_near(f.problem->log_total_likelihood(0), total_from_per_site, TOL,
                "log_total_likelihood(0) == Σ_c log_p_y(0, c)");

    std::printf("test_log_p_y_accessor: PASS\n");
    return 0;
}
