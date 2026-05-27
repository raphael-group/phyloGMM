// Unit test for the new pmm_ctmc_model + pmm_categorical_observation_model.
//
// Builds a tiny synthetic fixture (4-leaf tree, 1 character), runs
// phylocore::compute_inside_log_likelihood through the new model classes,
// and compares against an independent reference computed in this file from
// the closed-form PMM transition kernel (no log-space, no log-sum-exp —
// straight raw-probability sum-product over the small tree).
//
// PASS criterion: |new - reference| < 1e-9.
//
// This mirrors the validation strategy described in the task spec; it is
// shipped as a unit test (rather than a stand-alone driver) so it picks up
// `ctest -R ^test_` automatically.

#include <cmath>
#include <cstdio>
#include <vector>

#include "meta_data.h"
#include "observation_models/pmm_categorical.h"
#include "phylogeny.h"
#include "state_transition_models/laml_data.h"
#include "state_transition_models/pmm_ctmc.h"

using namespace phylocore;

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Tiny synthetic fixture
// ─────────────────────────────────────────────────────────────────────────────
//
// Tree:  ((A:0.5,B:0.3)I:0.4,(C:0.2,D:0.7)J:0.6)R[:1.0];
// 4 leaves (A=0, B=1, C=2, D=3), 2 internals (I=4, J=5), root R=6.
// Branch leading into root has length 1.0 (meaningful for LAML).
//
// 1 character; alphabet_size = 4:
//   index 0 → silenced
//   index 1 → unedited       (root prior 1.0)
//   index 2 → mutated cat 0  (prior 0.6)
//   index 3 → mutated cat 1  (prior 0.4)
//
// Observations (external convention):
//   A → 0   (observed unedited)
//   B → -1  (missing)
//   C → 1   (mutated cat 0 → alphabet index 2)
//   D → 2   (mutated cat 1 → alphabet index 3)
//
// Parameters: nu = 0.1, phi = 0.05.

struct tiny_fixture {
    tree                                t;
    std::vector<std::vector<int>>       obs;
    std::vector<size_t>                 alphabet_sizes{4};
    std::vector<std::vector<double>>    mutation_priors{{0.6, 0.4}};
    double nu  = 0.1;
    double phi = 0.05;
};

tiny_fixture build_fixture() {
    tiny_fixture f;
    digraph<size_t> g;
    for (int i = 0; i < 7; ++i) g.add_vertex(static_cast<size_t>(i));
    g.add_edge(4, 0); // I -> A
    g.add_edge(4, 1); // I -> B
    g.add_edge(5, 2); // J -> C
    g.add_edge(5, 3); // J -> D
    g.add_edge(6, 4); // R -> I
    g.add_edge(6, 5); // R -> J

    f.t.num_leaves = 4;
    f.t.num_nodes  = 7;
    f.t.root_id    = 6;
    f.t.tree       = std::move(g);
    f.t.branch_lengths = {0.5, 0.3, 0.2, 0.7, 0.4, 0.6, 1.0};
    f.t.node_names = {"A", "B", "C", "D", "I", "J", "R"};

    f.obs = {{0}, {-1}, {1}, {2}};
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Independent reference: closed-form PMM transition kernel and emission,
// sum-product over the small tree in raw probability space.
// ─────────────────────────────────────────────────────────────────────────────

double pmm_trans(double b, double nu, size_t alpha, size_t beta,
                 const std::vector<double>& q, size_t k) {
    if (alpha >= k || beta >= k) return 0.0;
    if (alpha == 0) return (beta == 0) ? 1.0 : 0.0;       // silenced absorbs
    if (alpha == 1) {
        if (beta == 0) return 1.0 - std::exp(-nu * b);
        if (beta == 1) return std::exp(-(1.0 + nu) * b);
        return q[beta - 2] * (1.0 - std::exp(-b)) * std::exp(-nu * b);
    }
    // alpha >= 2 (mutated)
    if (beta == 0)     return 1.0 - std::exp(-nu * b);
    if (beta == alpha) return std::exp(-nu * b);
    return 0.0;
}

double pmm_emit(int y, size_t alpha, double phi, size_t k) {
    if (alpha >= k) return 0.0;
    if (y == -1) return (alpha == 0) ? 1.0 : phi;
    const size_t target = static_cast<size_t>(y) + 1u;
    return (alpha == target) ? (1.0 - phi) : 0.0;
}

double reference_log_likelihood(const tiny_fixture& f) {
    const size_t k    = f.alphabet_sizes[0];
    const auto&  q    = f.mutation_priors[0];
    const double nu   = f.nu;
    const double phi  = f.phi;
    const size_t N    = f.t.num_nodes;

    std::vector<std::vector<double>> inside(N, std::vector<double>(k, 0.0));

    for (int node_id : f.t.tree.postorder_traversal(f.t.root_id)) {
        if (f.t.tree.out_degree(node_id) == 0) {
            size_t taxa_id = f.t.tree[node_id].data;
            int y = f.obs[taxa_id][0];
            for (size_t a = 0; a < k; ++a)
                inside[node_id][a] = pmm_emit(y, a, phi, k);
            continue;
        }
        for (size_t a = 0; a < k; ++a) inside[node_id][a] = 1.0;
        for (int u : f.t.tree.successors(node_id)) {
            double b = f.t.branch_lengths[u];
            for (size_t a = 0; a < k; ++a) {
                double s = 0.0;
                for (size_t bt = 0; bt < k; ++bt)
                    s += pmm_trans(b, nu, a, bt, q, k) * inside[u][bt];
                inside[node_id][a] *= s;
            }
        }
    }

    // Root branch (meaningful for LAML) + root prior δ(unedited).
    const double b_root = f.t.branch_lengths[f.t.root_id];
    double total = 0.0;
    for (size_t a = 0; a < k; ++a) {
        double prior = (a == 1) ? 1.0 : 0.0;
        double s = 0.0;
        for (size_t bt = 0; bt < k; ++bt)
            s += pmm_trans(b_root, nu, a, bt, q, k) * inside[f.t.root_id][bt];
        total += prior * s;
    }
    return std::log(total);
}

// ─────────────────────────────────────────────────────────────────────────────
// New model
// ─────────────────────────────────────────────────────────────────────────────

double new_log_likelihood(const tiny_fixture& f) {
    pmm_ctmc_model<laml_data> stm(
        f.alphabet_sizes, f.mutation_priors, f.nu, f.phi, /*timescale=*/1.0);
    pmm_categorical_observation_model<laml_data, std::vector<int>> obs(f.alphabet_sizes);

    size_t max_alpha = 0;
    for (size_t k : f.alphabet_sizes) max_alpha = std::max(max_alpha, k);
    std::vector<double> scratch(max_alpha, 0.0);
    auto node_data = stm.initialize_data(f.t, &scratch);

    meta_data md{1, max_alpha, f.alphabet_sizes, {"A","B","C","D"}};
    likelihood_buffer buf(1, max_alpha, f.t.num_nodes);

    auto per_site = compute_inside_log_likelihood(
        stm, obs, f.obs, md, f.t, buf, node_data);

    double total = 0.0;
    for (double v : per_site) total += v;
    return total;
}

// ─────────────────────────────────────────────────────────────────────────────
// Also exercise: missing observation alone, unedited everywhere, fully-mutated
// — three additional sanity points covering each emission branch.
// ─────────────────────────────────────────────────────────────────────────────

void run_one(const char* label, tiny_fixture& f, double tol) {
    const double ref = reference_log_likelihood(f);
    const double got = new_log_likelihood(f);
    const double d   = std::abs(got - ref);
    std::printf("  %-32s  ref=% .12g  new=% .12g  |delta|=%.3g\n",
                label, ref, got, d);
    if (!(d < tol)) {
        std::fprintf(stderr,
            "FAIL[%s]: reference=%.16g, new=%.16g, |delta|=%.3g, tol=%.3g\n",
            label, ref, got, d, tol);
        std::abort();
    }
}

} // namespace

int main() {
    std::printf("test_pmm_ctmc_inside_likelihood: 4-leaf tree, alphabet=4\n");

    // 1. Mixed observation (default fixture)
    {
        auto f = build_fixture();
        run_one("mixed (0, -1, 1, 2)", f, 1e-9);
    }

    // 2. All leaves observe unedited
    {
        auto f = build_fixture();
        f.obs = {{0}, {0}, {0}, {0}};
        run_one("all unedited", f, 1e-9);
    }

    // 3. All leaves observe missing
    {
        auto f = build_fixture();
        f.obs = {{-1}, {-1}, {-1}, {-1}};
        run_one("all missing", f, 1e-9);
    }

    // 4. Two mutated, two missing
    {
        auto f = build_fixture();
        f.obs = {{1}, {-1}, {2}, {-1}};
        run_one("mut + missing", f, 1e-9);
    }

    // 5. Different parameter regime: larger nu, larger phi
    {
        auto f = build_fixture();
        f.nu  = 0.5;
        f.phi = 0.3;
        run_one("nu=0.5 phi=0.3", f, 1e-9);
    }

    std::printf("test_pmm_ctmc_inside_likelihood: PASS\n");
    return 0;
}
