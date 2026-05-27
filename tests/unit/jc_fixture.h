#ifndef PHYLOCORE_TESTS_UNIT_JC_FIXTURE_H
#define PHYLOCORE_TESTS_UNIT_JC_FIXTURE_H

// Shared fixture builder for unit tests: a minimal 2-leaf JC nucleotide tree
// with a single site, plus a thin gtr_problem subclass that satisfies the
// joint_optimization_problem<...> required interface (get_params /
// set_params). All tests in tests/unit/ that need a runnable problem instance
// build on this fixture so the per-test cxx files stay focused on the
// invariant they're checking.
//
// Topology:
//
//        root
//       /    \
//   leaf0    leaf1   (branch length t on each)
//
// Built as a true rooted bifurcation; the internal root node has two outgoing
// edges to leaves. branch_lengths[root] is set to 0 (matches the convention
// used by run_gtr_em / phylogeny.h: the root's "incoming" branch is unused
// during traversal but the value is still indexed, so we keep it well-defined).

#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "meta_data.h"
#include "observation_models/identity.h"
#include "optimize/joint_optimization_problem.h"
#include "phylogeny.h"
#include "state_transition_models/reversible_ctmc.h"

namespace phylocore {
namespace tests {

/*!
 * @brief Concrete joint_optimization_problem for a reversible CTMC over a
 *        discrete (vector<int>) observation type. Provides the minimal
 *        get_params / set_params overrides so the problem is instantiable.
 *
 * Layout (matches gtr_em_problem in src/run_gtr_em.cxx):
 *   per site: [r_AC, r_AG, r_AT, r_CG, r_CT, r_GT, fA, fC, fG]
 *   freq_T = 1 - (fA + fC + fG)
 */
class jc_problem
    : public phylocore::optimize::joint_optimization_problem<
          phylocore::reversible_ctmc_model,
          phylocore::identity_observation_model<>,
          std::vector<int>>
{
public:
    std::vector<double> get_params() const override {
        const std::size_t num_sites = stm->rates.size();
        std::vector<double> p;
        p.reserve(num_sites * 9);
        for (std::size_t c = 0; c < num_sites; ++c) {
            const auto& r = stm->rates[c];
            const auto& f = stm->freqs[c];
            for (std::size_t i = 0; i < 6; ++i) p.push_back(r[i]);
            p.push_back(f[0]); p.push_back(f[1]); p.push_back(f[2]);
        }
        return p;
    }

    void set_params(const std::vector<double>& params) override {
        const std::size_t num_sites = stm->rates.size();
        if (params.size() != num_sites * 9) {
            throw phylocore::optimize::invalid_parameters_error(
                "jc_problem::set_params: expected " +
                std::to_string(num_sites * 9) + " params, got " +
                std::to_string(params.size()));
        }
        for (std::size_t c = 0; c < num_sites; ++c) {
            const std::size_t base = c * 9;
            auto& r = stm->rates[c];
            auto& f = stm->freqs[c];
            for (std::size_t i = 0; i < 6; ++i) r[i] = params[base + i];
            const double fA = params[base + 6];
            const double fC = params[base + 7];
            const double fG = params[base + 8];
            const double fT = 1.0 - (fA + fC + fG);
            if (fT <= 0.0 || fA <= 0.0 || fC <= 0.0 || fG <= 0.0) {
                throw phylocore::optimize::invalid_parameters_error(
                    "jc_problem::set_params: non-positive freq at site " +
                    std::to_string(c), base + 6);
            }
            f[0] = fA; f[1] = fC; f[2] = fG; f[3] = fT;
        }
        stm->build();
        obs->sites_latent_states = stm->sites_latent_states;
    }
};

/*!
 * @brief Owning bundle of the test fixture: a 2-leaf JC tree, JC model, etc.
 *
 * Stored in unique_ptrs so the pointers held by jc_problem (stm, obs, trees,
 * node_data, observed_data, metadata) remain stable through moves of the
 * bundle. Use jc_fixture::make(...) to construct.
 */
struct jc_fixture {
    std::unique_ptr<tree>                                    t;
    std::unique_ptr<reversible_ctmc_model>                   stm;
    std::unique_ptr<identity_observation_model<>>            obs;
    std::unique_ptr<std::vector<std::vector<int>>>           observed_data;
    std::unique_ptr<meta_data>                               metadata;
    std::unique_ptr<std::vector<std::monostate>>             node_data;
    std::unique_ptr<jc_problem>                              problem;

    /*!
     * @brief Build a fresh fixture.
     *
     * @param obs0   nucleotide state for leaf 0 at site 0 (0=A,1=C,2=G,3=T,-1=missing)
     * @param obs1   nucleotide state for leaf 1 at site 0
     * @param branch branch length on both edges (default 0.1)
     */
    static jc_fixture make(int obs0, int obs1, double branch = 0.1) {
        jc_fixture f;

        // ── Tree: leaf0 (id 0), leaf1 (id 1), root (id 2) ───────────────────
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

        // ── JC model: 1 site, rates = 1, freqs = 0.25 ───────────────────────
        f.stm = std::make_unique<reversible_ctmc_model>();
        f.stm->rates.assign(1, std::vector<double>(6, 1.0));
        f.stm->freqs.assign(1, std::vector<double>(4, 0.25));
        f.stm->build();

        f.obs = std::make_unique<identity_observation_model<>>();
        f.obs->sites_latent_states = f.stm->sites_latent_states;

        // ── Observed data: leaf 0 gets obs0, leaf 1 gets obs1 ───────────────
        // taxa_id matches the leaf vertex's data field. Leaves were added with
        // data 0 and 1 above.
        f.observed_data = std::make_unique<std::vector<std::vector<int>>>();
        f.observed_data->push_back({obs0});
        f.observed_data->push_back({obs1});

        f.metadata = std::make_unique<meta_data>();
        f.metadata->num_sites         = 1;
        f.metadata->max_alphabet_size = 4;
        f.metadata->alphabet_sizes    = {4};
        f.metadata->taxa_names        = {"L0", "L1"};

        f.node_data = std::make_unique<std::vector<std::monostate>>(3);

        // ── Wire up the problem ─────────────────────────────────────────────
        f.problem = std::make_unique<jc_problem>();
        f.problem->stm           = f.stm.get();
        f.problem->obs           = f.obs.get();
        f.problem->observed_data = f.observed_data.get();
        f.problem->metadata      = f.metadata.get();
        f.problem->trees         = { f.t.get() };
        f.problem->node_data     = { f.node_data.get() };

        return f;
    }
};

// ── Pretty assert helpers ────────────────────────────────────────────────────

inline void expect_near(double actual, double expected, double tol,
                        const std::string& what)
{
    const double diff = std::abs(actual - expected);
    if (!(diff <= tol)) {
        std::fprintf(stderr,
            "FAIL: %s: expected %.12g, got %.12g (|diff|=%.3g, tol=%.3g)\n",
            what.c_str(), expected, actual, diff, tol);
        std::abort();
    }
}

inline void expect_true(bool cond, const std::string& what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        std::abort();
    }
}

} // namespace tests
} // namespace phylocore

#endif // PHYLOCORE_TESTS_UNIT_JC_FIXTURE_H
