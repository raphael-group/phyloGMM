// test_multi_tree_aggregation: CTest unit test validating that
// phylo_gmm_problem::compute_suff_stats() correctly AGGREGATES per-tree
// contributions when the problem holds multiple trees.
//
// Context
// -------
// We have run into bugs in the past where multi-tree aggregation either
// (a) silently dropped trees past index 0, (b) summed with the wrong loop
// bound, or (c) had an off-by-one in the per-tree posterior accessors. The
// caller's caveat (which we agree with) is that splitting one tree into K
// sub-trees does NOT preserve the model: each sub-tree's root gets the prior
// π applied independently, so the MLE of a single tree of N leaves is NOT
// the same as that of K sub-trees of N/K leaves. Therefore this test does NOT
// compare "one big tree" vs "K small trees" — it compares the AGGREGATION
// step itself:
//
//   Way 1: build phylo_gmm_problem with K trees; run inside / edge_inside /
//          outside passes; call compute_suff_stats once. Capture suff_stats.
//
//   Way 2: for each tree separately, build a phylo_gmm_problem with that ONE
//          tree (same model parameters, same observed_data); run the three
//          passes; call compute_suff_stats. Sum the per-tree results.
//
// Both ways should agree to within 1e-10, because the per-tree contributions
// to compute_suff_stats() are independent functions of (1) the per-tree
// inside/outside posteriors and (2) the model parameters — and both of these
// are identical between Way 1 (multi-tree) and Way 2 (per-tree, then summed)
// since the model parameters are shared and the inside/outside passes are
// tree-local.
//
// What this catches: multi-tree loop bound bugs, indexing bugs, missing
// accumulation, Hobolth-Jensen's per-tree iteration disagreeing with the
// for-loop in compute_suff_stats, etc.
//
// Test layout
// -----------
//   1. Tiny fixture: 2 trees, each with 2 leaves (=> 3 nodes each). k=2,
//      d=1. Ground truth Q, μ, Σ, π chosen so suff_stats are hand-checkable
//      via the per-tree decomposition.
//   2. Sanity-check fixture: 2 trees of sizes (20, 60) — confirms responsibility
//      totals sum to 80 (not 40+40), proving the aggregation respects per-tree
//      sizes rather than splitting evenly.
//
// TODO: dedup phylo_gmm_problem with src/run_phylo_gmm.cxx when task #37
// (move the class into a header) lands. The class below is a verbatim copy.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include "meta_data.h"
#include "observation_models/gmm.h"
#include "optimize/em.h"
#include "optimize/estimation_results.h"
#include "optimize/hobolth_jensen.h"
#include "optimize/joint_optimization_problem.h"
#include "phylogeny.h"
#include "state_transition_models/ctmc.h"

#include "simulator.h"

using namespace phylocore;
using phylocore::tests::phylo_gmm::ground_truth;
using phylocore::tests::phylo_gmm::simulate_multi_tree;
using phylocore::tests::phylo_gmm::simulated_dataset;

// ─────────────────────────────────────────────────────────────────────────────
// phylo_gmm_problem (DUPLICATED from src/run_phylo_gmm.cxx — see TODO above).
// Trimmed down: we only need set_params / compute_suff_stats. m_step,
// initialize_random, initialize_kmeans_pp are not required for this test.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

class phylo_gmm_problem
    : public phylocore::optimize::joint_optimization_problem<
          phylocore::ctmc_model,
          phylocore::gmm_observation_model,
          Eigen::VectorXd,
          std::monostate> {
public:
    bool fix_Q          = false;
    bool fix_emission   = false;
    bool stationary_pi  = true;

    std::size_t k_() const { return static_cast<std::size_t>(stm->Q.rows()); }
    std::size_t d_() const { return obs->d; }

    void set_pi_on_root_prior_(const Eigen::VectorXd& pi) {
        const std::size_t k = k_();
        std::vector<double> v(k, 0.0);
        double s = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            v[i] = (pi[static_cast<Eigen::Index>(i)] > 1e-12) ?
                       pi[static_cast<Eigen::Index>(i)] : 1e-12;
            s += v[i];
        }
        for (std::size_t i = 0; i < k; ++i) v[i] /= s;
        stm->root_prior = { v };
    }

    Eigen::VectorXd extract_stationary_pi_() const {
        const Eigen::VectorXcd& lambda = stm->eigenvalues_q(0);
        const Eigen::MatrixXcd& Uinv   = stm->inverse_eigenvectors_q(0);
        const std::size_t k = k_();

        Eigen::Index best = 0;
        double best_mag   = std::abs(lambda[0]);
        for (Eigen::Index i = 1; i < lambda.size(); ++i) {
            const double m = std::abs(lambda[i]);
            if (m < best_mag) { best_mag = m; best = i; }
        }
        Eigen::VectorXd pi(k);
        for (std::size_t i = 0; i < k; ++i) {
            pi[static_cast<Eigen::Index>(i)] =
                Uinv(best, static_cast<Eigen::Index>(i)).real();
        }
        double s = 0.0;
        for (std::size_t i = 0; i < k; ++i)
            s += pi[static_cast<Eigen::Index>(i)];
        if (s < 0.0) pi = -pi;

        double total = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            const double v = pi[static_cast<Eigen::Index>(i)];
            pi[static_cast<Eigen::Index>(i)] = (v > 1e-12) ? v : 1e-12;
            total += pi[static_cast<Eigen::Index>(i)];
        }
        pi /= total;
        return pi;
    }

    std::vector<double> get_params() const override {
        const std::size_t k = k_();
        const std::size_t d = d_();
        std::vector<double> p;

        if (!fix_Q) {
            for (std::size_t i = 0; i < k; ++i)
                for (std::size_t j = 0; j < k; ++j)
                    if (i != j) p.push_back(stm->Q(i, j));
        }
        if (!fix_emission) {
            for (std::size_t i = 0; i < k; ++i)
                for (std::size_t a = 0; a < d; ++a)
                    p.push_back(obs->mu[i][a]);
            for (std::size_t i = 0; i < k; ++i)
                for (std::size_t r = 0; r < d; ++r)
                    for (std::size_t c = 0; c < d; ++c)
                        p.push_back(obs->Sigma[i](r, c));
        }
        if (!stationary_pi) {
            for (std::size_t i = 0; i + 1 < k; ++i) {
                p.push_back(stm->root_prior[0][i]);
            }
        }
        return p;
    }

    void set_params(const std::vector<double>& params) override {
        const std::size_t k = k_();
        const std::size_t d = d_();

        std::size_t expected = 0;
        if (!fix_Q)         expected += k * (k - 1);
        if (!fix_emission)  expected += k * d + k * d * d;
        if (!stationary_pi) expected += (k - 1);
        if (params.size() != expected) {
            throw phylocore::optimize::invalid_parameters_error(
                "phylo_gmm_problem::set_params: expected " +
                std::to_string(expected) + " params, got " +
                std::to_string(params.size()));
        }

        std::size_t idx = 0;

        if (!fix_Q) {
            Eigen::MatrixXd Qnew(k, k);
            for (std::size_t i = 0; i < k; ++i) {
                double row_off_sum = 0.0;
                for (std::size_t j = 0; j < k; ++j) {
                    if (i == j) continue;
                    const double v = params[idx];
                    if (v < -1e-10) {
                        throw phylocore::optimize::invalid_parameters_error(
                            "phylo_gmm_problem::set_params: Q(" +
                            std::to_string(i) + "," + std::to_string(j) +
                            ") = " + std::to_string(v) + " < 0",
                            idx);
                    }
                    Qnew(i, j) = std::max(0.0, v);
                    row_off_sum += Qnew(i, j);
                    ++idx;
                }
                Qnew(i, i) = -row_off_sum;
            }
            stm->Q = Qnew;
        }

        if (!fix_emission) {
            obs->d = d;
            obs->mu.assign(k, Eigen::VectorXd::Zero(d));
            obs->Sigma.assign(k, Eigen::MatrixXd::Zero(d, d));
            for (std::size_t i = 0; i < k; ++i) {
                for (std::size_t a = 0; a < d; ++a) {
                    obs->mu[i][a] = params[idx++];
                }
            }
            for (std::size_t i = 0; i < k; ++i) {
                for (std::size_t r = 0; r < d; ++r) {
                    for (std::size_t c = 0; c < d; ++c) {
                        obs->Sigma[i](r, c) = params[idx++];
                    }
                }
                obs->Sigma[i] = 0.5 * (obs->Sigma[i] + obs->Sigma[i].transpose());
                Eigen::LLT<Eigen::MatrixXd> llt(obs->Sigma[i]);
                if (llt.info() != Eigen::Success) {
                    throw phylocore::optimize::invalid_parameters_error(
                        "phylo_gmm_problem::set_params: Sigma[" +
                        std::to_string(i) + "] is not positive definite",
                        idx - 1);
                }
            }
        }

        if (!stationary_pi) {
            std::vector<double> pi(k, 0.0);
            double s = 0.0;
            for (std::size_t i = 0; i + 1 < k; ++i) {
                const double v = params[idx];
                if (v < -1e-12) {
                    throw phylocore::optimize::invalid_parameters_error(
                        "phylo_gmm_problem::set_params: pi[" +
                        std::to_string(i) + "] = " + std::to_string(v) + " < 0",
                        idx);
                }
                pi[i] = std::max(0.0, v);
                s += pi[i];
                ++idx;
            }
            const double last = 1.0 - s;
            if (last < -1e-10) {
                throw phylocore::optimize::invalid_parameters_error(
                    "phylo_gmm_problem::set_params: derived pi[k-1] = " +
                    std::to_string(last) + " < 0",
                    idx - 1);
            }
            pi[k - 1] = std::max(0.0, last);
            stm->root_prior = { pi };
        }

        if (!fix_Q) {
            stm->build();
            if (stationary_pi) {
                Eigen::VectorXd pi = extract_stationary_pi_();
                set_pi_on_root_prior_(pi);
            }
        } else {
            stm->build();
        }
        obs->sites_latent_states  = stm->sites_latent_states;
        obs->sites_observation_states.clear();
        obs->build();
    }

    struct suff_stats_type {
        Eigen::VectorXd               responsibility_total;
        std::vector<Eigen::VectorXd>  mu_numerator;
        std::vector<Eigen::MatrixXd>  Sigma_outer;
        Eigen::VectorXd               root_gamma_total;
        Eigen::MatrixXd               N_total;
        Eigen::VectorXd               T_total;
    };

    suff_stats_type compute_suff_stats() {
        const std::size_t k = k_();
        const std::size_t d = d_();
        const std::size_t num_sites = stm->sites_latent_states.size();

        suff_stats_type ss;
        ss.responsibility_total = Eigen::VectorXd::Zero(k);

        if (!fix_emission) {
            ss.mu_numerator.assign(k, Eigen::VectorXd::Zero(d));
            ss.Sigma_outer.assign(k, Eigen::MatrixXd::Zero(d, d));
        }
        if (!stationary_pi) {
            ss.root_gamma_total = Eigen::VectorXd::Zero(k);
        }

        for (std::size_t ti = 0; ti < trees.size(); ++ti) {
            const tree& t = *trees[ti];
            for (std::size_t v = 0; v < t.num_nodes; ++v) {
                if (t.tree.out_degree(static_cast<int>(v)) != 0) continue;
                const std::size_t taxa_id = t.tree[static_cast<int>(v)].data;
                const Eigen::VectorXd& y  = (*observed_data)[taxa_id];

                for (std::size_t c = 0; c < num_sites; ++c) {
                    auto log_gamma = leaf_posterior(ti, v, c);
                    for (std::size_t i = 0; i < k; ++i) {
                        const double r = std::exp(log_gamma[i]);
                        ss.responsibility_total[i] += r;
                        if (!fix_emission) {
                            ss.mu_numerator[i].noalias() += r * y;
                            ss.Sigma_outer[i].noalias()  += r * (y * y.transpose());
                        }
                    }
                }
            }

            if (!stationary_pi) {
                for (std::size_t c = 0; c < num_sites; ++c) {
                    auto log_gamma = root_posterior(ti, c);
                    for (std::size_t i = 0; i < k; ++i)
                        ss.root_gamma_total[i] += std::exp(log_gamma[i]);
                }
            }
        }

        if (!fix_Q) {
            auto hj = phylocore::optimize::hobolth_jensen_stats(*this);
            ss.N_total = hj.N_total;
            ss.T_total = hj.T_total;
        }

        return ss;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Wiring helpers.
// ─────────────────────────────────────────────────────────────────────────────

// Initialize the (stm, obs) pair from a ground_truth. Both Way 1 (multi-tree)
// and Way 2 (per-tree) need IDENTICAL model parameters for the test to be
// meaningful — so we centralise the setup here.
void install_ground_truth(ctmc_model& stm,
                          gmm_observation_model& obs,
                          const ground_truth& gt)
{
    const std::size_t k = static_cast<std::size_t>(gt.pi.size());
    const std::size_t d = static_cast<std::size_t>(gt.mu[0].size());

    stm.Q = gt.Q;
    std::vector<double> pi_vec(k);
    for (std::size_t i = 0; i < k; ++i)
        pi_vec[i] = gt.pi[static_cast<Eigen::Index>(i)];
    stm.root_prior = { pi_vec };
    stm.build();

    obs.d     = d;
    obs.mu    = gt.mu;
    obs.Sigma = gt.Sigma;
    obs.sites_latent_states = stm.sites_latent_states;
    obs.build();
}

// Run the three E-step passes (inside via log_likelihood(), then edge_inside,
// then outside) on every tree owned by `problem`. Required before
// compute_suff_stats() can be called.
void run_e_step_passes(phylo_gmm_problem& problem) {
    (void)problem.log_likelihood();
    for (std::size_t i = 0; i < problem.trees.size(); ++i) {
        phylocore::compute_edge_inside_log_likelihood(
            *problem.stm, *problem.trees[i],
            problem.inside_buffer(i), problem.edge_inside_buffer(i),
            *problem.node_data[i]);
        phylocore::compute_outside_log_likelihood(
            *problem.stm, *problem.trees[i],
            problem.edge_inside_buffer(i), problem.outside_buffer(i),
            *problem.node_data[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Comparison helpers.
// ─────────────────────────────────────────────────────────────────────────────

struct compare_result {
    bool        all_ok = true;
    double      max_abs_diff = 0.0;
    std::string first_failure;
};

void compare_field(compare_result& r,
                   const std::string& name,
                   double a, double b, double tol)
{
    const double diff = std::abs(a - b);
    if (diff > r.max_abs_diff) r.max_abs_diff = diff;
    if (diff > tol && r.all_ok) {
        r.all_ok = false;
        r.first_failure = name + ": Way1=" + std::to_string(a)
                        + " Way2=" + std::to_string(b)
                        + " diff=" + std::to_string(diff);
    }
}

void compare_vector(compare_result& r,
                    const std::string& name,
                    const Eigen::VectorXd& a, const Eigen::VectorXd& b,
                    double tol)
{
    if (a.size() != b.size()) {
        r.all_ok = false;
        if (r.first_failure.empty())
            r.first_failure = name + ": size mismatch " +
                std::to_string(a.size()) + " vs " + std::to_string(b.size());
        return;
    }
    for (Eigen::Index i = 0; i < a.size(); ++i) {
        compare_field(r, name + "[" + std::to_string(i) + "]",
                      a[i], b[i], tol);
    }
}

void compare_matrix(compare_result& r,
                    const std::string& name,
                    const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                    double tol)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        r.all_ok = false;
        if (r.first_failure.empty())
            r.first_failure = name + ": shape mismatch";
        return;
    }
    for (Eigen::Index i = 0; i < a.rows(); ++i)
        for (Eigen::Index j = 0; j < a.cols(); ++j)
            compare_field(r,
                          name + "(" + std::to_string(i) + "," +
                                std::to_string(j) + ")",
                          a(i, j), b(i, j), tol);
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute suff_stats with all of ds's trees in a single problem (Way 1).
// ─────────────────────────────────────────────────────────────────────────────

phylo_gmm_problem::suff_stats_type compute_way1(
    simulated_dataset& ds,
    const ground_truth& gt,
    ctmc_model& stm,
    gmm_observation_model& obs,
    std::vector<std::vector<std::monostate>>& node_data_storage)
{
    install_ground_truth(stm, obs, gt);

    node_data_storage.assign(ds.trees.size(), {});
    for (std::size_t i = 0; i < ds.trees.size(); ++i)
        node_data_storage[i].assign(ds.trees[i].num_nodes, std::monostate{});

    phylo_gmm_problem problem;
    problem.stm           = &stm;
    problem.obs           = &obs;
    problem.observed_data = &ds.observed_data;
    problem.metadata      = &ds.metadata;
    problem.trees.reserve(ds.trees.size());
    problem.node_data.reserve(ds.trees.size());
    for (std::size_t i = 0; i < ds.trees.size(); ++i) {
        problem.trees.push_back(&ds.trees[i]);
        problem.node_data.push_back(&node_data_storage[i]);
    }

    run_e_step_passes(problem);
    return problem.compute_suff_stats();
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute suff_stats by iterating one tree at a time and SUMMING per-tree
// contributions (Way 2). Each per-tree problem sees the same global
// observed_data + metadata (taxa_ids are disjoint, so this is safe).
// ─────────────────────────────────────────────────────────────────────────────

phylo_gmm_problem::suff_stats_type compute_way2(
    simulated_dataset& ds,
    const ground_truth& gt,
    std::vector<phylo_gmm_problem::suff_stats_type>* per_tree_out = nullptr)
{
    const std::size_t k = static_cast<std::size_t>(gt.pi.size());
    const std::size_t d = static_cast<std::size_t>(gt.mu[0].size());

    phylo_gmm_problem::suff_stats_type total;
    total.responsibility_total = Eigen::VectorXd::Zero(k);
    total.mu_numerator.assign(k, Eigen::VectorXd::Zero(d));
    total.Sigma_outer.assign(k, Eigen::MatrixXd::Zero(d, d));
    total.N_total = Eigen::MatrixXd::Zero(k, k);
    total.T_total = Eigen::VectorXd::Zero(k);

    if (per_tree_out) per_tree_out->clear();

    for (std::size_t ti = 0; ti < ds.trees.size(); ++ti) {
        // Fresh stm/obs/node_data per tree so each tree's E-step is isolated
        // (cleanest reading of "per-tree contribution").
        ctmc_model            stm;
        gmm_observation_model obs;
        install_ground_truth(stm, obs, gt);

        std::vector<std::monostate> nd(ds.trees[ti].num_nodes, std::monostate{});

        phylo_gmm_problem problem;
        problem.stm           = &stm;
        problem.obs           = &obs;
        problem.observed_data = &ds.observed_data;
        problem.metadata      = &ds.metadata;
        problem.trees.push_back(&ds.trees[ti]);
        problem.node_data.push_back(&nd);

        run_e_step_passes(problem);
        auto ss = problem.compute_suff_stats();

        // Sum contributions.
        total.responsibility_total += ss.responsibility_total;
        for (std::size_t i = 0; i < k; ++i) {
            total.mu_numerator[i] += ss.mu_numerator[i];
            total.Sigma_outer[i]  += ss.Sigma_outer[i];
        }
        total.N_total += ss.N_total;
        total.T_total += ss.T_total;

        if (per_tree_out) per_tree_out->push_back(std::move(ss));
    }

    return total;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture 1: 2 trees, each with 2 leaves. Hand-computable (k=2, d=1).
// ─────────────────────────────────────────────────────────────────────────────

ground_truth make_tiny_ground_truth() {
    ground_truth gt;

    // π = (0.6, 0.4)
    gt.pi = Eigen::VectorXd(2);
    gt.pi << 0.6, 0.4;

    // Q = [[-0.5, 0.5], [0.3, -0.3]] (asymmetric → general CTMC)
    gt.Q = Eigen::MatrixXd(2, 2);
    gt.Q << -0.5,  0.5,
             0.3, -0.3;

    // Well-separated means in 1D, small variance → posteriors will be
    // close to {0,1} for the leaves' true classes (hand-checkable behavior).
    Eigen::VectorXd m0(1); m0 << -2.0;
    Eigen::VectorXd m1(1); m1 <<  2.0;
    gt.mu = {m0, m1};

    Eigen::MatrixXd S0(1, 1); S0 << 0.25;
    Eigen::MatrixXd S1(1, 1); S1 << 0.25;
    gt.Sigma = {S0, S1};

    return gt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture 2: trees of sizes (20, 60). Tests that aggregation reflects per-tree
// sizes — total responsibility = 80, NOT split evenly into (40, 40).
// ─────────────────────────────────────────────────────────────────────────────

ground_truth make_sanity_ground_truth() {
    // Same shape as the tiny fixture (k=2, d=1) — keep things simple.
    return make_tiny_ground_truth();
}

// ─────────────────────────────────────────────────────────────────────────────

int run_fixture(const std::string& label,
                const ground_truth& gt,
                const std::vector<size_t>& leaves_per_tree,
                uint64_t sim_seed,
                std::vector<phylo_gmm_problem::suff_stats_type>* per_tree_out)
{
    std::cout << "\n── Fixture: " << label << " ────────────────────────\n";

    std::mt19937 sim_rng(sim_seed);
    simulated_dataset ds =
        simulate_multi_tree(gt, leaves_per_tree, sim_rng, /*mean_blen=*/0.5);

    std::cout << "  trees           = " << ds.trees.size() << "\n";
    std::cout << "  leaves per tree = [";
    for (std::size_t i = 0; i < leaves_per_tree.size(); ++i) {
        std::cout << leaves_per_tree[i]
                  << (i + 1 < leaves_per_tree.size() ? ", " : "");
    }
    std::cout << "]\n";
    std::cout << "  total leaves    = " << ds.observed_data.size() << "\n";

    ctmc_model            stm;
    gmm_observation_model obs;
    std::vector<std::vector<std::monostate>> node_data_storage;

    auto ss_way1 = compute_way1(ds, gt, stm, obs, node_data_storage);
    auto ss_way2 = compute_way2(ds, gt, per_tree_out);

    const double tol = 1e-10;
    compare_result r;
    compare_vector(r, "responsibility_total",
                   ss_way1.responsibility_total, ss_way2.responsibility_total, tol);
    for (std::size_t i = 0; i < ss_way1.mu_numerator.size(); ++i) {
        compare_vector(r, "mu_numerator[" + std::to_string(i) + "]",
                       ss_way1.mu_numerator[i], ss_way2.mu_numerator[i], tol);
        compare_matrix(r, "Sigma_outer[" + std::to_string(i) + "]",
                       ss_way1.Sigma_outer[i], ss_way2.Sigma_outer[i], tol);
    }
    compare_matrix(r, "N_total", ss_way1.N_total, ss_way2.N_total, tol);
    compare_vector(r, "T_total", ss_way1.T_total, ss_way2.T_total, tol);

    std::cout << "  Way 1 responsibility_total: " << ss_way1.responsibility_total.transpose() << "\n";
    std::cout << "  Way 2 responsibility_total: " << ss_way2.responsibility_total.transpose() << "\n";
    std::cout << "  Sum of responsibility (Way 1): "
              << ss_way1.responsibility_total.sum() << "\n";
    std::cout << "  Way 1 N_total: " << ss_way1.N_total.row(0)
              << " | " << ss_way1.N_total.row(1) << "\n";
    std::cout << "  Way 2 N_total: " << ss_way2.N_total.row(0)
              << " | " << ss_way2.N_total.row(1) << "\n";
    std::cout << "  Way 1 T_total: " << ss_way1.T_total.transpose() << "\n";
    std::cout << "  Way 2 T_total: " << ss_way2.T_total.transpose() << "\n";
    std::cout << "  max abs diff Way1 vs Way2 = " << r.max_abs_diff << "\n";

    if (r.all_ok) {
        std::cout << "  PASS  (all fields agree within " << tol << ")\n";
        return 0;
    } else {
        std::cout << "  FAIL  first failure: " << r.first_failure << "\n";
        return 1;
    }
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout.precision(12);

    int fail = 0;

    // Fixture 1: 2 trees of 2 leaves each. Aggregation correctness on a
    // tiny, hand-computable setup.
    {
        ground_truth gt = make_tiny_ground_truth();
        std::vector<size_t> leaves{2, 2};
        std::vector<phylo_gmm_problem::suff_stats_type> per_tree;
        fail += run_fixture("2 trees × 2 leaves (tiny hand-check)",
                            gt, leaves, /*seed=*/12345u, &per_tree);

        // Hand-check: with num_sites=1 and stationary_pi=true, the sum of
        // responsibility_total across i equals the total leaf count (each
        // leaf-site contributes 1.0 split across i). Per-tree contributions
        // should sum to 2 each.
        const double tol = 1e-10;
        for (std::size_t ti = 0; ti < per_tree.size(); ++ti) {
            const double s = per_tree[ti].responsibility_total.sum();
            std::cout << "  per-tree[" << ti
                      << "] responsibility_total.sum() = " << s
                      << "  (expected: 2.0)\n";
            if (std::abs(s - 2.0) > tol) {
                std::cout << "  FAIL  per-tree[" << ti
                          << "] does not sum to 2.0\n";
                fail += 1;
            }
        }
    }

    // Fixture 2: variable-size sanity check. Per-tree responsibility totals
    // should be ~20 and ~60, not split evenly.
    {
        ground_truth gt = make_sanity_ground_truth();
        std::vector<size_t> leaves{20, 60};
        std::vector<phylo_gmm_problem::suff_stats_type> per_tree;
        fail += run_fixture("2 trees × (20, 60) leaves (variable size)",
                            gt, leaves, /*seed=*/67890u, &per_tree);

        const double tol = 1e-10;
        const std::vector<double> expected_sums = {20.0, 60.0};
        for (std::size_t ti = 0; ti < per_tree.size(); ++ti) {
            const double s = per_tree[ti].responsibility_total.sum();
            std::cout << "  per-tree[" << ti
                      << "] responsibility_total.sum() = " << s
                      << "  (expected: " << expected_sums[ti] << ")\n";
            if (std::abs(s - expected_sums[ti]) > tol) {
                std::cout << "  FAIL  per-tree[" << ti
                          << "] does not sum to "
                          << expected_sums[ti] << "\n";
                fail += 1;
            }
        }
    }

    std::cout << "\n── Summary ─────────────────────────────────────────\n";
    if (fail == 0) {
        std::cout << "ALL PASS\n";
        return 0;
    } else {
        std::cout << "FAIL  (" << fail << " failure(s))\n";
        return 1;
    }
}
