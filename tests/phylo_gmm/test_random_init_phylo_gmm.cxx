// test_random_init_phylo_gmm: CTest unit test characterising the behaviour of
// phyloGMM EM under RANDOM initialisation ONLY (no k-means++).
//
// Context
// -------
// src/run_phylo_gmm.cxx currently runs 5 k-means++ + 5 random restarts and
// keeps the best by log-L. After recent fixes the k-means++ side reaches the
// consensus optimum on 5/5 restarts, so we want to drop random init from the
// default driver. This test continues to characterise the random-init
// behaviour in isolation so we can track any regression.
//
// What this test does
// -------------------
//   1. Builds the same synthetic 3-tree dataset as run_phylo_gmm.cxx (same
//      ground truth, same sim_rng seed 20260526u, same leaves_per_tree
//      {40, 40, 40}, same mean_blen = 0.5).
//   2. Runs 5 EM restarts using ONLY initialize_random with seeds 9000..9004.
//   3. Reports per-restart log L / iters / converged.
//   4. PASS criterion: at least 3 out of 5 restarts reach within 1e-3 of the
//      consensus best log L. EM under random init on a non-convex mixture
//      problem is expected to be lossy; 3/5 is a reasonable floor.
//
// TODO: dedup with run_phylo_gmm.cxx when phylo_gmm_problem moves to a header
// (task #37). The class below is a verbatim copy of the one currently inline
// in src/run_phylo_gmm.cxx. Picked (a) duplicate-the-class rather than (b)
// refactor-to-header because src/run_phylo_gmm.cxx is concurrently being
// edited by another agent (task #35 π-recovery diagnostic prints) and we
// don't want to interleave refactors with their changes.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
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

    void m_step(const suff_stats_type& ss) {
        const std::size_t k = k_();
        const std::size_t d = d_();
        const std::size_t num_sites = stm->sites_latent_states.size();

        std::vector<Eigen::VectorXd> mu_new = obs->mu;
        if (!fix_emission) {
            for (std::size_t i = 0; i < k; ++i) {
                const double N_i = ss.responsibility_total[i];
                if (N_i < 1e-12) continue;
                mu_new[i] = ss.mu_numerator[i] / N_i;
            }
        }

        std::vector<Eigen::MatrixXd> Sigma_new = obs->Sigma;
        if (!fix_emission) {
            for (std::size_t i = 0; i < k; ++i) {
                const double N_i = ss.responsibility_total[i];
                if (N_i < 1e-12) continue;
                Eigen::MatrixXd S = ss.Sigma_outer[i] / N_i
                                  - mu_new[i] * mu_new[i].transpose();
                S = 0.5 * (S + S.transpose());
                S += 1e-6 * Eigen::MatrixXd::Identity(d, d);
                Sigma_new[i] = S;
            }
        }

        Eigen::MatrixXd Q_new = stm->Q;
        if (!fix_Q) {
            for (std::size_t i = 0; i < k; ++i) {
                const double T_i = ss.T_total[i];
                if (T_i < 1e-12) continue;
                double row_off = 0.0;
                for (std::size_t j = 0; j < k; ++j) {
                    if (i == j) continue;
                    const double Nij = ss.N_total(i, j);
                    const double v = (Nij > 0.0) ? Nij / T_i : 0.0;
                    Q_new(i, j) = v;
                    row_off += v;
                }
                Q_new(i, i) = -row_off;
            }
        }

        if (!fix_emission) {
            obs->mu    = std::move(mu_new);
            obs->Sigma = std::move(Sigma_new);
        }
        if (!fix_Q) {
            stm->Q = Q_new;
        }

        stm->build();

        if (stationary_pi) {
            Eigen::VectorXd pi = extract_stationary_pi_();
            set_pi_on_root_prior_(pi);
        } else {
            double s = 0.0;
            for (std::size_t i = 0; i < k; ++i) s += ss.root_gamma_total[i];
            std::vector<double> pi(k, 0.0);
            if (s > 0.0) {
                for (std::size_t i = 0; i < k; ++i) pi[i] = ss.root_gamma_total[i] / s;
            } else {
                for (std::size_t i = 0; i < k; ++i) pi[i] = 1.0 / static_cast<double>(k);
            }
            for (std::size_t i = 0; i < k; ++i)
                if (pi[i] < 1e-12) pi[i] = 1e-12;
            double rs = 0.0;
            for (std::size_t i = 0; i < k; ++i) rs += pi[i];
            for (std::size_t i = 0; i < k; ++i) pi[i] /= rs;
            stm->root_prior = { pi };
        }

        obs->sites_latent_states = stm->sites_latent_states;
        obs->build();

        (void)num_sites;
    }

    // Random init only — k-means++ is intentionally omitted from this test.
    void initialize_random(uint64_t seed) {
        const std::size_t k = k_();
        const std::size_t d = obs->d;
        std::mt19937_64 rng(seed);

        const auto& Y = *observed_data;
        const std::size_t N = Y.size();
        if (N < k) {
            throw std::runtime_error(
                "initialize_random: not enough observations");
        }

        std::vector<std::size_t> idx(N);
        for (std::size_t i = 0; i < N; ++i) idx[i] = i;
        for (std::size_t i = 0; i < k; ++i) {
            std::uniform_int_distribution<std::size_t> u(i, N - 1);
            std::swap(idx[i], idx[u(rng)]);
        }
        std::vector<Eigen::VectorXd> centres;
        centres.reserve(k);
        for (std::size_t i = 0; i < k; ++i) centres.push_back(Y[idx[i]]);

        Eigen::VectorXd y_mean = Eigen::VectorXd::Zero(d);
        for (const auto& y : Y) y_mean += y;
        y_mean /= static_cast<double>(N);
        Eigen::MatrixXd Sg = Eigen::MatrixXd::Zero(d, d);
        for (const auto& y : Y) {
            Eigen::VectorXd c = y - y_mean;
            Sg += c * c.transpose();
        }
        Sg /= static_cast<double>(N);

        obs->d     = d;
        obs->mu    = std::move(centres);
        obs->Sigma.assign(k, Sg / static_cast<double>(k));
        for (std::size_t i = 0; i < k; ++i)
            obs->Sigma[i] += 1e-3 * Eigen::MatrixXd::Identity(d, d);

        const double q_off = 0.5;
        stm->Q = Eigen::MatrixXd::Zero(k, k);
        for (std::size_t i = 0; i < k; ++i) {
            double row_off = 0.0;
            for (std::size_t j = 0; j < k; ++j) {
                if (i == j) continue;
                stm->Q(i, j) = q_off;
                row_off += q_off;
            }
            stm->Q(i, i) = -row_off;
        }
        stm->root_prior = { std::vector<double>(k, 1.0 / static_cast<double>(k)) };

        stm->build();
        if (stationary_pi) {
            Eigen::VectorXd pi = extract_stationary_pi_();
            set_pi_on_root_prior_(pi);
        }
        obs->sites_latent_states = stm->sites_latent_states;
        obs->build();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Ground truth (matches src/run_phylo_gmm.cxx and tests/phylo_gmm/simulate.cxx).
// ─────────────────────────────────────────────────────────────────────────────

ground_truth make_ground_truth() {
    ground_truth gt;
    gt.pi = Eigen::VectorXd(3);
    gt.pi << 0.3, 0.4, 0.3;

    gt.Q = Eigen::MatrixXd(3, 3);
    gt.Q << -0.8, 0.5, 0.3,
             0.2, -0.9, 0.7,
             0.4, 0.1, -0.5;

    Eigen::VectorXd m0(2); m0 << 0.0,  0.0;
    Eigen::VectorXd m1(2); m1 << 3.0,  3.0;
    Eigen::VectorXd m2(2); m2 << -3.0, 3.0;
    gt.mu = {m0, m1, m2};

    Eigen::MatrixXd S0(2, 2); S0 << 0.5, 0.0,
                                     0.0, 0.5;
    Eigen::MatrixXd S1(2, 2); S1 << 1.0, 0.3,
                                     0.3, 0.5;
    Eigen::MatrixXd S2(2, 2); S2 << 0.8, -0.2,
                                    -0.2, 0.6;
    gt.Sigma = {S0, S1, S2};

    return gt;
}

// ─────────────────────────────────────────────────────────────────────────────
// One EM restart from a random init.
// ─────────────────────────────────────────────────────────────────────────────

struct restart_record {
    std::string label;
    double      log_likelihood;
    bool        converged;
    int         iterations;
};

restart_record run_random_restart(simulated_dataset& ds,
                                  const std::string& label,
                                  uint64_t seed) {
    const std::size_t k = 3;
    const std::size_t d = 2;

    ctmc_model stm;
    gmm_observation_model obs;
    obs.d = d;

    // Seed Q so ctmc_model::build() in set_params doesn't get an empty Q.
    stm.Q = Eigen::MatrixXd::Identity(k, k);
    for (std::size_t i = 0; i < k; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            if (i == j) continue;
            stm.Q(i, j) = 0.5;
            s += 0.5;
        }
        stm.Q(i, i) = -s;
    }
    stm.build();

    obs.mu.assign(k, Eigen::VectorXd::Zero(d));
    obs.Sigma.assign(k, Eigen::MatrixXd::Identity(d, d));
    obs.sites_latent_states = stm.sites_latent_states;
    obs.build();

    std::vector<std::vector<std::monostate>> node_data_storage(ds.trees.size());
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

    problem.initialize_random(seed);

    phylocore::optimize::em_options opts;
    opts.rel_tol           = 1e-5;
    opts.max_iter          = 200;
    opts.verbose           = false;
    opts.warn_on_decrease  = false;

    auto results = phylocore::optimize::em(problem, opts);

    restart_record r;
    r.label          = label;
    r.log_likelihood = results.log_likelihood;
    r.converged      = results.converged;
    r.iterations     = results.num_iterations;
    return r;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout.precision(8);

    // Simulate dataset — identical seeds & parameters to run_phylo_gmm.cxx so
    // the consensus log L is directly comparable to the driver's number.
    ground_truth gt = make_ground_truth();
    std::mt19937 sim_rng(20260526u);
    std::vector<size_t> leaves_per_tree{40, 40, 40};
    simulated_dataset ds =
        simulate_multi_tree(gt, leaves_per_tree, sim_rng, /*mean_blen=*/0.5);

    std::cout << "── phyloGMM EM random-init characterisation ───────────────\n";
    std::cout << "trees           = " << ds.trees.size() << "\n";
    std::cout << "total leaves    = " << ds.observed_data.size() << "\n\n";

    // 5 random-init restarts with seeds 9000..9004.
    const int num_restarts = 5;
    std::vector<restart_record> records;
    records.reserve(num_restarts);

    for (int s = 0; s < num_restarts; ++s) {
        std::cout << "  random restart " << s << " (seed=" << (9000u + s)
                  << ") ... " << std::flush;
        auto r = run_random_restart(ds,
                                    "random #" + std::to_string(s),
                                    /*seed=*/9000u + s);
        std::cout << "log L = " << r.log_likelihood
                  << "  (iters = " << r.iterations
                  << ", converged = " << (r.converged ? "true" : "false")
                  << ")\n";
        records.push_back(std::move(r));
    }

    // Consensus = best log L across all restarts.
    const restart_record* best = &records[0];
    for (const auto& r : records)
        if (r.log_likelihood > best->log_likelihood) best = &r;

    const double best_log_L = best->log_likelihood;
    // EM uses rel_tol = 1e-5 (LAML-Pro convention). Two restarts that find
    // the same local optimum can stop up to ~|best_log_L| * 1e-5 apart due to
    // stopping noise; 1e-2 covers that comfortably while still distinguishing
    // genuinely-different optima (the bad-init basin in this fixture is >30
    // nats away from the consensus optimum).
    const double match_tol  = 1e-2;

    // Count how many reached the consensus best.
    int num_at_best = 0;
    for (const auto& r : records) {
        if (best_log_L - r.log_likelihood <= match_tol) ++num_at_best;
    }

    std::cout << "\n── Summary ─────────────────────────────────────────────────\n";
    std::cout << "best log L (consensus) = " << best_log_L
              << "  (from " << best->label << ")\n";
    std::cout << "tolerance              = " << match_tol << "\n";
    std::cout << "restarts at consensus  = " << num_at_best
              << " / " << num_restarts << "\n\n";

    std::cout << "Per-restart gaps from consensus best:\n";
    for (const auto& r : records) {
        const double gap = best_log_L - r.log_likelihood;
        const bool   ok  = gap <= match_tol;
        std::cout << "  " << r.label
                  << "  log L = " << r.log_likelihood
                  << "  gap = " << gap
                  << "  " << (ok ? "[at-best]" : "[lossy] ") << "\n";
    }

    // PASS criterion: at least 3 / 5 random restarts hit the consensus best.
    const int required = 3;
    std::cout << "\nPASS criterion: >= " << required << " / " << num_restarts
              << " restarts within " << match_tol << " of best log L.\n";

    if (num_at_best >= required) {
        std::cout << "PASS  (" << num_at_best << " / " << num_restarts << ")\n";
        return 0;
    } else {
        std::cout << "FAIL  (" << num_at_best << " / " << num_restarts << ")\n";
        return 1;
    }
}
