// test_phylo_gmm_flags: CTest unit test that exercises the conditional logic
// for the three model-level flags on phylo_gmm_problem:
//
//   - fix_Q          (default false): skip Q update + Hobolth-Jensen aggregation
//   - fix_emission   (default false): skip μ, Σ updates + leaf responsibility
//                                     aggregation for emission moments
//   - stationary_pi  (default true ): π is derived from Q's stationary;
//                                     otherwise π is updated from root_gamma_total
//
// run_phylo_gmm.cxx's existing CTest only covers the defaults
// (fix_Q = fix_emission = false, stationary_pi = true). This test sweeps four
// non-default flag combinations and verifies the invariants implied by each
// flag setting (params held fixed actually stay fixed, π comes from the right
// source, etc.).
//
// TODO: dedup with task #37 when the class moves to a header. The
// phylo_gmm_problem class below is a verbatim copy of the one currently in an
// anonymous namespace in src/run_phylo_gmm.cxx (same pattern used by
// tests/phylo_gmm/test_random_init_phylo_gmm.cxx).

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

    // k-means++ initialisation (mirrors src/run_phylo_gmm.cxx so we can seed
    // from a known starting point identical to the existing test fixtures).
    void initialize_kmeans_pp(uint64_t seed) {
        const std::size_t k = k_();
        const std::size_t d = obs->d;
        std::mt19937_64 rng(seed);

        const auto& Y = *observed_data;
        const std::size_t N = Y.size();
        if (N < k) {
            throw std::runtime_error(
                "initialize_kmeans_pp: not enough observations");
        }

        std::uniform_int_distribution<std::size_t> uidx(0, N - 1);
        std::vector<Eigen::VectorXd> centres;
        centres.reserve(k);
        centres.push_back(Y[uidx(rng)]);

        std::vector<double> sqd(N, phylocore::POSITIVE_INFINITY);
        while (centres.size() < k) {
            const Eigen::VectorXd& last = centres.back();
            double total = 0.0;
            for (std::size_t n = 0; n < N; ++n) {
                const double dist = (Y[n] - last).squaredNorm();
                if (dist < sqd[n]) sqd[n] = dist;
                total += sqd[n];
            }
            if (total <= 0.0) {
                centres.push_back(Y[uidx(rng)]);
                continue;
            }
            std::uniform_real_distribution<double> u(0.0, total);
            const double r = u(rng);
            double acc = 0.0;
            std::size_t pick = N - 1;
            for (std::size_t n = 0; n < N; ++n) {
                acc += sqd[n];
                if (r <= acc) { pick = n; break; }
            }
            centres.push_back(Y[pick]);
        }

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
// Wiring helpers.
// ─────────────────────────────────────────────────────────────────────────────

// Owning bundle so the model/state pointers stay alive for the lifetime of
// each test scenario.
struct problem_bundle {
    ctmc_model                                   stm;
    gmm_observation_model                        obs;
    std::vector<std::vector<std::monostate>>     node_data_storage;
    phylo_gmm_problem                            problem;
};

// Build a problem_bundle wired up to `ds` with a generic seed Q so that
// ctmc_model::build() succeeds even before any initialize_* call.
std::unique_ptr<problem_bundle> make_bundle(simulated_dataset& ds,
                                            std::size_t k,
                                            std::size_t d) {
    auto b = std::make_unique<problem_bundle>();
    b->obs.d = d;

    b->stm.Q = Eigen::MatrixXd::Identity(k, k);
    for (std::size_t i = 0; i < k; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            if (i == j) continue;
            b->stm.Q(i, j) = 0.5;
            s += 0.5;
        }
        b->stm.Q(i, i) = -s;
    }
    b->stm.build();

    b->obs.mu.assign(k, Eigen::VectorXd::Zero(d));
    b->obs.Sigma.assign(k, Eigen::MatrixXd::Identity(d, d));
    b->obs.sites_latent_states = b->stm.sites_latent_states;
    b->obs.build();

    b->node_data_storage.assign(ds.trees.size(), {});
    for (std::size_t i = 0; i < ds.trees.size(); ++i)
        b->node_data_storage[i].assign(ds.trees[i].num_nodes, std::monostate{});

    b->problem.stm           = &b->stm;
    b->problem.obs           = &b->obs;
    b->problem.observed_data = &ds.observed_data;
    b->problem.metadata      = &ds.metadata;
    b->problem.trees.reserve(ds.trees.size());
    b->problem.node_data.reserve(ds.trees.size());
    for (std::size_t i = 0; i < ds.trees.size(); ++i) {
        b->problem.trees.push_back(&ds.trees[i]);
        b->problem.node_data.push_back(&b->node_data_storage[i]);
    }
    return b;
}

// Overwrite the *current* obs and stm state with the ground-truth Q/μ/Σ/π,
// then refresh caches. Used to inject the "initialized at truth" starting
// point for fix_* combinations.
void seed_truth(problem_bundle& b, const ground_truth& gt) {
    const std::size_t k = static_cast<std::size_t>(gt.pi.size());
    b.stm.Q = gt.Q;
    std::vector<double> pi_v(k);
    for (std::size_t i = 0; i < k; ++i)
        pi_v[i] = gt.pi[static_cast<Eigen::Index>(i)];
    b.stm.root_prior = { pi_v };
    b.stm.build();

    b.obs.mu    = gt.mu;
    b.obs.Sigma = gt.Sigma;
    b.obs.sites_latent_states = b.stm.sites_latent_states;
    b.obs.build();
}

// Pretty-print helpers.
void print_Q(const std::string& label, const Eigen::MatrixXd& Q) {
    std::cout << "    " << label << " =\n";
    for (Eigen::Index i = 0; i < Q.rows(); ++i) {
        std::cout << "      " << Q.row(i) << "\n";
    }
}

void print_mu(const std::string& label, const std::vector<Eigen::VectorXd>& mu) {
    for (std::size_t i = 0; i < mu.size(); ++i) {
        std::cout << "    " << label << "[" << i << "] = "
                  << mu[i].transpose() << "\n";
    }
}

void print_Sigma(const std::string& label,
                 const std::vector<Eigen::MatrixXd>& Sigma) {
    for (std::size_t i = 0; i < Sigma.size(); ++i) {
        std::cout << "    " << label << "[" << i << "] =\n";
        for (Eigen::Index r = 0; r < Sigma[i].rows(); ++r) {
            std::cout << "      " << Sigma[i].row(r) << "\n";
        }
    }
}

// Common EM options for these tests. max_iter = 50 keeps runtime down.
phylocore::optimize::em_options test_em_opts() {
    phylocore::optimize::em_options opts;
    opts.rel_tol           = 1e-5;
    opts.max_iter          = 50;
    opts.verbose           = false;
    opts.warn_on_decrease  = false;
    return opts;
}

// Per-scenario result.
struct scenario_result {
    std::string label;
    bool        passed = false;
    std::string detail;
};

// (a) fix_Q = true. Initialize Q at truth via seed_truth + k-means++ for
//     μ, Σ. Then OVERWRITE Q back to truth after init (initialize_kmeans_pp
//     resets Q to the uniform-q_off matrix). Run EM. Assert Q is bit-exact
//     unchanged. μ, Σ should still move; log-L should improve.
scenario_result test_fix_Q(simulated_dataset& ds, const ground_truth& gt) {
    scenario_result sr;
    sr.label = "(a) fix_Q = true";

    auto b = make_bundle(ds, 3, 2);
    b->problem.fix_Q         = true;
    b->problem.fix_emission  = false;
    b->problem.stationary_pi = true;

    // K-means++ seed for μ, Σ — same seed as one of the existing kmeans++ runs.
    b->problem.initialize_kmeans_pp(1000u);

    // Now overwrite Q with the truth Q (k-means++ set Q to the uniform 0.5
    // off-diagonal matrix). fix_Q means this Q will be retained throughout EM.
    b->stm.Q = gt.Q;
    b->stm.build();
    // stationary_pi = true; refresh π from this new Q.
    {
        Eigen::VectorXd pi = b->problem.extract_stationary_pi_();
        b->problem.set_pi_on_root_prior_(pi);
    }
    b->obs.sites_latent_states = b->stm.sites_latent_states;
    b->obs.build();

    const Eigen::MatrixXd Q_initial = b->stm.Q;
    const auto mu_initial    = b->obs.mu;
    const auto Sigma_initial = b->obs.Sigma;
    const double log_L_initial = b->problem.log_likelihood();

    auto results = phylocore::optimize::em(b->problem, test_em_opts());

    const Eigen::MatrixXd& Q_final = b->stm.Q;
    const auto& mu_final    = b->obs.mu;
    const auto& Sigma_final = b->obs.Sigma;

    std::cout << "  initial log L = " << log_L_initial << "\n";
    std::cout << "  final   log L = " << results.log_likelihood
              << "  (iters=" << results.num_iterations
              << ", converged=" << (results.converged ? "true" : "false") << ")\n";
    print_Q("Q (initial = truth)", Q_initial);
    print_Q("Q (final)          ", Q_final);
    print_mu("mu (initial)", mu_initial);
    print_mu("mu (final)  ", mu_final);

    // Invariant: Q exactly unchanged.
    double q_diff = (Q_final - Q_initial).cwiseAbs().maxCoeff();
    std::cout << "  ||Q_final - Q_initial||_inf = " << q_diff << "\n";

    // μ, Σ should have moved (k-means++ centres are not at the MLE).
    double mu_diff = 0.0;
    for (std::size_t i = 0; i < mu_final.size(); ++i)
        mu_diff += (mu_final[i] - mu_initial[i]).norm();
    double sigma_diff = 0.0;
    for (std::size_t i = 0; i < Sigma_final.size(); ++i)
        sigma_diff += (Sigma_final[i] - Sigma_initial[i]).norm();
    std::cout << "  sum ||mu_final - mu_initial|| = " << mu_diff << "\n";
    std::cout << "  sum ||Sigma_final - Sigma_initial||_F = " << sigma_diff << "\n";

    bool q_ok        = q_diff == 0.0;  // bit-exact
    bool mu_moved    = mu_diff > 1e-6;
    bool sigma_moved = sigma_diff > 1e-6;
    bool ll_improved = results.log_likelihood > log_L_initial - 1e-6;

    sr.passed = q_ok && mu_moved && sigma_moved && ll_improved;
    if (!sr.passed) {
        sr.detail = (q_ok        ? "" : "Q drifted; ") +
                    std::string(mu_moved    ? "" : "mu unchanged; ") +
                    std::string(sigma_moved ? "" : "Sigma unchanged; ") +
                    std::string(ll_improved ? "" : "log-L decreased");
    }
    return sr;
}

// (b) fix_emission = true. Initialize μ, Σ at truth, leave Q at uniform 0.5
//     seed. Run EM. Assert μ, Σ are bit-exact unchanged; Q should move; log-L
//     should improve.
scenario_result test_fix_emission(simulated_dataset& ds, const ground_truth& gt) {
    scenario_result sr;
    sr.label = "(b) fix_emission = true";

    auto b = make_bundle(ds, 3, 2);
    b->problem.fix_Q         = false;
    b->problem.fix_emission  = true;
    b->problem.stationary_pi = true;

    // k-means++ first (this populates Q with uniform 0.5 off-diag) then
    // overwrite μ, Σ with the truth.
    b->problem.initialize_kmeans_pp(1000u);
    b->obs.mu    = gt.mu;
    b->obs.Sigma = gt.Sigma;
    b->obs.build();

    const Eigen::MatrixXd Q_initial = b->stm.Q;
    const auto mu_initial    = b->obs.mu;
    const auto Sigma_initial = b->obs.Sigma;
    const double log_L_initial = b->problem.log_likelihood();

    auto results = phylocore::optimize::em(b->problem, test_em_opts());

    const Eigen::MatrixXd& Q_final = b->stm.Q;
    const auto& mu_final    = b->obs.mu;
    const auto& Sigma_final = b->obs.Sigma;

    std::cout << "  initial log L = " << log_L_initial << "\n";
    std::cout << "  final   log L = " << results.log_likelihood
              << "  (iters=" << results.num_iterations
              << ", converged=" << (results.converged ? "true" : "false") << ")\n";
    print_Q("Q (initial)", Q_initial);
    print_Q("Q (final)  ", Q_final);
    print_mu("mu (initial = truth)", mu_initial);
    print_mu("mu (final)          ", mu_final);
    print_Sigma("Sigma (initial = truth)", Sigma_initial);
    print_Sigma("Sigma (final)          ", Sigma_final);

    double mu_diff = 0.0;
    for (std::size_t i = 0; i < mu_final.size(); ++i)
        mu_diff += (mu_final[i] - mu_initial[i]).cwiseAbs().maxCoeff();
    double sigma_diff = 0.0;
    for (std::size_t i = 0; i < Sigma_final.size(); ++i)
        sigma_diff += (Sigma_final[i] - Sigma_initial[i]).cwiseAbs().maxCoeff();
    double q_diff = (Q_final - Q_initial).norm();

    std::cout << "  max ||mu_final - mu_initial||_inf       = " << mu_diff << "\n";
    std::cout << "  max ||Sigma_final - Sigma_initial||_inf = " << sigma_diff << "\n";
    std::cout << "  ||Q_final - Q_initial||_F               = " << q_diff << "\n";

    bool mu_ok       = mu_diff == 0.0;
    bool sigma_ok    = sigma_diff == 0.0;
    bool q_moved     = q_diff > 1e-6;
    bool ll_improved = results.log_likelihood > log_L_initial - 1e-6;

    sr.passed = mu_ok && sigma_ok && q_moved && ll_improved;
    if (!sr.passed) {
        sr.detail = (mu_ok      ? "" : "mu drifted; ") +
                    std::string(sigma_ok    ? "" : "Sigma drifted; ") +
                    std::string(q_moved     ? "" : "Q unchanged; ") +
                    std::string(ll_improved ? "" : "log-L decreased");
    }
    return sr;
}

// (c) stationary_pi = false. EM updates π from root_gamma_total instead of
//     deriving it from Q's stationary distribution. Run two EM runs from the
//     SAME starting point — one with stationary_pi = false, one with
//     stationary_pi = true — and assert the recovered π differs.
scenario_result test_stationary_pi_false(simulated_dataset& ds,
                                          const ground_truth& gt) {
    (void)gt;
    scenario_result sr;
    sr.label = "(c) stationary_pi = false";

    auto b_a = make_bundle(ds, 3, 2);
    b_a->problem.fix_Q         = false;
    b_a->problem.fix_emission  = false;
    b_a->problem.stationary_pi = false;
    b_a->problem.initialize_kmeans_pp(1000u);
    // initialize_kmeans_pp sets stm->root_prior = uniform when stationary_pi
    // is false (the stationary-pi refresh branch is skipped). That's fine —
    // it's a valid starting π for non-stationary EM.

    auto results_a = phylocore::optimize::em(b_a->problem, test_em_opts());

    auto b_b = make_bundle(ds, 3, 2);
    b_b->problem.fix_Q         = false;
    b_b->problem.fix_emission  = false;
    b_b->problem.stationary_pi = true;
    b_b->problem.initialize_kmeans_pp(1000u);

    auto results_b = phylocore::optimize::em(b_b->problem, test_em_opts());

    const std::size_t k = b_a->problem.k_();
    Eigen::VectorXd pi_nonstat(k);
    Eigen::VectorXd pi_stat(k);
    for (std::size_t i = 0; i < k; ++i) {
        pi_nonstat[static_cast<Eigen::Index>(i)] = b_a->stm.root_prior[0][i];
        pi_stat   [static_cast<Eigen::Index>(i)] = b_b->stm.root_prior[0][i];
    }

    std::cout << "  log L (stationary_pi=false) = " << results_a.log_likelihood
              << "  (iters=" << results_a.num_iterations
              << ", converged=" << (results_a.converged ? "true" : "false") << ")\n";
    std::cout << "  log L (stationary_pi=true)  = " << results_b.log_likelihood
              << "  (iters=" << results_b.num_iterations
              << ", converged=" << (results_b.converged ? "true" : "false") << ")\n";
    std::cout << "  pi (stationary_pi=false)    = " << pi_nonstat.transpose() << "\n";
    std::cout << "  pi (stationary_pi=true)     = " << pi_stat.transpose() << "\n";

    // The two π's should differ — they're computed from different sources.
    double pi_diff = (pi_nonstat - pi_stat).lpNorm<1>();
    std::cout << "  ||pi_nonstat - pi_stat||_L1 = " << pi_diff << "\n";

    // Sanity: each π is a valid probability vector.
    auto is_simplex = [](const Eigen::VectorXd& v) {
        if (v.minCoeff() < -1e-9) return false;
        double s = v.sum();
        return std::abs(s - 1.0) < 1e-6;
    };

    bool pi_differs = pi_diff > 1e-4;  // expected: they are computed differently
    bool simplex_ok = is_simplex(pi_nonstat) && is_simplex(pi_stat);

    sr.passed = pi_differs && simplex_ok;
    if (!sr.passed) {
        sr.detail = std::string(pi_differs ? "" : "pi did not differ; ") +
                    std::string(simplex_ok ? "" : "pi not on simplex; ");
    }
    return sr;
}

// (d) fix_Q = true AND fix_emission = true. Only π update (or nothing if
//     stationary_pi). Initialize Q, μ, Σ at truth. Run EM. Assert nothing
//     changes for Q, μ, Σ. With stationary_pi=true, π is recomputed each
//     iter from the (unchanged) Q's stationary — so π should also stay put.
//     With stationary_pi=false, π may move (it's updated from root_gamma_total).
scenario_result test_fix_both(simulated_dataset& ds, const ground_truth& gt) {
    scenario_result sr;
    sr.label = "(d) fix_Q = true AND fix_emission = true";

    auto b = make_bundle(ds, 3, 2);
    b->problem.fix_Q         = true;
    b->problem.fix_emission  = true;
    b->problem.stationary_pi = true;

    // initialize_kmeans_pp seeds μ via k-means++ and sets Q to uniform 0.5
    // off-diag. Then we OVERWRITE everything with the truth to test the
    // "fixed at truth" invariant.
    b->problem.initialize_kmeans_pp(1000u);
    seed_truth(*b, gt);

    // seed_truth installs root_prior = gt.pi, which is not necessarily the
    // stationary distribution of gt.Q. With stationary_pi=true, EM will
    // recompute root_prior from gt.Q's stationary on iter 1, so pi_initial
    // must already be that stationary or the "pi_final == pi_initial"
    // invariant becomes a one-step-correction observation, not drift.
    {
        Eigen::VectorXd pi = b->problem.extract_stationary_pi_();
        b->problem.set_pi_on_root_prior_(pi);
    }

    const Eigen::MatrixXd Q_initial = b->stm.Q;
    const auto mu_initial    = b->obs.mu;
    const auto Sigma_initial = b->obs.Sigma;
    const std::size_t k = b->problem.k_();
    Eigen::VectorXd pi_initial(k);
    for (std::size_t i = 0; i < k; ++i)
        pi_initial[static_cast<Eigen::Index>(i)] = b->stm.root_prior[0][i];
    const double log_L_initial = b->problem.log_likelihood();

    auto results = phylocore::optimize::em(b->problem, test_em_opts());

    const Eigen::MatrixXd& Q_final = b->stm.Q;
    const auto& mu_final    = b->obs.mu;
    const auto& Sigma_final = b->obs.Sigma;
    Eigen::VectorXd pi_final(k);
    for (std::size_t i = 0; i < k; ++i)
        pi_final[static_cast<Eigen::Index>(i)] = b->stm.root_prior[0][i];

    double q_diff = (Q_final - Q_initial).cwiseAbs().maxCoeff();
    double mu_diff = 0.0;
    for (std::size_t i = 0; i < mu_final.size(); ++i)
        mu_diff += (mu_final[i] - mu_initial[i]).cwiseAbs().maxCoeff();
    double sigma_diff = 0.0;
    for (std::size_t i = 0; i < Sigma_final.size(); ++i)
        sigma_diff += (Sigma_final[i] - Sigma_initial[i]).cwiseAbs().maxCoeff();
    double pi_diff = (pi_final - pi_initial).cwiseAbs().maxCoeff();

    std::cout << "  initial log L = " << log_L_initial << "\n";
    std::cout << "  final   log L = " << results.log_likelihood
              << "  (iters=" << results.num_iterations
              << ", converged=" << (results.converged ? "true" : "false") << ")\n";
    std::cout << "  ||Q_final - Q_initial||_inf       = " << q_diff << "\n";
    std::cout << "  max ||mu_final - mu_initial||_inf = " << mu_diff << "\n";
    std::cout << "  max ||Sigma_final - Sigma||_inf   = " << sigma_diff << "\n";
    std::cout << "  ||pi_final - pi_initial||_inf     = " << pi_diff << "\n";

    // With both fix flags set AND stationary_pi=true, π is recomputed each
    // iter as extract_stationary_pi_(Q). Since Q is fixed, this reproduces
    // the same π — modulo eigendecomp rounding. Use a small numerical
    // tolerance for π rather than bit-exact.
    bool q_ok     = q_diff == 0.0;
    bool mu_ok    = mu_diff == 0.0;
    bool sigma_ok = sigma_diff == 0.0;
    bool pi_ok    = pi_diff < 1e-9;  // eigendecomp may add tiny rounding
    bool ll_ok    = std::isfinite(results.log_likelihood);

    sr.passed = q_ok && mu_ok && sigma_ok && pi_ok && ll_ok;
    if (!sr.passed) {
        sr.detail = std::string(q_ok     ? "" : "Q drifted; ") +
                    std::string(mu_ok    ? "" : "mu drifted; ") +
                    std::string(sigma_ok ? "" : "Sigma drifted; ") +
                    std::string(pi_ok    ? "" : "pi drifted; ") +
                    std::string(ll_ok    ? "" : "log-L not finite");
    }
    return sr;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout.precision(8);

    // ── Simulate dataset ───────────────────────────────────────────────────
    ground_truth gt = make_ground_truth();
    std::mt19937 sim_rng(20260526u);
    std::vector<size_t> leaves_per_tree{40, 40, 40};
    simulated_dataset ds =
        simulate_multi_tree(gt, leaves_per_tree, sim_rng, /*mean_blen=*/0.5);

    std::cout << "── phyloGMM flag-combination tests ─────────────────────────\n";
    std::cout << "trees           = " << ds.trees.size() << "\n";
    std::cout << "total leaves    = " << ds.observed_data.size() << "\n\n";

    std::vector<scenario_result> results;
    results.reserve(4);

    std::cout << "── (a) fix_Q = true ────────────────────────────────────────\n";
    results.push_back(test_fix_Q(ds, gt));
    std::cout << "\n── (b) fix_emission = true ─────────────────────────────────\n";
    results.push_back(test_fix_emission(ds, gt));
    std::cout << "\n── (c) stationary_pi = false ───────────────────────────────\n";
    results.push_back(test_stationary_pi_false(ds, gt));
    std::cout << "\n── (d) fix_Q = true AND fix_emission = true ────────────────\n";
    results.push_back(test_fix_both(ds, gt));

    std::cout << "\n── Summary ─────────────────────────────────────────────────\n";
    int pass_count = 0;
    for (const auto& r : results) {
        std::cout << "  " << r.label << ": "
                  << (r.passed ? "PASS" : "FAIL");
        if (!r.passed && !r.detail.empty()) std::cout << "  (" << r.detail << ")";
        std::cout << "\n";
        if (r.passed) ++pass_count;
    }
    std::cout << "  total: " << pass_count << " / " << results.size() << "\n";

    return (pass_count == static_cast<int>(results.size())) ? 0 : 1;
}
