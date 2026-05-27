// run_phylo_gmm: phyloGMM EM on real data (Newick tree + CSV observations).
//
// CLI:
//   run_phylo_gmm <tree_file> <obs_csv> <k> \
//       [max_iter] [tol] [q_init_rate] [fix_pi] [n_init]
//
// Conventions:
//   * obs CSV: first column = taxon/cell name, remaining columns = y_0..y_{d-1}
//   * tree:    single Newick per first non-empty line of <tree_file>
//   * 2 * n_init restarts total: n_init k-means++ (seeds {42, 43, ...}) plus
//     n_init random (seeds {1000, 1001, ...}). Deterministic seeds keep
//     per-restart outputs reproducible across runs.
//
// Output (JSON to stdout):
//   { "log_likelihood", "k", "d", "n_observed", "n_leaves",
//     "Q", "mu", "Sigma", "pi", "per_restart": [...] }

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include "io.h"
#include "meta_data.h"
#include "observation_models/gmm.h"
#include "optimize/em.h"
#include "optimize/estimation_results.h"
#include "optimize/hobolth_jensen.h"
#include "optimize/joint_optimization_problem.h"
#include "phylogeny.h"
#include "state_transition_models/ctmc.h"

using namespace phylocore;

// ─────────────────────────────────────────────────────────────────────────────
// phylo_gmm_problem — defined inline in this driver (anonymous namespace) so
// no public header is needed. The unit tests in tests/phylo_gmm/ each carry
// their own local copy of the same class for the same reason.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

class phylo_gmm_problem
    : public phylocore::optimize::joint_optimization_problem<
          phylocore::ctmc_model,
          phylocore::gmm_observation_model,
          Eigen::VectorXd,
          std::monostate> {
public:
    bool fix_Q         = false;
    bool fix_emission  = false;
    bool stationary_pi = true;  // disabled when CLI fix_pi=1 (pi held at 1/k)

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

    // Fix D: shared rebuild + stationary-pi-refresh path used by BOTH
    // set_params() and m_step() (and any future caller that mutates Q, mu,
    // Sigma, or root_prior). Matches the semantics of the canonical
    // set_params() path:
    //
    //   1. stm->build() unconditionally (cached eigendecompositions of Q must
    //      be refreshed after any parameter change).
    //   2. If stationary_pi is enabled AND Q has actually changed (fix_Q is
    //      false), re-derive root_prior from the stationary distribution of
    //      the new Q. When fix_Q=true the stationary pi is unchanged from the
    //      previous iteration, so we skip the recompute.
    //   3. Re-sync obs->sites_latent_states from stm.
    //   4. obs->sites_observation_states.clear() — drop any cached observed-
    //      state index from a previous build (set_params does this to avoid
    //      stale entries when the emission distribution mutates).
    //   5. obs->build().
    void apply_state_and_rebuild_() {
        stm->build();
        if (!fix_Q && stationary_pi) {
            set_pi_on_root_prior_(extract_stationary_pi_());
        }
        obs->sites_latent_states = stm->sites_latent_states;
        obs->sites_observation_states.clear();
        obs->build();
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
                for (std::size_t a = 0; a < d; ++a) p.push_back(obs->mu[i][a]);
            for (std::size_t i = 0; i < k; ++i)
                for (std::size_t r = 0; r < d; ++r)
                    for (std::size_t c = 0; c < d; ++c)
                        p.push_back(obs->Sigma[i](r, c));
        }
        if (!stationary_pi) {
            for (std::size_t i = 0; i + 1 < k; ++i)
                p.push_back(stm->root_prior[0][i]);
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
                            "Q < 0", idx);
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
            for (std::size_t i = 0; i < k; ++i)
                for (std::size_t a = 0; a < d; ++a) obs->mu[i][a] = params[idx++];
            for (std::size_t i = 0; i < k; ++i) {
                for (std::size_t r = 0; r < d; ++r)
                    for (std::size_t c = 0; c < d; ++c)
                        obs->Sigma[i](r, c) = params[idx++];
                obs->Sigma[i] = 0.5 * (obs->Sigma[i] + obs->Sigma[i].transpose());
                Eigen::LLT<Eigen::MatrixXd> llt(obs->Sigma[i]);
                if (llt.info() != Eigen::Success) {
                    throw phylocore::optimize::invalid_parameters_error(
                        "Sigma not PD", idx - 1);
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
                        "pi < 0", idx);
                }
                pi[i] = std::max(0.0, v); s += pi[i]; ++idx;
            }
            const double last = 1.0 - s;
            if (last < -1e-10) {
                throw phylocore::optimize::invalid_parameters_error(
                    "derived pi[k-1] < 0", idx - 1);
            }
            pi[k - 1] = std::max(0.0, last);
            stm->root_prior = { pi };
        }

        apply_state_and_rebuild_();
    }

    // EM Pattern-B
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
        if (!stationary_pi) ss.root_gamma_total = Eigen::VectorXd::Zero(k);

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
        if (!fix_Q) stm->Q = Q_new;

        // Non-stationary pi is derived from root_gamma_total here, BEFORE the
        // shared rebuild — apply_state_and_rebuild_() then refreshes the
        // stationary-pi branch (when stationary_pi=true && fix_Q=false) and
        // rebuilds stm + obs.
        if (!stationary_pi) {
            double s = 0.0;
            for (std::size_t i = 0; i < k; ++i) s += ss.root_gamma_total[i];
            std::vector<double> pi(k, 0.0);
            if (s > 0.0) {
                for (std::size_t i = 0; i < k; ++i) pi[i] = ss.root_gamma_total[i] / s;
            } else {
                for (std::size_t i = 0; i < k; ++i) pi[i] = 1.0 / static_cast<double>(k);
            }
            for (std::size_t i = 0; i < k; ++i) if (pi[i] < 1e-12) pi[i] = 1e-12;
            double rs = 0.0;
            for (std::size_t i = 0; i < k; ++i) rs += pi[i];
            for (std::size_t i = 0; i < k; ++i) pi[i] /= rs;
            stm->root_prior = { pi };
        }
        apply_state_and_rebuild_();
    }

    // Per-row total exit rate used to initialize Q. -1 means rate = 1/k
    // (each off-diag entry = (1/k)/(k-1)); a positive value v means rate = v
    // (each off-diag entry = v/(k-1)). Diagonal is always -rate.
    double q_init_rate_ = -1.0;

    void initialize_q_from_rate_() {
        const std::size_t k = k_();
        // Fix A: a single-state CTMC has no off-diagonal entries; the only
        // valid Q is [[0]] (rows of Q sum to zero, so the lone diagonal is 0).
        // Dividing rate / (k - 1) would be division by zero, so short-circuit.
        if (k == 1) {
            stm->Q = Eigen::MatrixXd::Zero(1, 1);
            return;
        }
        const double rate = (q_init_rate_ > 0.0)
                                ? q_init_rate_
                                : 1.0 / static_cast<double>(k);
        const double off  = rate / static_cast<double>(k - 1);
        stm->Q = Eigen::MatrixXd::Zero(k, k);
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = 0; j < k; ++j) {
                if (i == j) continue;
                stm->Q(i, j) = off;
            }
            stm->Q(i, i) = -rate;
        }
    }

    void initialize_kmeans_pp(uint64_t seed) {
        const std::size_t k = k_();
        const std::size_t d = obs->d;
        std::mt19937_64 rng(seed);
        const auto& Y = *observed_data;
        const std::size_t N = Y.size();
        if (N < k) throw std::runtime_error("initialize_kmeans_pp: N < k");

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
            if (total <= 0.0) { centres.push_back(Y[uidx(rng)]); continue; }
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

        obs->d  = d;
        obs->mu = std::move(centres);
        obs->Sigma.assign(k, Sg / static_cast<double>(k));
        for (std::size_t i = 0; i < k; ++i)
            obs->Sigma[i] += 1e-3 * Eigen::MatrixXd::Identity(d, d);

        initialize_q_from_rate_();
        stm->root_prior = { std::vector<double>(k, 1.0 / static_cast<double>(k)) };
        stm->build();
        if (stationary_pi) set_pi_on_root_prior_(extract_stationary_pi_());
        obs->sites_latent_states = stm->sites_latent_states;
        obs->build();
    }

    void initialize_random(uint64_t seed) {
        const std::size_t k = k_();
        const std::size_t d = obs->d;
        std::mt19937_64 rng(seed);
        const auto& Y = *observed_data;
        const std::size_t N = Y.size();
        if (N < k) throw std::runtime_error("initialize_random: N < k");
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

        obs->d  = d;
        obs->mu = std::move(centres);
        obs->Sigma.assign(k, Sg / static_cast<double>(k));
        for (std::size_t i = 0; i < k; ++i)
            obs->Sigma[i] += 1e-3 * Eigen::MatrixXd::Identity(d, d);

        initialize_q_from_rate_();
        stm->root_prior = { std::vector<double>(k, 1.0 / static_cast<double>(k)) };
        stm->build();
        if (stationary_pi) set_pi_on_root_prior_(extract_stationary_pi_());
        obs->sites_latent_states = stm->sites_latent_states;
        obs->build();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// CSV observation parser
// ─────────────────────────────────────────────────────────────────────────────
//
// First row is a header "cell,x0,x1,..."; each subsequent row is
// "<name>,<v0>,<v1>,...". d is inferred as (num_columns - 1). Empty lines
// are skipped.

struct obs_csv_result {
    std::unordered_map<std::string, Eigen::VectorXd> by_name;
    std::size_t d = 0;
};

obs_csv_result parse_observations_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open obs CSV: " + path);

    obs_csv_result out;
    std::string line;

    // Header
    if (!std::getline(in, line)) {
        throw std::runtime_error("Empty obs CSV: " + path);
    }
    {
        std::stringstream ss(line);
        std::string col;
        std::size_t ncol = 0;
        while (std::getline(ss, col, ',')) ++ncol;
        if (ncol < 2) {
            throw std::runtime_error("Obs CSV needs >= 2 cols (name + 1 feature)");
        }
        out.d = ncol - 1;
    }

    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        std::stringstream ss(line);
        std::string field;

        if (!std::getline(ss, field, ',')) continue;
        std::string name = field;

        Eigen::VectorXd y(out.d);
        for (std::size_t j = 0; j < out.d; ++j) {
            if (!std::getline(ss, field, ',')) {
                throw std::runtime_error(
                    "Obs CSV: row '" + name + "' has < " +
                    std::to_string(out.d) + " features");
            }
            y[static_cast<Eigen::Index>(j)] = std::stod(field);
        }
        out.by_name[std::move(name)] = std::move(y);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Restart record
// ─────────────────────────────────────────────────────────────────────────────

struct restart_record {
    std::string                  label;
    int                          seed;
    double                       log_likelihood;
    bool                         converged;
    int                          iterations;
    std::vector<Eigen::VectorXd> mu;
    std::vector<Eigen::MatrixXd> Sigma;
    Eigen::MatrixXd              Q;
    Eigen::VectorXd              pi;
};

restart_record run_em_restart(tree&                              t,
                              std::vector<Eigen::VectorXd>&     observed_data,
                              meta_data&                         metadata,
                              std::size_t                        k,
                              std::size_t                        d,
                              const std::string&                 label,
                              int                                seed,
                              bool                               use_kmeans_pp,
                              bool                               fix_pi,
                              double                             tol,
                              int                                max_iter,
                              double                             q_init_rate) {
    ctmc_model stm;
    gmm_observation_model obs;
    obs.d = d;

    stm.Q = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(k),
                                       static_cast<Eigen::Index>(k));
    for (std::size_t i = 0; i < k; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            if (i == j) continue;
            stm.Q(static_cast<Eigen::Index>(i),
                  static_cast<Eigen::Index>(j)) = 0.5;
            s += 0.5;
        }
        stm.Q(static_cast<Eigen::Index>(i),
              static_cast<Eigen::Index>(i)) = -s;
    }
    stm.build();

    obs.mu.assign(k, Eigen::VectorXd::Zero(d));
    obs.Sigma.assign(k, Eigen::MatrixXd::Identity(d, d));
    obs.sites_latent_states = stm.sites_latent_states;
    obs.build();

    std::vector<std::monostate> nd(t.num_nodes, std::monostate{});

    phylo_gmm_problem problem;
    problem.stm           = &stm;
    problem.obs           = &obs;
    problem.observed_data = &observed_data;
    problem.metadata      = &metadata;
    problem.trees.push_back(&t);
    problem.node_data.push_back(&nd);

    // CLI fix_pi=1 means hold pi = 1/k constant — we map that to
    // stationary_pi=false (so the EM doesn't recompute pi from Q) and
    // post-init clobber root_prior to 1/k. Default (fix_pi=0) uses
    // stationary_pi=true, so pi tracks the stationary distribution of Q.
    problem.stationary_pi = !fix_pi;
    problem.q_init_rate_  = q_init_rate;

    if (use_kmeans_pp) problem.initialize_kmeans_pp(static_cast<uint64_t>(seed));
    else               problem.initialize_random  (static_cast<uint64_t>(seed));

    if (fix_pi) {
        std::vector<double> uniform_pi(k, 1.0 / static_cast<double>(k));
        stm.root_prior = { uniform_pi };
    }

    phylocore::optimize::em_options opts;
    opts.rel_tol          = tol;
    opts.max_iter         = max_iter;
    opts.verbose          = false;
    opts.warn_on_decrease = false;

    auto results = phylocore::optimize::em(problem, opts);

    Eigen::VectorXd pi_rec(static_cast<Eigen::Index>(k));
    for (std::size_t i = 0; i < k; ++i)
        pi_rec[static_cast<Eigen::Index>(i)] = stm.root_prior[0][i];

    restart_record r;
    r.label          = label;
    r.seed           = seed;
    r.log_likelihood = results.log_likelihood;
    r.converged      = results.converged;
    r.iterations     = results.num_iterations;
    r.mu             = obs.mu;
    r.Sigma          = obs.Sigma;
    r.Q              = stm.Q;
    r.pi             = pi_rec;
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON dump
// ─────────────────────────────────────────────────────────────────────────────

void dump_matrix(std::ostream& os, const Eigen::MatrixXd& M) {
    os << "[";
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        if (i > 0) os << ", ";
        os << "[";
        for (Eigen::Index j = 0; j < M.cols(); ++j) {
            if (j > 0) os << ", ";
            os << M(i, j);
        }
        os << "]";
    }
    os << "]";
}

void dump_vec(std::ostream& os, const Eigen::VectorXd& v) {
    os << "[";
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (i > 0) os << ", ";
        os << v[i];
    }
    os << "]";
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <tree_file> <obs_csv> <k>"
                  << " [max_iter=100] [tol=1e-6] [q_init_rate=-1]"
                  << " [fix_pi=0] [n_init=10]\n";
        return 1;
    }

    const std::string tree_file = argv[1];
    const std::string obs_file  = argv[2];
    const std::size_t k         = std::stoul(argv[3]);
    const int    max_iter       = (argc > 4) ? std::stoi(argv[4]) : 100;
    const double tol            = (argc > 5) ? std::stod(argv[5]) : 1e-6;
    // q_init_rate semantics match OLD's run_phylo_gmm:
    //   q_init_rate > 0  : use that as the per-row total exit rate of the
    //                      initial Q (each off-diag entry = q_init_rate/(k-1),
    //                      diagonal = -q_init_rate).
    //   q_init_rate <= 0 : OLD default (rate = 1/k).
    const double q_init_rate    = (argc > 6) ? std::stod(argv[6]) : -1.0;
    const bool   fix_pi         = (argc > 7) && std::stoi(argv[7]) != 0;
    const int    n_init         = (argc > 8) ? std::stoi(argv[8]) : 10;

    std::cerr << "── phyloGMM (phylocore impl) ────────────────────────────────\n";
    std::cerr << "tree     = " << tree_file << "\n";
    std::cerr << "obs      = " << obs_file  << "\n";
    std::cerr << "k        = " << k         << "\n";
    std::cerr << "max_iter = " << max_iter  << ", tol = " << tol << "\n";
    std::cerr << "fix_pi   = " << fix_pi    << ", n_init = " << n_init << "\n";
    {
        const double effective_rate =
            (q_init_rate > 0.0) ? q_init_rate : 1.0 / static_cast<double>(k);
        std::cerr << "q_init_rate = " << q_init_rate
                  << " (effective rate = " << effective_rate;
        // Fix B: k=1 has no off-diagonal entries, so reporting
        // "rate / (k-1)" would divide by zero. Print a k=1-specific note.
        if (k >= 2) {
            std::cerr << ", off-diag entry = "
                      << effective_rate / static_cast<double>(k - 1);
        } else {
            std::cerr << ", k=1: no off-diag entries (Q = [[0]])";
        }
        std::cerr << ")\n";
    }

    // ── Parse tree ─────────────────────────────────────────────────────────
    auto trees = parse_newick_trees(tree_file);
    if (trees.empty()) { std::cerr << "no trees parsed\n"; return 2; }
    tree& t = trees[0];
    std::cerr << "Tree: " << t.num_leaves << " leaves, "
              << t.num_nodes << " nodes\n";

    // ── Parse observations ─────────────────────────────────────────────────
    auto obs_csv = parse_observations_csv(obs_file);
    const std::size_t d = obs_csv.d;
    std::cerr << "Obs:  " << obs_csv.by_name.size() << " cells, d = " << d << "\n";

    // ── Align leaves: build observed_data indexed by leaf.data ─────────────
    //
    // joint_optimization_problem expects observed_data indexed by taxa_id
    // (= leaf vertex .data, set to 0..num_leaves-1 in parse order). For each
    // leaf we write y into observed_data[taxa_id] if the leaf name matched a
    // row in the CSV. Every leaf must be observed; we ERROR out otherwise
    // rather than silently corrupting log-L.
    std::vector<Eigen::VectorXd> observed_data(t.num_leaves,
                                                Eigen::VectorXd::Zero(d));
    std::size_t n_observed = 0;
    std::vector<std::string> leaf_names(t.num_leaves);
    for (int node_id : t.tree.nodes()) {
        if (t.tree.out_degree(node_id) != 0) continue;
        const std::string& name = t.node_names[node_id];
        const std::size_t leaf_data = t.tree[node_id].data;
        leaf_names[leaf_data] = name;
        auto it = obs_csv.by_name.find(name);
        if (it != obs_csv.by_name.end()) {
            observed_data[leaf_data] = it->second;
            ++n_observed;
        }
    }
    std::cerr << "Aligned: " << n_observed << " / " << t.num_leaves << " leaves\n";
    if (n_observed != t.num_leaves) {
        std::cerr << "ERROR: every leaf must be observed.\n"
                  << "  Drop unobserved leaves from the tree or fill in observations.\n";
        return 3;
    }

    // ── Metadata ───────────────────────────────────────────────────────────
    meta_data metadata;
    metadata.num_sites         = 1;
    metadata.max_alphabet_size = k;
    metadata.alphabet_sizes    = { k };
    metadata.taxa_names        = std::move(leaf_names);

    // ── 2 * n_init restarts ────────────────────────────────────────────────
    std::vector<restart_record> records;
    records.reserve(2 * static_cast<std::size_t>(n_init));

    // k-means++ seeds {42, 43, ...} for reproducibility.
    for (int i = 0; i < n_init; ++i) {
        const int seed = 42 + i;
        std::cerr << "  kmeans++ " << (i + 1) << "/" << n_init
                  << "  seed=" << seed << " ... " << std::flush;
        auto r = run_em_restart(t, observed_data, metadata, k, d,
                                "kmeans++ #" + std::to_string(i),
                                seed, /*use_kmeans_pp=*/true, fix_pi,
                                tol, max_iter, q_init_rate);
        std::cerr << "log L = " << r.log_likelihood
                  << " (iters=" << r.iterations
                  << ", converged=" << r.converged << ")\n";
        records.push_back(std::move(r));
    }
    // random seeds {1000, 1001, ...} for reproducibility.
    for (int i = 0; i < n_init; ++i) {
        const int seed = 1000 + i;
        std::cerr << "  random   " << (i + 1) << "/" << n_init
                  << "  seed=" << seed << " ... " << std::flush;
        auto r = run_em_restart(t, observed_data, metadata, k, d,
                                "random #" + std::to_string(i),
                                seed, /*use_kmeans_pp=*/false, fix_pi,
                                tol, max_iter, q_init_rate);
        std::cerr << "log L = " << r.log_likelihood
                  << " (iters=" << r.iterations
                  << ", converged=" << r.converged << ")\n";
        records.push_back(std::move(r));
    }

    // ── Best ───────────────────────────────────────────────────────────────
    const restart_record* best = &records[0];
    for (const auto& r : records)
        if (r.log_likelihood > best->log_likelihood) best = &r;

    std::cerr << "\nBest: " << best->label << " seed=" << best->seed
              << " log L = " << best->log_likelihood << "\n";

    // ── JSON output ────────────────────────────────────────────────────────
    std::cout.precision(10);
    std::cout << "{\n";
    std::cout << "  \"log_likelihood\": " << best->log_likelihood << ",\n";
    std::cout << "  \"k\": " << k << ",\n";
    std::cout << "  \"d\": " << d << ",\n";
    std::cout << "  \"n_observed\": " << n_observed << ",\n";
    std::cout << "  \"n_leaves\": " << t.num_leaves << ",\n";

    std::cout << "  \"Q\": ";
    dump_matrix(std::cout, best->Q);
    std::cout << ",\n";

    std::cout << "  \"mu\": [";
    for (std::size_t i = 0; i < k; ++i) {
        if (i > 0) std::cout << ", ";
        dump_vec(std::cout, best->mu[i]);
    }
    std::cout << "],\n";

    std::cout << "  \"Sigma\": [";
    for (std::size_t i = 0; i < k; ++i) {
        if (i > 0) std::cout << ", ";
        dump_matrix(std::cout, best->Sigma[i]);
    }
    std::cout << "],\n";

    std::cout << "  \"pi\": ";
    dump_vec(std::cout, best->pi);
    std::cout << ",\n";

    // Extra: per-restart log-L list, plus convergence flag — supports the
    // success-rate diff that's the headline metric of Task #33.
    std::cout << "  \"per_restart\": [\n";
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        if (i > 0) std::cout << ",\n";
        std::cout << "    {\"label\":\"" << r.label
                  << "\",\"seed\":" << r.seed
                  << ",\"log_likelihood\":" << r.log_likelihood
                  << ",\"iterations\":" << r.iterations
                  << ",\"converged\":" << (r.converged ? "true" : "false")
                  << "}";
    }
    std::cout << "\n  ]\n";
    std::cout << "}\n";
    return 0;
}
