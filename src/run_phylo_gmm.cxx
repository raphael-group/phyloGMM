// run_phylo_gmm: phyloGMM maximum-likelihood estimation via EM. A discrete
// CTMC (ctmc_model — general, not assumed reversible) evolves latent states
// X_v on a rooted tree; at each leaf ℓ we observe y_ℓ | X_ℓ = i ~ N(μ_i, Σ_i)
// (gmm_observation_model). The EM driver (optimize::em) iterates closed-form
// M-step updates derived from the standard library posteriors plus the
// Hobolth-Jensen aggregate (E[N_ij | y], E[T_i | y]).
//
// Validation:
//   * Synthesise a 3-tree dataset (40 leaves each) from a known ground truth.
//   * Run EM from 10 restarts (5 k-means++ + 5 random), keep the best by log-L.
//   * Compare recovered (μ, π) to ground truth under best label permutation.
//   * Pass if best log-L >= log-L-at-truth - 5.0 (MLE should beat or slightly
//     trail the truth log-L since the truth is noised).

#include <algorithm>
#include <cmath>
#include <cstddef>
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

#include "../tests/phylo_gmm/simulator.h"

using namespace phylocore;
using phylocore::tests::phylo_gmm::ground_truth;
using phylocore::tests::phylo_gmm::simulate_multi_tree;
using phylocore::tests::phylo_gmm::simulated_dataset;

// ─────────────────────────────────────────────────────────────────────────────
// phylo_gmm_problem
// ─────────────────────────────────────────────────────────────────────────────

namespace {

class phylo_gmm_problem
    : public phylocore::optimize::joint_optimization_problem<
          phylocore::ctmc_model,
          phylocore::gmm_observation_model,
          Eigen::VectorXd,
          std::monostate> {
public:
    // ── Model-level flags ──────────────────────────────────────────────────
    bool fix_Q          = false;  // estimate Q?
    bool fix_emission   = false;  // estimate μ, Σ?
    bool stationary_pi  = true;   // π is the stationary distribution of Q

    // ── Helpers ────────────────────────────────────────────────────────────

    std::size_t k_() const { return static_cast<std::size_t>(stm->Q.rows()); }
    std::size_t d_() const { return obs->d; }

    // Set π on stm->root_prior from a length-k vector. Clipped to a small
    // floor and renormalised so the log_root_prior in the inside pass stays
    // finite.
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

    // Return the stationary distribution of Q from the cached complex
    // eigendecomposition: pick the eigenvalue with smallest |λ|, take the
    // real part of the corresponding LEFT eigenvector (i.e. row of U_inv),
    // floor + normalise. Q already satisfies Q · 1 = 0, so the eigenvalue
    // near zero is the relevant one.
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
        // Left eigenvector for λ_best is row `best` of U_inv (since
        // U_inv · Q = diag(λ) · U_inv, i.e. (row best) Q = λ_best · (row best)).
        Eigen::VectorXd pi(k);
        for (std::size_t i = 0; i < k; ++i) {
            pi[static_cast<Eigen::Index>(i)] =
                Uinv(best, static_cast<Eigen::Index>(i)).real();
        }
        // Sign may be flipped (eigenvectors are determined up to scale).
        double s = 0.0;
        for (std::size_t i = 0; i < k; ++i)
            s += pi[static_cast<Eigen::Index>(i)];
        if (s < 0.0) pi = -pi;

        // Floor + renormalise.
        double total = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            const double v = pi[static_cast<Eigen::Index>(i)];
            pi[static_cast<Eigen::Index>(i)] = (v > 1e-12) ? v : 1e-12;
            total += pi[static_cast<Eigen::Index>(i)];
        }
        pi /= total;
        return pi;
    }

    // ── Parameter layout (get / set) ───────────────────────────────────────

    std::vector<double> get_params() const override {
        const std::size_t k = k_();
        const std::size_t d = d_();
        std::vector<double> p;

        if (!fix_Q) {
            // Q off-diagonals row-major, skipping the diagonal.
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
            // π_0..π_{k-2}, π_{k-1} derived.
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

        // ── Q ──────────────────────────────────────────────────────────────
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

        // ── μ, Σ ───────────────────────────────────────────────────────────
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
                // Symmetrise (numerical noise / asymmetric M-step) and
                // verify positive-definiteness via Cholesky.
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

        // ── π ──────────────────────────────────────────────────────────────
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

        // ── Refresh caches ─────────────────────────────────────────────────
        // ctmc_model::build() re-runs the eigendecomp + validates Q. When
        // stationary_pi is true and Q changed, refresh π too.
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
        obs->sites_observation_states.clear();  // continuous; not used by inside pass
        obs->build();
    }

    // ── EM Pattern-B duck-typed extension ──────────────────────────────────

    struct suff_stats_type {
        Eigen::VectorXd               responsibility_total;  // k
        std::vector<Eigen::VectorXd>  mu_numerator;          // k × d
        std::vector<Eigen::MatrixXd>  Sigma_outer;           // k × (d×d)
        Eigen::VectorXd               root_gamma_total;      // k (only if !stationary_pi)
        Eigen::MatrixXd               N_total;               // k × k (only if !fix_Q)
        Eigen::VectorXd               T_total;               // k    (only if !fix_Q)
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

        // Aggregate leaf responsibilities + (optionally) emission moments.
        // Also responsibility_total: the spec asks the validator to compare
        // this against total leaf count.
        for (std::size_t ti = 0; ti < trees.size(); ++ti) {
            const tree& t = *trees[ti];
            for (std::size_t v = 0; v < t.num_nodes; ++v) {
                if (t.tree.out_degree(static_cast<int>(v)) != 0) continue;  // leaves only
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

        // ── 1. μ ───────────────────────────────────────────────────────────
        std::vector<Eigen::VectorXd> mu_new = obs->mu;
        if (!fix_emission) {
            for (std::size_t i = 0; i < k; ++i) {
                const double N_i = ss.responsibility_total[i];
                if (N_i < 1e-12) continue;  // collapsed cluster — leave unchanged
                mu_new[i] = ss.mu_numerator[i] / N_i;
            }
        }

        // ── 2. Σ ───────────────────────────────────────────────────────────
        std::vector<Eigen::MatrixXd> Sigma_new = obs->Sigma;
        if (!fix_emission) {
            for (std::size_t i = 0; i < k; ++i) {
                const double N_i = ss.responsibility_total[i];
                if (N_i < 1e-12) continue;
                Eigen::MatrixXd S = ss.Sigma_outer[i] / N_i
                                  - mu_new[i] * mu_new[i].transpose();
                S = 0.5 * (S + S.transpose());
                // Ridge to keep PD even with extreme responsibilities.
                S += 1e-6 * Eigen::MatrixXd::Identity(d, d);
                Sigma_new[i] = S;
            }
        }

        // ── 3. Q ───────────────────────────────────────────────────────────
        // Start from the previous Q. When a cluster collapses (T_i ≈ 0), KEEP
        // the previous row for that cluster rather than zeroing it. Zeroing
        // produces a defective rank-deficient Q that destabilises the complex
        // eigendecomposition on subsequent iterations (this caused EM
        // divergence on some random seeds; matches the old phylo_gmm code).
        Eigen::MatrixXd Q_new = stm->Q;
        if (!fix_Q) {
            for (std::size_t i = 0; i < k; ++i) {
                const double T_i = ss.T_total[i];
                if (T_i < 1e-12) continue;  // collapsed cluster — keep prior row
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

        // ── Write everything back, then build() in the right order. ────────
        if (!fix_emission) {
            obs->mu    = std::move(mu_new);
            obs->Sigma = std::move(Sigma_new);
        }
        if (!fix_Q) {
            stm->Q = Q_new;
        }

        // ── 4. stm->build() (refresh eigendecomp / validate Q) ─────────────
        stm->build();

        // ── 5. π ───────────────────────────────────────────────────────────
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

        // ── 6. obs->build() (refresh Cholesky cache; validates Σ) ──────────
        obs->sites_latent_states = stm->sites_latent_states;
        obs->build();

        (void)num_sites;
    }

    // ── Initialisation: k-means++ and random ───────────────────────────────

    void initialize_kmeans_pp(uint64_t seed) {
        const std::size_t k = k_();
        const std::size_t d = obs->d;
        std::mt19937_64 rng(seed);

        const auto& Y = *observed_data;
        const std::size_t N = Y.size();
        if (N < k) {
            throw std::runtime_error(
                "initialize_kmeans_pp: not enough observations (" +
                std::to_string(N) + ") for k = " + std::to_string(k));
        }

        // 1. Pick first centre uniformly at random.
        std::uniform_int_distribution<std::size_t> uidx(0, N - 1);
        std::vector<Eigen::VectorXd> centres;
        centres.reserve(k);
        centres.push_back(Y[uidx(rng)]);

        // 2. Pick remaining centres weighted by squared distance.
        std::vector<double> sqd(N, phylocore::POSITIVE_INFINITY);  // finite sentinel; -ffast-math-safe
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

        // 3. Global sample covariance.
        Eigen::VectorXd y_mean = Eigen::VectorXd::Zero(d);
        for (const auto& y : Y) y_mean += y;
        y_mean /= static_cast<double>(N);
        Eigen::MatrixXd Sg = Eigen::MatrixXd::Zero(d, d);
        for (const auto& y : Y) {
            Eigen::VectorXd c = y - y_mean;
            Sg += c * c.transpose();
        }
        Sg /= static_cast<double>(N);

        // Setup model state.
        obs->d     = d;
        obs->mu    = std::move(centres);
        obs->Sigma.assign(k, Sg / static_cast<double>(k));
        // Ridge to guarantee PD.
        for (std::size_t i = 0; i < k; ++i)
            obs->Sigma[i] += 1e-3 * Eigen::MatrixXd::Identity(d, d);

        // Q: uniform off-diagonal (0.5), diagonal closes rows.
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

        // Sample k distinct indices.
        std::vector<std::size_t> idx(N);
        for (std::size_t i = 0; i < N; ++i) idx[i] = i;
        for (std::size_t i = 0; i < k; ++i) {
            std::uniform_int_distribution<std::size_t> u(i, N - 1);
            std::swap(idx[i], idx[u(rng)]);
        }
        std::vector<Eigen::VectorXd> centres;
        centres.reserve(k);
        for (std::size_t i = 0; i < k; ++i) centres.push_back(Y[idx[i]]);

        // Global sample covariance.
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
// Ground truth (matches tests/phylo_gmm/simulate.cxx).
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
// Permutation alignment between recovered and true labels.
// ─────────────────────────────────────────────────────────────────────────────

struct alignment_result {
    // perm[i] = true-label index that recovered cluster i corresponds to.
    // i.e. recovered cluster `i` is compared against gt.mu[perm[i]].
    std::vector<std::size_t> perm;
    double mean_mu_err;        // (1/k) Σ_i ||mu_rec[i] - gt.mu[perm[i]]||
    double pi_err;             // L1 distance after permuting truth.
};

// Search all k! permutations of {0,...,k-1} for the assignment of recovered
// cluster i → true cluster perm[i] that minimises the average Euclidean
// distance between the recovered mean mu_rec[i] and the true mean
// gt.mu[perm[i]]. Returns the best permutation and matching errors.
//
// NB: we deliberately do *not* initialise `best_score` with
// std::numeric_limits<double>::infinity(): the release build is compiled with
// -ffast-math (-> -ffinite-math-only), under which the compiler is free to
// assume no infinities and the comparison `s < best_score` becomes degenerate
// (best_score is effectively treated as 0, so no permutation ever beats the
// identity). Instead we seed best_score with the score of the identity
// permutation up-front.
alignment_result align_to_truth(const std::vector<Eigen::VectorXd>& mu_rec,
                                 const Eigen::VectorXd&              pi_rec,
                                 const ground_truth&                 gt) {
    const std::size_t k = gt.mu.size();

    auto score_perm = [&](const std::vector<std::size_t>& p) {
        double s = 0.0;
        for (std::size_t i = 0; i < k; ++i)
            s += (mu_rec[i] - gt.mu[p[i]]).norm();
        return s;
    };

    std::vector<std::size_t> perm(k);
    for (std::size_t i = 0; i < k; ++i) perm[i] = i;

    std::vector<std::size_t> best_perm = perm;
    double best_score = score_perm(perm);  // finite seed; -ffast-math safe
    while (std::next_permutation(perm.begin(), perm.end())) {
        double s = score_perm(perm);
        if (s < best_score) {
            best_score = s;
            best_perm  = perm;
        }
    }

    alignment_result a;
    a.perm = best_perm;
    a.mean_mu_err = best_score / static_cast<double>(k);

    double pe = 0.0;
    for (std::size_t i = 0; i < k; ++i)
        pe += std::abs(pi_rec[static_cast<Eigen::Index>(i)]
                       - gt.pi[static_cast<Eigen::Index>(best_perm[i])]);
    a.pi_err = pe;
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute log-L at the ground truth (for the validation threshold).
// ─────────────────────────────────────────────────────────────────────────────

double log_likelihood_at_truth(const ground_truth&          gt,
                                simulated_dataset&          ds) {
    ctmc_model stm;
    stm.Q = gt.Q;
    stm.build();
    // Use the GT π as the root prior.
    std::vector<double> pi_v(gt.pi.size());
    for (Eigen::Index i = 0; i < gt.pi.size(); ++i) pi_v[i] = gt.pi[i];
    stm.root_prior = { pi_v };

    gmm_observation_model obs;
    obs.d     = gt.mu[0].size();
    obs.mu    = gt.mu;
    obs.Sigma = gt.Sigma;
    obs.build();
    obs.sites_latent_states = stm.sites_latent_states;

    std::vector<std::vector<std::monostate>> node_data_storage(ds.trees.size());
    for (std::size_t i = 0; i < ds.trees.size(); ++i)
        node_data_storage[i].assign(ds.trees[i].num_nodes, std::monostate{});

    phylo_gmm_problem p;
    p.stm           = &stm;
    p.obs           = &obs;
    p.observed_data = &ds.observed_data;
    p.metadata      = &ds.metadata;
    p.trees.reserve(ds.trees.size());
    p.node_data.reserve(ds.trees.size());
    for (std::size_t i = 0; i < ds.trees.size(); ++i) {
        p.trees.push_back(&ds.trees[i]);
        p.node_data.push_back(&node_data_storage[i]);
    }
    return p.log_likelihood();
}

// ─────────────────────────────────────────────────────────────────────────────
// Restart records.
// ─────────────────────────────────────────────────────────────────────────────

struct restart_record {
    std::string                  label;
    double                       log_likelihood;
    bool                         converged;
    int                          iterations;
    std::vector<Eigen::VectorXd> mu;
    std::vector<Eigen::MatrixXd> Sigma;
    Eigen::MatrixXd              Q;
    Eigen::VectorXd              pi;
    double                       responsibility_sum;
};

restart_record run_em_restart(simulated_dataset& ds,
                              const std::string& label,
                              bool use_kmeans_pp,
                              uint64_t seed) {
    const std::size_t k = 3;
    const std::size_t d = 2;

    ctmc_model stm;
    gmm_observation_model obs;
    obs.d = d;

    // Seed Q so ctmc_model::build inside set_params doesn't get an empty Q.
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

    if (use_kmeans_pp) problem.initialize_kmeans_pp(seed);
    else               problem.initialize_random(seed);

    phylocore::optimize::em_options opts;
    opts.tol               = 1e-6;
    opts.max_iter          = 200;
    opts.verbose           = false;
    opts.warn_on_decrease  = false;  // closed-form M-step + complex eigendecomp
                                     // can produce tiny non-monotone steps

    auto results = phylocore::optimize::em(problem, opts);

    // Aggregate sanity stat: total responsibility across all leaves & sites.
    // For num_sites = 1 this should equal total leaf count.
    double total_resp = 0.0;
    {
        auto ss = problem.compute_suff_stats();
        for (Eigen::Index i = 0; i < ss.responsibility_total.size(); ++i)
            total_resp += ss.responsibility_total[i];
    }

    Eigen::VectorXd pi_rec(k);
    for (std::size_t i = 0; i < k; ++i)
        pi_rec[static_cast<Eigen::Index>(i)] = stm.root_prior[0][i];

    restart_record r;
    r.label              = label;
    r.log_likelihood     = results.log_likelihood;
    r.converged          = results.converged;
    r.iterations         = results.num_iterations;
    r.mu                 = obs.mu;
    r.Sigma              = obs.Sigma;
    r.Q                  = stm.Q;
    r.pi                 = pi_rec;
    r.responsibility_sum = total_resp;
    return r;
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

    std::cout << "── phyloGMM EM validation ──────────────────────────────────\n";
    std::cout << "trees           = " << ds.trees.size() << "\n";
    std::cout << "total leaves    = " << ds.observed_data.size() << "\n";

    // ── log-L at truth ─────────────────────────────────────────────────────
    const double log_L_truth = log_likelihood_at_truth(gt, ds);
    std::cout << "log L at truth  = " << log_L_truth << "\n\n";

    // ── 10 restarts ────────────────────────────────────────────────────────
    std::vector<restart_record> records;
    records.reserve(10);

    for (int s = 0; s < 5; ++s) {
        std::cout << "  kmeans++ restart " << s << " ... " << std::flush;
        auto r = run_em_restart(ds,
                                "kmeans++ #" + std::to_string(s),
                                /*use_kmeans_pp=*/true,
                                /*seed=*/1000u + s);
        std::cout << "log L = " << r.log_likelihood
                  << "  (iters = " << r.iterations
                  << ", converged = " << r.converged
                  << ", total_resp = " << r.responsibility_sum << ")\n";
        // TEMP per-restart param dump (for characterization; revert when done):
        std::cout << "    pi = " << r.pi.transpose() << "\n";
        for (size_t i = 0; i < r.mu.size(); ++i) {
            std::cout << "    mu[" << i << "] = " << r.mu[i].transpose() << "\n";
        }
        std::cout << "    Q =\n";
        for (int i = 0; i < r.Q.rows(); ++i) {
            std::cout << "      " << r.Q.row(i) << "\n";
        }
        records.push_back(std::move(r));
    }
    for (int s = 0; s < 5; ++s) {
        std::cout << "  random   restart " << s << " ... " << std::flush;
        auto r = run_em_restart(ds,
                                "random #" + std::to_string(s),
                                /*use_kmeans_pp=*/false,
                                /*seed=*/9000u + s);
        std::cout << "log L = " << r.log_likelihood
                  << "  (iters = " << r.iterations
                  << ", converged = " << r.converged
                  << ", total_resp = " << r.responsibility_sum << ")\n";
        // TEMP per-restart param dump (for characterization; revert when done):
        std::cout << "    pi = " << r.pi.transpose() << "\n";
        for (size_t i = 0; i < r.mu.size(); ++i) {
            std::cout << "    mu[" << i << "] = " << r.mu[i].transpose() << "\n";
        }
        std::cout << "    Q =\n";
        for (int i = 0; i < r.Q.rows(); ++i) {
            std::cout << "      " << r.Q.row(i) << "\n";
        }
        records.push_back(std::move(r));
    }

    // ── Best restart ───────────────────────────────────────────────────────
    const restart_record* best = &records[0];
    for (const auto& r : records)
        if (r.log_likelihood > best->log_likelihood) best = &r;

    std::cout << "\n── Best restart ────────────────────────────────────────────\n";
    std::cout << "label           = " << best->label << "\n";
    std::cout << "log L           = " << best->log_likelihood << "\n";
    std::cout << "iterations      = " << best->iterations << "\n";
    std::cout << "converged       = " << (best->converged ? "true" : "false") << "\n";
    std::cout << "total resp.     = " << best->responsibility_sum
              << "  (expected " << ds.observed_data.size() << ")\n";

    // ── Alignment ──────────────────────────────────────────────────────────
    // perm[i] = true cluster index that recovered cluster i corresponds to.
    // We re-order the *recovered* params so that aligned[t] is the recovered
    // parameter that pairs with the *true* params at index t. Build inv_perm
    // such that for each true index t, inv_perm[t] = recovered index i with
    // perm[i] = t.
    auto align = align_to_truth(best->mu, best->pi, gt);
    const std::size_t k = align.perm.size();
    std::vector<std::size_t> inv_perm(k);
    for (std::size_t i = 0; i < k; ++i) inv_perm[align.perm[i]] = i;

    std::cout << "\nbest permutation (recovered -> true): ";
    for (std::size_t i = 0; i < k; ++i) {
        std::cout << i << "->" << align.perm[i];
        if (i + 1 < k) std::cout << ", ";
    }
    std::cout << "\n";

    // Build aligned-to-truth views of recovered params.
    std::vector<Eigen::VectorXd> mu_aligned(k);
    std::vector<Eigen::MatrixXd> Sigma_aligned(k);
    Eigen::VectorXd pi_aligned(static_cast<Eigen::Index>(k));
    Eigen::MatrixXd Q_aligned(static_cast<Eigen::Index>(k),
                              static_cast<Eigen::Index>(k));
    for (std::size_t t = 0; t < k; ++t) {
        const std::size_t r = inv_perm[t];        // recovered index for true t
        mu_aligned[t]    = best->mu[r];
        Sigma_aligned[t] = best->Sigma[r];
        pi_aligned[static_cast<Eigen::Index>(t)] =
            best->pi[static_cast<Eigen::Index>(r)];
    }
    for (std::size_t t1 = 0; t1 < k; ++t1)
        for (std::size_t t2 = 0; t2 < k; ++t2)
            Q_aligned(static_cast<Eigen::Index>(t1),
                      static_cast<Eigen::Index>(t2)) =
                best->Q(static_cast<Eigen::Index>(inv_perm[t1]),
                        static_cast<Eigen::Index>(inv_perm[t2]));

    std::cout << "\nRecovered μ_i (aligned with true μ_i):\n";
    for (std::size_t t = 0; t < k; ++t) {
        std::cout << "  μ_" << t
                  << ":  recovered = (" << mu_aligned[t].transpose() << ")"
                  << "     true = (" << gt.mu[t].transpose() << ")\n";
    }

    std::cout << "\nRecovered π (aligned) = " << pi_aligned.transpose() << "\n";
    std::cout <<   "True       π          = " << gt.pi.transpose() << "\n";

    std::cout << "\nRecovered Σ_i (aligned) vs true Σ_i:\n";
    for (std::size_t t = 0; t < k; ++t) {
        std::cout << "  Σ_" << t << " recovered =\n" << Sigma_aligned[t] << "\n";
        std::cout << "  Σ_" << t << " true      =\n" << gt.Sigma[t] << "\n";
    }

    std::cout << "\nRecovered Q (aligned) =\n" << Q_aligned << "\n";
    std::cout <<   "True       Q          =\n" << gt.Q << "\n";

    // ── Diff metrics (consistent with alignment direction) ─────────────────
    // μ: mean Euclidean distance per cluster.
    double mu_avg = 0.0;
    for (std::size_t t = 0; t < k; ++t)
        mu_avg += (mu_aligned[t] - gt.mu[t]).norm();
    mu_avg /= static_cast<double>(k);

    // Σ: max Frobenius norm over clusters.
    double sigma_max_frob = 0.0;
    for (std::size_t t = 0; t < k; ++t) {
        double f = (Sigma_aligned[t] - gt.Sigma[t]).norm();  // Frobenius
        if (f > sigma_max_frob) sigma_max_frob = f;
    }

    // π: L1 distance.
    double pi_l1 = (pi_aligned - gt.pi).lpNorm<1>();

    // Q: Frobenius distance over the full permuted matrix.
    double Q_frob = (Q_aligned - gt.Q).norm();

    std::cout << "\n||μ_rec - μ_true||_avg       = " << mu_avg << "\n";
    std::cout <<   "||Σ_rec - Σ_true||_max_Frob  = " << sigma_max_frob << "\n";
    std::cout <<   "||π_rec - π_true||_L1        = " << pi_l1 << "\n";
    std::cout <<   "||Q_rec - Q_true||_Frob      = " << Q_frob << "\n";

    // ── Validation ─────────────────────────────────────────────────────────
    const double threshold = log_L_truth - 5.0;
    std::cout << "\nthreshold       = log_L_truth - 5.0 = " << threshold << "\n";
    std::cout << "best log L      = " << best->log_likelihood << "\n";
    if (best->log_likelihood >= threshold) {
        std::cout << "PASS\n";
        return 0;
    } else {
        std::cout << "FAIL (best log L < threshold)\n";
        return 1;
    }
}
