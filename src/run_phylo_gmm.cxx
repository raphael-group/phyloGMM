// run_phylo_gmm: phyloGMM driver. One binary, four modes.
//
// Modes (mutually exclusive; default is full EM):
//
//   default          full joint EM over Q, μ, Σ, π
//   --fix_q <rate>   EM on (μ, Σ, π) only; Q is uniform-exit-rate Q,
//                    Q[i,j] = rate/(k-1), Q[i,i] = -rate
//   --fix_emission   EM on Q only; μ, Σ, π stay at the kmeans++ init values
//   --fix_labels <csv>  EM on Q only; leaf state assignments clamped to the
//                       CSV. Implies --fix_emission (μ, Σ also frozen).
//                       Labels CSV: "cell,state" header; state is 0-indexed.
//   --bayes_opt      No EM. True (Q, μ, Σ, π) are constructed from
//                    --true_q_exit_rate <r> and --true_simplex_scale <s>:
//                       Q = uniform-exit-rate(r)
//                       μ = scale × SIMPLEX_4   (k=4, d=3 only)
//                       Σ = I per cluster
//                       π = 1/k
//                    A single inside–outside pass yields the Bayes-optimal
//                    posteriors at the true parameters.
//                    (TODO: generalize true params to arbitrary CSV inputs.)
//
// CLI (legacy positional, full-EM only):
//   run_phylo_gmm <tree_file> <obs_csv> <k> \
//       [max_iter] [tol] [q_init_rate] [fix_pi] [n_init]
//
// CLI (flag form — required for any non-default mode and for multi-tree):
//   run_phylo_gmm --k K [--max_iter N] [--tol T] [--q_init_rate R] \
//                 [--fix_pi 0|1] [--n_init N] \
//                 [--ctmc_weight W_C] [--emission_weight W_E] \
//                 [--kmeans_init seeding|standard] \
//                 [--init_params_json params.json] \
//                 --tree T1 --obs O1 [--tree T2 --obs O2 ...]            \
//                 [--fix_q <rate> | --fix_emission | --fix_labels <csv>  \
//                  | --bayes_opt --true_q_exit_rate <r>                  \
//                                --true_simplex_scale <s>]
//
// Conventions:
//   * obs CSV: first column = taxon/cell name, remaining columns = y_0..y_{d-1}
//   * tree:    single Newick per first non-empty line of <tree_file>
//   * default EM: 2 * n_init restarts total: n_init k-means++ (seeds 42..)
//     plus n_init random (seeds 1000..). --kmeans_init standard adds Lloyd
//     refinement after k-means++ seeding. Deterministic seeds keep per-restart
//     outputs reproducible across runs.
//
// Output (JSON to stdout): unified across modes. Mode flags echo into the
// JSON under "mode" so callers can verify how a run was parameterized.
//   { "log_likelihood", "k", "d", "n_observed", "n_leaves", "n_trees",
//     "Q", "mu", "Sigma", "pi", "leaf_names", "posteriors", "leaf_labels",
//     "internal_node_posteriors", "internal_node_ids",
//     "expected_transitions", "expected_dwell_times", "per_restart",
//     "per_restart[].log_likelihood_trace",
//     "per_restart[].complete_data_objective_trace",
//     "mode": { "fix_q_rate": <r>?, "fix_emission": <b>?,
//               "ctmc_weight": <w_C>, "emission_weight": <w_E>,
//               "fix_labels_csv": "<path>"?, "bayes_opt": <b>?,
//               "true_q_exit_rate": <r>?, "true_simplex_scale": <s>?,
//               "init_params_json": "<path>"? } }

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
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
// Shared helpers (visible to both the anon-namespace EM machinery and the
// outer-scope main_impl).
// ─────────────────────────────────────────────────────────────────────────────

// Build the uniform-exit-rate Q matrix: Q[i,j] = rate/(k-1) for i!=j,
// Q[i,i] = -rate. Stationary distribution is uniform (1/k, …, 1/k).
static Eigen::MatrixXd make_uniform_exit_q(std::size_t k, double rate) {
    Eigen::MatrixXd Q(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(k));
    const double offdiag = (k > 1) ? rate / static_cast<double>(k - 1) : 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            Q(static_cast<Eigen::Index>(i),
              static_cast<Eigen::Index>(j)) = (i == j) ? -rate : offdiag;
        }
    }
    return Q;
}

// Hardcoded regular-tetrahedron vertices in R^3 with centroid at origin and
// unit edge length. Matches the SIMPLEX_4 constant in the legacy bayes_opt
// driver and in sim/params.py. Only used by --bayes_opt mode.
static constexpr double SIMPLEX_4[4][3] = {
    {-0.612,  0.000,  0.000},
    { 0.204, -0.577,  0.000},
    { 0.204,  0.289, -0.500},
    { 0.204,  0.289,  0.500},
};

// ─────────────────────────────────────────────────────────────────────────────
// Tiny JSON loader (sufficient for --bayes_opt_truth_json / --fix_q_json /
// --init_params_json).
// ─────────────────────────────────────────────────────────────────────────────
//
// Recognizes a flat object with numeric scalars and 1D/2D/3D numeric arrays.
// Strings, nested objects, bools, and deeper arrays are skipped for unknown
// keys. Suitable only for the parameter schema documented in the CLI help
// text. Throws std::runtime_error on any malformed input.
//
// Schema accepted (extra keys ignored):
//   { "Q": [[K x K]], "mu": [[K x d]],
//     "Sigma": [[[d x d], ... K]], "sigma_diag": [[K x d]], "pi": [K] }
// "Sigma" is preferred for full covariance initialization; "sigma_diag" is a
// compact diagonal-covariance spelling and remains compatible with older
// --init_emission_json files.
struct truth_json {
    std::vector<std::vector<double>>              Q;          // K × K
    std::vector<std::vector<double>>              mu;         // K × d
    std::vector<std::vector<std::vector<double>>> Sigma;      // K × d × d
    std::vector<std::vector<double>>              sigma_diag; // K × d
    std::vector<double>                           pi;         // K (optional)
    bool                                          has_pi = false;
};

static truth_json load_truth_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open JSON: " + path);
    std::string src((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
    // Single-pass char cursor with whitespace skipping.
    std::size_t p = 0;
    auto skip_ws = [&]() {
        while (p < src.size() &&
               (src[p] == ' ' || src[p] == '\t' || src[p] == '\n' ||
                src[p] == '\r')) ++p;
    };
    auto expect = [&](char c) {
        skip_ws();
        if (p >= src.size() || src[p] != c)
            throw std::runtime_error(
                std::string("JSON parse error: expected '") + c +
                "' at offset " + std::to_string(p));
        ++p;
    };
    auto parse_string = [&]() -> std::string {
        skip_ws();
        if (p >= src.size() || src[p] != '"')
            throw std::runtime_error("JSON parse error: expected string");
        ++p;
        std::string out;
        while (p < src.size() && src[p] != '"') { out.push_back(src[p++]); }
        if (p >= src.size())
            throw std::runtime_error("JSON parse error: unterminated string");
        ++p;
        return out;
    };
    auto parse_number = [&]() -> double {
        skip_ws();
        std::size_t end = p;
        while (end < src.size() &&
               (std::isdigit(static_cast<unsigned char>(src[end])) ||
                src[end] == '-' || src[end] == '+' || src[end] == '.' ||
                src[end] == 'e' || src[end] == 'E')) ++end;
        if (end == p)
            throw std::runtime_error("JSON parse error: expected number");
        double v = std::stod(src.substr(p, end - p));
        p = end;
        return v;
    };
    // Parse a 1D array of numbers (caller positioned at '[').
    auto parse_num_array = [&]() -> std::vector<double> {
        expect('[');
        std::vector<double> out;
        skip_ws();
        if (p < src.size() && src[p] == ']') { ++p; return out; }
        while (true) {
            out.push_back(parse_number());
            skip_ws();
            if (p < src.size() && src[p] == ',') { ++p; continue; }
            break;
        }
        expect(']');
        return out;
    };
    // Parse a 2D array of numbers (caller positioned at outer '[').
    auto parse_num_2d = [&]() -> std::vector<std::vector<double>> {
        expect('[');
        std::vector<std::vector<double>> out;
        skip_ws();
        if (p < src.size() && src[p] == ']') { ++p; return out; }
        while (true) {
            out.push_back(parse_num_array());
            skip_ws();
            if (p < src.size() && src[p] == ',') { ++p; continue; }
            break;
        }
        expect(']');
        return out;
    };
    // Parse a 3D array of numbers (caller positioned at outer '[').
    auto parse_num_3d = [&]() -> std::vector<std::vector<std::vector<double>>> {
        expect('[');
        std::vector<std::vector<std::vector<double>>> out;
        skip_ws();
        if (p < src.size() && src[p] == ']') { ++p; return out; }
        while (true) {
            out.push_back(parse_num_2d());
            skip_ws();
            if (p < src.size() && src[p] == ',') { ++p; continue; }
            break;
        }
        expect(']');
        return out;
    };
    // Skip an arbitrary JSON value (used for tolerated extra keys at the top
    // level). Supports nested objects/arrays so the skip is balanced, but the
    // contents are discarded.
    std::function<void()> skip_value = [&]() {
        skip_ws();
        if (p >= src.size())
            throw std::runtime_error("JSON parse error: unexpected EOF");
        char c = src[p];
        if (c == '"') { (void)parse_string(); return; }
        if (c == '{' || c == '[') {
            char open = c, close = (c == '{') ? '}' : ']';
            int depth = 0;
            bool in_str = false;
            while (p < src.size()) {
                char x = src[p++];
                if (in_str) { if (x == '"') in_str = false; continue; }
                if (x == '"') { in_str = true; continue; }
                if (x == open) ++depth;
                else if (x == close) { --depth; if (depth == 0) return; }
            }
            throw std::runtime_error("JSON parse error: unbalanced container");
        }
        // number / true / false / null — read until delimiter.
        while (p < src.size() && src[p] != ',' && src[p] != '}' &&
               src[p] != ']') ++p;
    };

    truth_json out;
    skip_ws();
    expect('{');
    skip_ws();
    if (p < src.size() && src[p] == '}') { ++p; return out; }
    while (true) {
        std::string key = parse_string();
        expect(':');
        if (key == "Q") {
            out.Q = parse_num_2d();
        } else if (key == "mu") {
            out.mu = parse_num_2d();
        } else if (key == "Sigma" || key == "sigma") {
            out.Sigma = parse_num_3d();
        } else if (key == "sigma_diag") {
            out.sigma_diag = parse_num_2d();
        } else if (key == "pi") {
            out.pi = parse_num_array();
            out.has_pi = true;
        } else {
            skip_value();
        }
        skip_ws();
        if (p < src.size() && src[p] == ',') { ++p; continue; }
        break;
    }
    expect('}');
    return out;
}

// Convert a K-row × C-col std::vector<std::vector<double>> to Eigen, with
// shape validation. label is used in error messages.
static Eigen::MatrixXd vec2d_to_matrix(const std::vector<std::vector<double>>& v,
                                       std::size_t rows, std::size_t cols,
                                       const char* label) {
    if (v.size() != rows)
        throw std::runtime_error(std::string(label) + ": expected " +
            std::to_string(rows) + " rows, got " + std::to_string(v.size()));
    Eigen::MatrixXd M(static_cast<Eigen::Index>(rows),
                      static_cast<Eigen::Index>(cols));
    for (std::size_t i = 0; i < rows; ++i) {
        if (v[i].size() != cols)
            throw std::runtime_error(std::string(label) + ": row " +
                std::to_string(i) + " has " + std::to_string(v[i].size()) +
                " cols, expected " + std::to_string(cols));
        for (std::size_t j = 0; j < cols; ++j)
            M(static_cast<Eigen::Index>(i),
              static_cast<Eigen::Index>(j)) = v[i][j];
    }
    return M;
}

static std::vector<Eigen::VectorXd>
vec2d_to_vector_list(const std::vector<std::vector<double>>& v,
                     std::size_t rows,
                     std::size_t cols,
                     const char* label) {
    const Eigen::MatrixXd M = vec2d_to_matrix(v, rows, cols, label);
    std::vector<Eigen::VectorXd> out(rows, Eigen::VectorXd::Zero(cols));
    for (std::size_t i = 0; i < rows; ++i)
        out[i] = M.row(static_cast<Eigen::Index>(i)).transpose();
    return out;
}

static std::vector<Eigen::MatrixXd>
sigma_from_json(const truth_json& tj,
                std::size_t       k,
                std::size_t       d,
                bool              diagonal_covariance,
                double            diag_ridge,
                const char*       context) {
    std::vector<Eigen::MatrixXd> out(k, Eigen::MatrixXd::Zero(d, d));

    if (!tj.Sigma.empty()) {
        if (tj.Sigma.size() != k)
            throw std::runtime_error(
                std::string(context) + ": Sigma expected " +
                std::to_string(k) + " matrices, got " +
                std::to_string(tj.Sigma.size()));
        for (std::size_t i = 0; i < k; ++i) {
            const std::string label =
                std::string(context) + ": Sigma[" + std::to_string(i) + "]";
            Eigen::MatrixXd S = vec2d_to_matrix(
                tj.Sigma[i], d, d, label.c_str());

            for (Eigen::Index r = 0; r < S.rows(); ++r) {
                for (Eigen::Index c = 0; c < S.cols(); ++c) {
                    if (!std::isfinite(S(r, c)))
                        throw std::runtime_error(label + " contains non-finite entries");
                }
            }

            if (diagonal_covariance) {
                Eigen::MatrixXd D = Eigen::MatrixXd::Zero(d, d);
                for (std::size_t j = 0; j < d; ++j) {
                    const double s = S(static_cast<Eigen::Index>(j),
                                       static_cast<Eigen::Index>(j));
                    if (s <= 0.0)
                        throw std::runtime_error(
                            label + " diagonal entries must be > 0");
                    D(static_cast<Eigen::Index>(j),
                      static_cast<Eigen::Index>(j)) = s;
                }
                out[i] = D;
            } else {
                const double asym = (S - S.transpose()).cwiseAbs().maxCoeff();
                const double tol = 1e-8;
                if (asym > tol)
                    throw std::runtime_error(label + " must be symmetric");
                S = 0.5 * (S + S.transpose());
                Eigen::LLT<Eigen::MatrixXd> llt(S);
                if (llt.info() != Eigen::Success)
                    throw std::runtime_error(label + " must be positive definite");
                out[i] = std::move(S);
            }
        }
        return out;
    }

    if (tj.sigma_diag.empty())
        throw std::runtime_error(
            std::string(context) +
            ": required key 'Sigma' or 'sigma_diag' missing or empty");

    const Eigen::MatrixXd sd_mat =
        vec2d_to_matrix(tj.sigma_diag, k, d,
                        (std::string(context) + ": sigma_diag").c_str());
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            const double s = sd_mat(static_cast<Eigen::Index>(i),
                                    static_cast<Eigen::Index>(j));
            if (!std::isfinite(s) || s <= 0.0)
                throw std::runtime_error(
                    std::string(context) +
                    ": sigma_diag entries must be > 0 (cluster " +
                    std::to_string(i) + ", dim " + std::to_string(j) + ")");
            out[i](static_cast<Eigen::Index>(j),
                   static_cast<Eigen::Index>(j)) = s + diag_ridge;
        }
    }
    return out;
}

static void validate_ctmc_q(Eigen::MatrixXd& Q, const char* label) {
    if (Q.rows() != Q.cols())
        throw std::runtime_error(
            std::string(label) + ": Q must be square");
    const Eigen::Index k = Q.rows();
    const double tol = 1e-8;
    for (Eigen::Index i = 0; i < k; ++i) {
        double off_sum = 0.0;
        for (Eigen::Index j = 0; j < k; ++j) {
            const double v = Q(i, j);
            if (!std::isfinite(v))
                throw std::runtime_error(
                    std::string(label) + ": Q contains non-finite entries");
            if (i == j) continue;
            if (v < -tol)
                throw std::runtime_error(
                    std::string(label) + ": Q off-diagonal entries must be >= 0");
            if (v < 0.0) Q(i, j) = 0.0;
            off_sum += Q(i, j);
        }
        if (Q(i, i) > tol)
            throw std::runtime_error(
                std::string(label) + ": Q diagonal entries must be <= 0");
        const double row_sum = Q.row(i).sum();
        const double scaled_tol = tol * std::max(1.0, off_sum);
        if (std::abs(row_sum) > scaled_tol)
            throw std::runtime_error(
                std::string(label) +
                ": each Q row must sum to zero; row " +
                std::to_string(i) + " sums to " + std::to_string(row_sum));
        Q(i, i) = -off_sum;
    }
}

static Eigen::VectorXd normalized_pi_from_json(const truth_json& tj,
                                               std::size_t       k,
                                               const char*       context) {
    if (!tj.has_pi)
        return Eigen::VectorXd::Constant(static_cast<Eigen::Index>(k),
                                         1.0 / static_cast<double>(k));
    if (tj.pi.size() != k)
        throw std::runtime_error(
            std::string(context) + ": pi expected " + std::to_string(k) +
            " entries, got " + std::to_string(tj.pi.size()));
    Eigen::VectorXd pi(static_cast<Eigen::Index>(k));
    double sum = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        const double v = tj.pi[i];
        if (!std::isfinite(v) || v < 0.0)
            throw std::runtime_error(
                std::string(context) + ": pi entries must be finite and >= 0");
        pi[static_cast<Eigen::Index>(i)] = v;
        sum += v;
    }
    if (sum <= 0.0)
        throw std::runtime_error(
            std::string(context) + ": pi must have positive total mass");
    pi /= sum;
    return pi;
}

// ─────────────────────────────────────────────────────────────────────────────
// phylo_gmm_problem — defined inline in this driver (anonymous namespace) so
// no public header is needed. The unit tests in tests/phylo_gmm/ each carry
// their own local copy of the same class for the same reason.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

constexpr double expected_transition_tol = 1e-10;

// Weighted EM target:
//   Q_w = Q_root + w_C Q_CTMC + w_E Q_emission.
// The E-step uses the corresponding powered factors. Gaussian leaf log
// emissions are multiplied by w_E. Integrating a powered CTMC path density
// along a branch gives exp(tA), A_ii=w_C q_ii and A_ij=q_ij^w_C; ctmc_model
// supplies that kernel and Hobolth-Jensen uses the same operator for E[N,T].
// The positive scalar weights cancel from each block's closed-form M-step,
// but they change its sufficient statistics through the weighted E-step.

class phylo_gmm_problem
    : public phylocore::optimize::joint_optimization_problem<
          phylocore::ctmc_model,
          phylocore::gmm_observation_model,
          Eigen::VectorXd,
          std::monostate> {
public:
    bool fix_Q         = false;
    bool fix_emission  = false;
    bool stationary_pi = false;  // default: pi is a free EM parameter (= root posterior)
    double ctmc_weight = 1.0;
    double emission_weight = 1.0;

    double emission_log_weight() const override { return emission_weight; }

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
        const Eigen::VectorXcd& lambda = stm->rate_eigenvalues_q();
        const Eigen::MatrixXcd& Uinv   = stm->inverse_rate_eigenvectors_q();
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
            if (obs->diagonal_covariance) {
                for (std::size_t i = 0; i < k; ++i)
                    for (std::size_t a = 0; a < d; ++a)
                        p.push_back(obs->Sigma[i](a, a));
            } else {
                for (std::size_t i = 0; i < k; ++i)
                    for (std::size_t r = 0; r < d; ++r)
                        for (std::size_t c = 0; c < d; ++c)
                            p.push_back(obs->Sigma[i](r, c));
            }
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
        if (!fix_emission) {
            expected += k * d;
            expected += obs->diagonal_covariance ? k * d : k * d * d;
        }
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
                if (obs->diagonal_covariance) {
                    for (std::size_t a = 0; a < d; ++a) {
                        const double v = params[idx++];
                        if (v <= 0.0) {
                            throw phylocore::optimize::invalid_parameters_error(
                                "Sigma diagonal <= 0", idx - 1);
                        }
                        obs->Sigma[i](a, a) = v;
                    }
                } else {
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
        std::vector<Eigen::VectorXd>  Sigma_diag_outer;
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
        ss.mu_numerator.assign(k, Eigen::VectorXd::Zero(d));
        if (obs->diagonal_covariance) {
            ss.Sigma_diag_outer.assign(k, Eigen::VectorXd::Zero(d));
        } else {
            ss.Sigma_outer.assign(k, Eigen::MatrixXd::Zero(d, d));
        }
        ss.root_gamma_total = Eigen::VectorXd::Zero(k);

        for (std::size_t ti = 0; ti < trees.size(); ++ti) {
            const tree& t = *trees[ti];
            for (std::size_t v = 0; v < t.num_nodes; ++v) {
                if (t.tree.out_degree(static_cast<int>(v)) != 0) continue;
                const std::size_t taxa_id = t.tree[static_cast<int>(v)].data;
                const Eigen::VectorXd& y  = (*observed_data)[taxa_id];
                Eigen::VectorXd y_sq;
                Eigen::MatrixXd yyT;
                if (obs->diagonal_covariance) {
                    y_sq = y.array().square().matrix();
                } else {
                    yyT = y * y.transpose();
                }

                for (std::size_t c = 0; c < num_sites; ++c) {
                    auto log_gamma = leaf_posterior(ti, v, c);
                    for (std::size_t i = 0; i < k; ++i) {
                        const double r = std::exp(log_gamma[i]);
                        ss.responsibility_total[i] += r;
                        ss.mu_numerator[i].noalias() += r * y;
                        if (obs->diagonal_covariance) {
                            ss.Sigma_diag_outer[i].noalias() += r * y_sq;
                        } else {
                            ss.Sigma_outer[i].noalias() += r * yyT;
                        }
                    }
                }
            }
            for (std::size_t c = 0; c < num_sites; ++c) {
                auto log_gamma = root_posterior(ti, c);
                for (std::size_t i = 0; i < k; ++i)
                    ss.root_gamma_total[i] += std::exp(log_gamma[i]);
            }
        }
        auto hj = phylocore::optimize::hobolth_jensen_stats(*this);
        ss.N_total = hj.N_total;
        ss.T_total = hj.T_total;
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
                if (obs->diagonal_covariance) {
                    Eigen::VectorXd diag =
                        ss.Sigma_diag_outer[i] / N_i
                        - mu_new[i].array().square().matrix();
                    for (std::size_t a = 0; a < d; ++a) {
                        if (diag[static_cast<Eigen::Index>(a)] < 0.0)
                            diag[static_cast<Eigen::Index>(a)] = 0.0;
                        diag[static_cast<Eigen::Index>(a)] += 1e-6;
                    }
                    Sigma_new[i] = diag.asDiagonal();
                } else {
                    Eigen::MatrixXd S = ss.Sigma_outer[i] / N_i
                                      - mu_new[i] * mu_new[i].transpose();
                    S = 0.5 * (S + S.transpose());
                    S += 1e-6 * Eigen::MatrixXd::Identity(d, d);
                    Sigma_new[i] = S;
                }
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
	                    const double raw_Nij = ss.N_total(i, j);
	                    const double Nij =
	                        (raw_Nij > expected_transition_tol) ? raw_Nij : 0.0;
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

    std::vector<Eigen::VectorXd> kmeans_pp_seed_centres_(uint64_t seed) const {
        const std::size_t k = k_();
        std::mt19937_64 rng(seed);
        const auto& Y = *observed_data;
        const std::size_t N = Y.size();
        if (N < k) throw std::runtime_error("kmeans_pp_seed_centres_: N < k");

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
        return centres;
    }

    void refine_centres_lloyd_(std::vector<Eigen::VectorXd>& centres,
                               int max_iter,
                               double tol) const {
        const std::size_t k = centres.size();
        const std::size_t d = obs->d;
        const auto& Y = *observed_data;
        const std::size_t N = Y.size();
        if (max_iter <= 0 || k == 0 || N == 0) return;

        std::vector<std::size_t> assignment(N, k);
        for (int iter = 0; iter < max_iter; ++iter) {
            std::vector<Eigen::VectorXd> sums(k, Eigen::VectorXd::Zero(d));
            std::vector<std::size_t> counts(k, 0);
            bool changed = false;

            for (std::size_t n = 0; n < N; ++n) {
                std::size_t best = 0;
                double best_dist = (Y[n] - centres[0]).squaredNorm();
                for (std::size_t i = 1; i < k; ++i) {
                    const double dist = (Y[n] - centres[i]).squaredNorm();
                    if (dist < best_dist) {
                        best_dist = dist;
                        best = i;
                    }
                }
                if (assignment[n] != best) {
                    assignment[n] = best;
                    changed = true;
                }
                sums[best].noalias() += Y[n];
                ++counts[best];
            }

            double shift = 0.0;
            for (std::size_t i = 0; i < k; ++i) {
                if (counts[i] == 0) continue;
                Eigen::VectorXd next = sums[i] / static_cast<double>(counts[i]);
                shift += (next - centres[i]).squaredNorm();
                centres[i] = std::move(next);
            }

            if (!changed || std::sqrt(shift) <= tol) break;
        }
    }

    void initialize_kmeans_pp(uint64_t seed,
                              bool run_lloyd_refinement,
                              int lloyd_max_iter,
                              double lloyd_tol) {
        const std::size_t k = k_();
        const std::size_t d = obs->d;
        const auto& Y = *observed_data;
        const std::size_t N = Y.size();
        if (N < k) throw std::runtime_error("initialize_kmeans_pp: N < k");

        std::vector<Eigen::VectorXd> centres = kmeans_pp_seed_centres_(seed);
        if (run_lloyd_refinement) {
            refine_centres_lloyd_(centres, lloyd_max_iter, lloyd_tol);
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
        if (obs->diagonal_covariance) {
            const Eigen::VectorXd diag =
                Sg.diagonal() / static_cast<double>(k)
                + 1e-3 * Eigen::VectorXd::Ones(d);
            Eigen::MatrixXd S0 = diag.asDiagonal();
            obs->Sigma.assign(k, S0);
        } else {
            obs->Sigma.assign(k, Sg / static_cast<double>(k));
            for (std::size_t i = 0; i < k; ++i)
                obs->Sigma[i] += 1e-3 * Eigen::MatrixXd::Identity(d, d);
        }

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
        if (obs->diagonal_covariance) {
            const Eigen::VectorXd diag =
                Sg.diagonal() / static_cast<double>(k)
                + 1e-3 * Eigen::VectorXd::Ones(d);
            Eigen::MatrixXd S0 = diag.asDiagonal();
            obs->Sigma.assign(k, S0);
        } else {
            obs->Sigma.assign(k, Sg / static_cast<double>(k));
            for (std::size_t i = 0; i < k; ++i)
                obs->Sigma[i] += 1e-3 * Eigen::MatrixXd::Identity(d, d);
        }

        initialize_q_from_rate_();
        stm->root_prior = { std::vector<double>(k, 1.0 / static_cast<double>(k)) };
        stm->build();
        if (stationary_pi) set_pi_on_root_prior_(extract_stationary_pi_());
        obs->sites_latent_states = stm->sites_latent_states;
        obs->build();
    }
};

struct complete_data_objective_record {
    int    iteration = 0;
    double root      = 0.0;
    double ctmc      = 0.0;
    double emission  = 0.0;
    double weighted_ctmc     = 0.0;
    double weighted_emission = 0.0;
    double total     = 0.0;
};

struct phylo_gmm_em_output {
    phylocore::optimize::estimation_results results;
    std::vector<complete_data_objective_record> complete_data_objective_trace;
};

double weighted_log_sum(const Eigen::VectorXd& weights,
                        const std::vector<double>& probs) {
    double total = 0.0;
    for (Eigen::Index i = 0; i < weights.size(); ++i) {
        const double w = weights[i];
        if (w == 0.0) continue;
        const double p = probs[static_cast<std::size_t>(i)];
        if (p <= 0.0) return -std::numeric_limits<double>::infinity();
        total += w * std::log(p);
    }
    return total;
}

double complete_root_objective(
    const phylo_gmm_problem& problem,
    const phylo_gmm_problem::suff_stats_type& ss)
{
    return weighted_log_sum(ss.root_gamma_total, problem.stm->root_prior[0]);
}

double complete_ctmc_objective(
    const phylo_gmm_problem& problem,
    const phylo_gmm_problem::suff_stats_type& ss)
{
    const std::size_t k = problem.k_();
    double jump = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            if (i == j) continue;
	            const double Nij = ss.N_total(
	                static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
	            if (std::abs(Nij) <= expected_transition_tol) continue;
	            if (Nij < 0.0) continue;
	            const double qij = problem.stm->Q(
	                static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
            if (qij <= 0.0) return -std::numeric_limits<double>::infinity();
            jump += Nij * std::log(qij);
        }
    }

    double dwell = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        dwell += ss.T_total[static_cast<Eigen::Index>(i)]
               * problem.stm->Q(static_cast<Eigen::Index>(i),
                                static_cast<Eigen::Index>(i));
    }
    return jump + dwell;
}

double complete_emission_objective(phylo_gmm_problem& problem) {
    const std::size_t num_sites = problem.stm->sites_latent_states.size();
    double total = 0.0;

    for (std::size_t ti = 0; ti < problem.trees.size(); ++ti) {
        const tree& t = *problem.trees[ti];
        for (std::size_t v = 0; v < t.num_nodes; ++v) {
            if (t.tree.out_degree(static_cast<int>(v)) != 0) continue;
            const std::size_t taxa_id = t.tree[static_cast<int>(v)].data;
            const Eigen::VectorXd& y = (*problem.observed_data)[taxa_id];
            const int hard_label =
                (!problem.metadata->hard_labels.empty()
                 && taxa_id < problem.metadata->hard_labels.size())
                    ? problem.metadata->hard_labels[taxa_id]
                    : -1;

            for (std::size_t c = 0; c < num_sites; ++c) {
                const std::size_t alpha =
                    problem.stm->sites_latent_states[c].alphabet_size;
                auto log_gamma = problem.leaf_posterior(ti, v, c);
                for (std::size_t i = 0; i < alpha; ++i) {
                    const double r = std::exp(log_gamma[i]);
                    if (r == 0.0) continue;
                    if (hard_label >= 0) {
                        if (static_cast<int>(i) != hard_label)
                            return -std::numeric_limits<double>::infinity();
                        continue;  // log delta(label) = 0
                    }
                    total += r * problem.obs->compute_log_emission_probability(
                        (*problem.node_data[ti])[v], i, c, y);
                }
            }
        }
    }
    return total;
}

complete_data_objective_record compute_complete_data_objective_record(
    phylo_gmm_problem& problem,
    const phylo_gmm_problem::suff_stats_type& ss,
    int iteration)
{
    complete_data_objective_record rec;
    rec.iteration = iteration;
    rec.root      = complete_root_objective(problem, ss);
    rec.ctmc      = complete_ctmc_objective(problem, ss);
    rec.emission  = complete_emission_objective(problem);
    rec.weighted_ctmc = problem.ctmc_weight * rec.ctmc;
    rec.weighted_emission = problem.emission_weight * rec.emission;
    rec.total = rec.root + rec.weighted_ctmc + rec.weighted_emission;
    return rec;
}

phylo_gmm_em_output run_phylo_gmm_em(
    phylo_gmm_problem& problem,
    phylocore::optimize::em_options opts)
{
    problem.validate_disjoint_taxa();

    double prev_llh = problem.log_likelihood();
    std::vector<double> log_likelihood_trace;
    log_likelihood_trace.reserve(
        static_cast<std::size_t>(opts.max_iter > 0 ? opts.max_iter : 0) + 1);
    log_likelihood_trace.push_back(prev_llh);

    std::vector<complete_data_objective_record> complete_trace;
    complete_trace.reserve(
        static_cast<std::size_t>(opts.max_iter > 0 ? opts.max_iter : 0));

    int  iter_count = 0;
    bool converged  = false;

    for (int iter = 0; iter < opts.max_iter; ++iter) {
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

        auto suff = problem.compute_suff_stats();
        problem.m_step(suff);
        ++iter_count;
        complete_trace.push_back(
            compute_complete_data_objective_record(problem, suff, iter_count));

        double new_llh = problem.log_likelihood();
        log_likelihood_trace.push_back(new_llh);

        if (opts.verbose) {
            std::cerr << "[run_phylo_gmm_em] iter " << iter_count
                      << ": log L = " << new_llh
                      << "  (delta = " << (new_llh - prev_llh) << ")\n";
        }

        if (opts.warn_on_decrease && new_llh < prev_llh - 1e-9) {
            std::cerr << "[run_phylo_gmm_em] WARNING: log-L decreased at iter "
                      << iter_count << ": " << prev_llh
                      << " -> " << new_llh
                      << "  (delta = " << (new_llh - prev_llh) << ")\n";
        }

        double rel_delta = (new_llh - prev_llh) / std::abs(prev_llh);
        if (rel_delta >= 0.0 && rel_delta < opts.rel_tol) {
            prev_llh  = new_llh;
            converged = true;
            break;
        }
        prev_llh = new_llh;
    }

    phylo_gmm_em_output out;
    out.results.params               = problem.get_params();
    out.results.log_likelihood       = prev_llh;
    out.results.log_likelihood_trace = std::move(log_likelihood_trace);
    out.results.num_iterations       = iter_count;
    out.results.converged            = converged;
    out.results.trees.reserve(problem.trees.size());
    for (const auto* t : problem.trees) out.results.trees.push_back(*t);
    out.complete_data_objective_trace = std::move(complete_trace);
    return out;
}

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
    int                          seed = 0;
    double                       initial_log_likelihood = 0.0;
    double                       log_likelihood = 0.0;
    bool                         converged = false;
    int                          iterations = 0;
    std::vector<double>          log_likelihood_trace;
    std::vector<complete_data_objective_record>
                                 complete_data_objective_trace;
    std::vector<Eigen::VectorXd> initial_mu;
    std::vector<Eigen::MatrixXd> initial_Sigma;
    Eigen::MatrixXd              initial_Q;
    Eigen::VectorXd              initial_pi;
    std::vector<Eigen::VectorXd> mu;
    std::vector<Eigen::MatrixXd> Sigma;
    Eigen::MatrixXd              Q;
    Eigen::VectorXd              pi;
};

// Bundle of parameters loaded from --init_params_json / --init_emission_json.
// mu and Sigma are required; Q and pi are optional (when std::nullopt the
// caller falls back to the binary's normal init for that field). When present,
// this struct replaces the kmeans++/random init of (μ, Σ) and optionally
// (Q, π) for that restart.
struct init_emission_bundle {
    std::vector<Eigen::VectorXd>  mu;     // K vectors of length d
    std::vector<Eigen::MatrixXd>  Sigma;  // K d×d covariance matrices
    std::optional<Eigen::MatrixXd> Q;     // K×K (optional)
    std::optional<Eigen::VectorXd> pi;    // K (optional)
};

struct restart_job {
    std::size_t index = 0;
    std::string label;
    int seed = 0;
    bool use_kmeans_pp = true;
};

restart_record run_em_restart(std::vector<tree*>&               trees,
                              std::vector<Eigen::VectorXd>&     observed_data,
                              meta_data&                         metadata,
                              std::size_t                        k,
                              std::size_t                        d,
                              const std::string&                 label,
                              int                                seed,
                              bool                               use_kmeans_pp,
                              bool                               kmeans_standard,
                              int                                kmeans_max_iter,
                              double                             kmeans_tol,
                              bool                               fix_pi,
                              double                             tol,
                              int                                max_iter,
                              double                             q_init_rate,
                              double                             ctmc_weight,
                              double                             emission_weight,
                              std::optional<double>              fix_q_rate    = std::nullopt,
                              bool                               fix_emission  = false,
                              bool                               diagonal_covariance = false,
                              std::optional<Eigen::MatrixXd>     fix_q_matrix  = std::nullopt,
                              const init_emission_bundle*        init_emission = nullptr,
                              bool                               verbose       = false) {
    ctmc_model stm;
    stm.complete_data_weight = ctmc_weight;
    gmm_observation_model obs;
    obs.d = d;
    obs.diagonal_covariance = diagonal_covariance;

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

    std::vector<std::vector<std::monostate>> nds(trees.size());
    for (std::size_t i = 0; i < trees.size(); ++i)
        nds[i].assign(trees[i]->num_nodes, std::monostate{});

    phylo_gmm_problem problem;
    problem.stm           = &stm;
    problem.obs           = &obs;
    problem.observed_data = &observed_data;
    problem.metadata      = &metadata;
    problem.ctmc_weight   = ctmc_weight;
    problem.emission_weight = emission_weight;
    for (std::size_t i = 0; i < trees.size(); ++i) {
        problem.trees.push_back(trees[i]);
        problem.node_data.push_back(&nds[i]);
    }

    // CLI fix_pi=1 is the legacy spelling for --pi_mode free. We map that to
    // stationary_pi=false so EM updates pi from root posteriors instead of
    // deriving it from Q. Without a JSON pi, the free-pi path starts at 1/k.
    // Default (fix_pi=0) uses stationary_pi=true, so pi tracks the stationary
    // distribution of Q.
    problem.stationary_pi = !fix_pi;
    problem.q_init_rate_  = q_init_rate;

    if (init_emission) {
        // --init_params_json / --init_emission_json: skip kmeans++/random
        // init. Install (μ, Σ) from JSON; install Q from JSON if present,
        // otherwise build the standard uniform-exit-rate Q from q_init_rate;
        // install π from JSON if present, otherwise uniform. The same EM
        // machinery (compute_suff_stats / m_step / set_params) runs on top of
        // this init unchanged. In stationary pi mode, apply_state_and_rebuild_
        // will derive pi from Q; use --pi_mode free to use a supplied pi as a
        // true starting point.
        obs.d     = d;
        obs.mu    = init_emission->mu;
        obs.Sigma = init_emission->Sigma;
        if (init_emission->Q) {
            stm.Q = *init_emission->Q;
        } else {
            problem.q_init_rate_ = q_init_rate;
            problem.initialize_q_from_rate_();
        }
        if (init_emission->pi) {
            std::vector<double> pi_v(k);
            for (std::size_t i = 0; i < k; ++i)
                pi_v[i] = (*init_emission->pi)[static_cast<Eigen::Index>(i)];
            stm.root_prior = { pi_v };
        } else {
            stm.root_prior = { std::vector<double>(k, 1.0 / static_cast<double>(k)) };
        }
        problem.apply_state_and_rebuild_();
    } else if (use_kmeans_pp) {
        problem.initialize_kmeans_pp(static_cast<uint64_t>(seed),
                                     kmeans_standard,
                                     kmeans_max_iter,
                                     kmeans_tol);
    } else {
        problem.initialize_random(static_cast<uint64_t>(seed));
    }

    // --fix_q / --fix_q_json: clobber Q with the given matrix and pin it. In
    // stationary-pi mode, derive pi from this fixed Q once before setting
    // fix_Q=true; subsequent rebuilds can then skip the recompute.
    // fix_q_matrix (from --fix_q_json) takes precedence over fix_q_rate
    // (from --fix_q <rate>); the CLI parser already enforces they're not
    // both set.
    if (fix_q_matrix) {
        stm.Q = *fix_q_matrix;
        if (problem.stationary_pi) {
            stm.build();
            problem.set_pi_on_root_prior_(problem.extract_stationary_pi_());
        }
        problem.fix_Q = true;
        problem.apply_state_and_rebuild_();
    } else if (fix_q_rate) {
        stm.Q = make_uniform_exit_q(k, *fix_q_rate);
        if (problem.stationary_pi) {
            stm.build();
            problem.set_pi_on_root_prior_(problem.extract_stationary_pi_());
        }
        problem.fix_Q = true;
        problem.apply_state_and_rebuild_();
    }

    if (fix_pi && !(init_emission && init_emission->pi)) {
        std::vector<double> uniform_pi(k, 1.0 / static_cast<double>(k));
        stm.root_prior = { uniform_pi };
    }

    // --fix_emission, --fix_labels: μ, Σ frozen at kmeans++ init. Hard-label
    // clamping itself lives in metadata.hard_labels (populated by the caller)
    // and takes effect inside compute_inside_log_likelihood.
    if (fix_emission) problem.fix_emission = true;

    Eigen::VectorXd initial_pi_rec(static_cast<Eigen::Index>(k));
    for (std::size_t i = 0; i < k; ++i)
        initial_pi_rec[static_cast<Eigen::Index>(i)] = stm.root_prior[0][i];

    const std::vector<Eigen::VectorXd> initial_mu    = obs.mu;
    const std::vector<Eigen::MatrixXd> initial_Sigma = obs.Sigma;
    const Eigen::MatrixXd              initial_Q     = stm.Q;
    const Eigen::VectorXd              initial_pi    = initial_pi_rec;
    const double initial_log_likelihood = problem.log_likelihood();

    phylocore::optimize::em_options opts;
    opts.rel_tol          = tol;
    opts.max_iter         = max_iter;
    opts.verbose          = verbose;
    opts.warn_on_decrease = false;

    auto em_out = run_phylo_gmm_em(problem, opts);
    auto& results = em_out.results;

    Eigen::VectorXd pi_rec(static_cast<Eigen::Index>(k));
    for (std::size_t i = 0; i < k; ++i)
        pi_rec[static_cast<Eigen::Index>(i)] = stm.root_prior[0][i];

    restart_record r;
    r.label                  = label;
    r.seed                   = seed;
    r.initial_log_likelihood = initial_log_likelihood;
    r.log_likelihood         = results.log_likelihood;
    r.converged              = results.converged;
    r.iterations             = results.num_iterations;
    r.log_likelihood_trace   = results.log_likelihood_trace;
    r.complete_data_objective_trace =
        std::move(em_out.complete_data_objective_trace);
    r.initial_mu             = initial_mu;
    r.initial_Sigma          = initial_Sigma;
    r.initial_Q              = initial_Q;
    r.initial_pi             = initial_pi;
    r.mu                     = obs.mu;
    r.Sigma                  = obs.Sigma;
    r.Q                      = stm.Q;
    r.pi                     = pi_rec;
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON dump
// ─────────────────────────────────────────────────────────────────────────────

void dump_json_number(std::ostream& os, double v) {
    if (std::isfinite(v)) {
        os << v;
    } else if (std::isnan(v)) {
        os << "NaN";
    } else if (v > 0.0) {
        os << "Infinity";
    } else {
        os << "-Infinity";
    }
}

void dump_matrix(std::ostream& os, const Eigen::MatrixXd& M) {
    os << "[";
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        if (i > 0) os << ", ";
        os << "[";
        for (Eigen::Index j = 0; j < M.cols(); ++j) {
            if (j > 0) os << ", ";
            dump_json_number(os, M(i, j));
        }
        os << "]";
    }
    os << "]";
}

void dump_vec(std::ostream& os, const Eigen::VectorXd& v) {
    os << "[";
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (i > 0) os << ", ";
        dump_json_number(os, v[i]);
    }
    os << "]";
}

void dump_double_vector(std::ostream& os, const std::vector<double>& values) {
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) os << ", ";
        dump_json_number(os, values[i]);
    }
    os << "]";
}

void dump_complete_data_objective_trace(
    std::ostream& os,
    const std::vector<complete_data_objective_record>& values)
{
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) os << ", ";
        const auto& r = values[i];
        os << "{\"iteration\":" << r.iteration
           << ",\"root\":";
        dump_json_number(os, r.root);
        os << ",\"ctmc\":";
        dump_json_number(os, r.ctmc);
        os << ",\"emission\":";
        dump_json_number(os, r.emission);
        os << ",\"weighted_ctmc\":";
        dump_json_number(os, r.weighted_ctmc);
        os << ",\"weighted_emission\":";
        dump_json_number(os, r.weighted_emission);
        os << ",\"total\":";
        dump_json_number(os, r.total);
        os << "}";
    }
    os << "]";
}

void dump_vector_list(std::ostream& os, const std::vector<Eigen::VectorXd>& values) {
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) os << ", ";
        dump_vec(os, values[i]);
    }
    os << "]";
}

void dump_matrix_list(std::ostream& os, const std::vector<Eigen::MatrixXd>& values) {
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) os << ", ";
        dump_matrix(os, values[i]);
    }
    os << "]";
}

// ─────────────────────────────────────────────────────────────────────────────
// Posterior extraction for the best restart
// ─────────────────────────────────────────────────────────────────────────────

struct posterior_outputs {
    // Marginal log-likelihood at the best params, computed during the final
    // E-step pass. In EM modes this matches (up to floating-point drift) the
    // best restart's log_likelihood. In --bayes_opt mode this is the only
    // log-likelihood that's ever computed (no EM ran).
    double                            log_likelihood = 0.0;

    // Total_N_leaves × K, indexed by global taxa_id (= leaf vertex .data,
    // contiguous across trees). Rows are normalized probabilities (sum to 1).
    std::vector<std::vector<double>> leaf_posteriors;
    std::vector<int>                  leaf_labels;          // argmax per row

    // Total_N_internal × K. Internal nodes have no global taxa_id; row r
    // corresponds to phylogeny node internal_node_ids[r] WITHIN tree
    // tree_index_per_internal[r]. Callers MUST use both arrays to demux.
    std::vector<std::vector<double>> internal_posteriors;
    std::vector<int>                  internal_node_ids;
    std::vector<int>                  tree_index_per_internal;

    Eigen::MatrixXd expected_transitions;  // K × K, hj.N_total summed across trees
    Eigen::VectorXd expected_dwell_times;  // K,     hj.T_total summed across trees
};

// Rebuilds a fresh problem with the best restart's params, runs one E-step
// pass (inside + edge_inside + outside) per tree, then extracts node-level
// posteriors and Hobolth-Jensen aggregate. Cost: one EM iteration's worth.
posterior_outputs compute_posteriors(
    std::vector<tree*>&            trees,
    std::vector<Eigen::VectorXd>& observed_data,
    meta_data&                    metadata,
    std::size_t                    k,
    std::size_t                    d,
    bool                           fix_pi,
    bool                           diagonal_covariance,
    double                         ctmc_weight,
    double                         emission_weight,
    const restart_record&         best)
{
    ctmc_model stm;
    stm.complete_data_weight = ctmc_weight;
    stm.Q = best.Q;
    // Pre-install π as root_prior so ctmc_model::build() doesn't overwrite it
    // with uniform (its empty-or-wrong-size fallback).
    {
        std::vector<double> pi_v(k);
        for (std::size_t i = 0; i < k; ++i)
            pi_v[i] = best.pi[static_cast<Eigen::Index>(i)];
        stm.root_prior = { pi_v };
    }
    stm.build();

    gmm_observation_model obs;
    obs.d     = d;
    obs.diagonal_covariance = diagonal_covariance;
    obs.mu    = best.mu;
    obs.Sigma = best.Sigma;
    obs.sites_latent_states = stm.sites_latent_states;
    obs.build();

    std::vector<std::vector<std::monostate>> nds(trees.size());
    for (std::size_t i = 0; i < trees.size(); ++i)
        nds[i].assign(trees[i]->num_nodes, std::monostate{});

    phylo_gmm_problem prob;
    prob.stm           = &stm;
    prob.obs           = &obs;
    prob.observed_data = &observed_data;
    prob.metadata      = &metadata;
    prob.ctmc_weight   = ctmc_weight;
    prob.emission_weight = emission_weight;
    for (std::size_t i = 0; i < trees.size(); ++i) {
        prob.trees.push_back(trees[i]);
        prob.node_data.push_back(&nds[i]);
    }
    prob.stationary_pi = !fix_pi;

    // E-step pass at the best params (inside is populated by log_likelihood;
    // edge_inside + outside need explicit calls per tree). The marginal log L
    // returned here is what compute_posteriors stashes into out.log_likelihood
    // — the only log-L source in --bayes_opt mode.
    const double final_log_likelihood = prob.log_likelihood();
    for (std::size_t ti = 0; ti < trees.size(); ++ti) {
        phylocore::compute_edge_inside_log_likelihood(
            stm, *trees[ti],
            prob.inside_buffer(ti), prob.edge_inside_buffer(ti), nds[ti]);
        phylocore::compute_outside_log_likelihood(
            stm, *trees[ti],
            prob.edge_inside_buffer(ti), prob.outside_buffer(ti), nds[ti]);
    }

    std::size_t total_leaves = 0;
    std::size_t total_internal = 0;
    for (auto* t : trees) {
        total_leaves   += t->num_leaves;
        total_internal += t->num_nodes - t->num_leaves;
    }

    posterior_outputs out;
    out.leaf_posteriors.assign(total_leaves, std::vector<double>(k, 0.0));
    out.leaf_labels.assign(total_leaves, 0);
    out.internal_posteriors.reserve(total_internal);
    out.internal_node_ids.reserve(total_internal);
    out.tree_index_per_internal.reserve(total_internal);

    for (std::size_t ti = 0; ti < trees.size(); ++ti) {
        const tree& t = *trees[ti];
        for (int v : t.tree.nodes()) {
            const bool is_leaf = (t.tree.out_degree(v) == 0);
            auto log_gamma = prob.node_posterior(ti, static_cast<std::size_t>(v), 0);

            // exp + normalize. Subtract max first to avoid underflow.
            double m = log_gamma[0];
            for (std::size_t i = 1; i < k; ++i)
                if (log_gamma[i] > m) m = log_gamma[i];

            std::vector<double> probs(k);
            double s = 0.0;
            for (std::size_t i = 0; i < k; ++i) {
                probs[i] = std::exp(log_gamma[i] - m);
                s += probs[i];
            }
            if (s > 0.0) {
                for (auto& p : probs) p /= s;
            } else {
                // Defensive: degenerate node (shouldn't happen on a converged
                // model). Fall through to uniform 1/k so rows still sum to 1.
                for (auto& p : probs) p = 1.0 / static_cast<double>(k);
            }

            int label = 0;
            for (std::size_t i = 1; i < k; ++i)
                if (probs[i] > probs[label]) label = static_cast<int>(i);

            if (is_leaf) {
                const std::size_t taxa_id =
                    static_cast<std::size_t>(t.tree[v].data);
                out.leaf_posteriors[taxa_id] = std::move(probs);
                out.leaf_labels[taxa_id]     = label;
            } else {
                out.internal_posteriors.push_back(std::move(probs));
                out.internal_node_ids.push_back(v);
                out.tree_index_per_internal.push_back(static_cast<int>(ti));
            }
        }
    }

    auto hj = phylocore::optimize::hobolth_jensen_stats(prob);
    out.expected_transitions = hj.N_total;
    out.expected_dwell_times = hj.T_total;
    out.log_likelihood       = final_log_likelihood;
    return out;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// CLI parsing — supports both legacy positional and new flag-based forms.
// Flag form lets us pass multiple --tree/--obs pairs for joint inference.
// ─────────────────────────────────────────────────────────────────────────────

struct cli_args {
    std::vector<std::string> tree_files;
    std::vector<std::string> obs_files;
    std::size_t k            = 0;
    int         max_iter     = 100;
    double      tol          = 1e-5;
    double      q_init_rate  = -1.0;
    bool        fix_pi       = true;   // default pi_mode=free (fix_pi=true => stationary_pi=false)
    int         n_init       = 10;
    bool        verbose      = false;
    double      ctmc_weight  = 1.0;
    double      emission_weight = 1.0;
    bool        kmeans_standard = false;
    int         kmeans_max_iter = 100;
    double      kmeans_tol      = 1e-6;
    bool        diagonal_covariance = false;
    bool        parallel_restarts   = false;
    int         restart_threads     = 0;  // 0 means hardware_concurrency

    // Mode selectors. At most one of {fix_q_rate/fix_q_json, fix_labels_csv,
    // bayes_opt} may be set. --fix_emission may be combined with --fix_q
    // (fix both), and is implicit under --fix_labels and --bayes_opt.
    bool                       fix_emission     = false;
    std::optional<double>      fix_q_rate;             // --fix_q <rate>
    std::optional<std::string> fix_q_json;             // --fix_q_json <path>
    std::optional<std::string> fix_labels_csv;         // --fix_labels <csv>
    bool                       bayes_opt        = false;
    std::optional<double>      true_q_exit_rate;       // --true_q_exit_rate
    std::optional<double>      true_simplex_scale;     // --true_simplex_scale
    std::optional<std::string> bayes_opt_truth_json;   // --bayes_opt_truth_json
    std::optional<std::string> init_params_json;       // --init_params_json
    std::optional<std::string> init_emission_json;     // legacy alias
};

void print_usage(const char* prog) {
    std::cerr <<
      "Usage (single tree, positional — default full-EM mode only):\n"
      "  " << prog << " <tree_file> <obs_csv> <k> "
                       "[max_iter] [tol] [q_init_rate] [fix_pi] [n_init]\n"
      "\n"
      "Usage (joint inference / mode selection, flag-based):\n"
      "  " << prog << " --k K [--max_iter N] [--tol T] [--q_init_rate R] \\\n"
      "                [--pi_mode stationary|free] [--n_init N] [--verbose] \\\n"
      "                [--ctmc_weight W_C] [--emission_weight W_E] \\\n"
      "                [--kmeans_init seeding|standard] \\\n"
      "                [--init_params_json params.json] \\\n"
      "                --tree T1 --obs O1 [--tree T2 --obs O2 ...] \\\n"
      "                [mode flag]\n"
      "\n"
      "K-means++ initialization (default 'seeding'):\n"
      "  --kmeans_init seeding   choose k-means++ centers only; no Lloyd steps\n"
      "  --kmeans_init standard  choose k-means++ centers, then run Lloyd k-means\n"
      "                          before EM uses the centers as initial μ\n"
      "  --kmeans_max_iter N     Lloyd iteration cap for --kmeans_init standard\n"
      "  --kmeans_tol T          Lloyd convergence tolerance on center movement\n"
      "\n"
      "Performance / covariance options:\n"
      "  --diag_covariance       fit diagonal Σ_i instead of full d×d covariance\n"
      "                          matrices. Equivalent spelling: --diagonal_covariance.\n"
      "  --parallel_restarts [N] run independent restarts concurrently. Optional N\n"
      "                          caps worker threads; omit or use 0 for hardware count.\n"
      "  --restart_threads N     set the parallel restart worker cap and enable\n"
      "                          --parallel_restarts.\n"
      "\n"
      "Diagnostics:\n"
      "  --verbose               Emit per-iteration EM log-likelihood trace to stderr.\n"
      "\n"
      "Pi mode (default 'free'):\n"
      "  --pi_mode stationary    pi = stationary distribution of Q, recomputed\n"
      "                          after every Q update. NOT a free EM parameter.\n"
      "  --pi_mode free          pi initialized uniformly, then freely fit each\n"
      "                          M-step from the root posteriors.\n"
      "  --fix_pi 0|1            DEPRECATED alias. 0==stationary, 1==free.\n"
      "\n"
      "Weighted complete-data objective (defaults both to 1):\n"
      "  --ctmc_weight W_C       weight on the complete CTMC path term\n"
      "  --emission_weight W_E   weight on the Gaussian emission term\n"
      "  --w_C / --w_E           aliases for the two options above\n"
      "                          Weights must be finite and strictly positive.\n"
      "                          With non-unit weights, JSON field log_likelihood\n"
      "                          is the generalized (unnormalized) marginal\n"
      "                          log-objective, used for convergence/restarts.\n"
      "\n"
      "Mode flags (at most one of the first three, plus optional --fix_emission):\n"
      "  --fix_q <rate>          Q held at uniform-exit-rate matrix; EM on μ, Σ, π\n"
      "  --fix_q_json <path>     Q held at the K×K matrix read from JSON (key \"Q\");\n"
      "                          EM on μ, Σ, π. Alternative to --fix_q for arbitrary\n"
      "                          (asymmetric / absorbing) Q matrices.\n"
      "  --fix_emission          μ, Σ held at kmeans++ init; EM on Q (and π)\n"
      "  --fix_labels <csv>      Leaf state assignments clamped to CSV labels;\n"
      "                          EM on Q only. CSV header: cell,state (0-indexed).\n"
      "                          Implies --fix_emission.\n"
      "  --bayes_opt --true_q_exit_rate <r> --true_simplex_scale <s>\n"
      "                          No EM. True (Q, μ, Σ, π) constructed from flags;\n"
      "                          single inside-outside pass yields posteriors.\n"
      "                          k=4 d=3 only (simplex hardcoded).\n"
      "  --bayes_opt --bayes_opt_truth_json <path>\n"
      "                          As above, but Q (K×K), mu (K×d), covariance\n"
      "                          (Sigma K×d×d, or sigma_diag K×d), and optional\n"
      "                          pi (K) are read from JSON. pi defaults to 1/k if\n"
      "                          absent. Replaces --true_q_exit_rate and\n"
      "                          --true_simplex_scale for arbitrary K, d, and μ.\n"
      "\n"
      "Init override (combinable with EM modes, not with --bayes_opt / --fix_labels):\n"
      "  --init_params_json <path>\n"
      "                          Use user-defined (μ, Σ) — and optionally (Q, π) —\n"
      "                          as the EM starting point instead of kmeans++/random\n"
      "                          init. Q defaults to uniform-exit-rate from\n"
      "                          --q_init_rate if absent; π defaults to 1/k. In\n"
      "                          --pi_mode stationary, π is derived from Q; use\n"
      "                          --pi_mode free to honor a JSON π as the start.\n"
      "                          EM then proceeds normally (updates Q, μ, Σ, π — or\n"
      "                          whatever isn't frozen by --fix_q / --fix_q_json /\n"
      "                          --fix_emission). With --n_init > 1, every restart\n"
      "                          uses the same JSON init (deterministic).\n"
      "  --init_emission_json <path>\n"
      "                          Deprecated alias for --init_params_json. Retains\n"
      "                          compatibility with older files that provide\n"
      "                          mu + sigma_diag (small ridge added to sigma_diag).\n"
      "\n"
      "JSON schema (single file works for --fix_q_json, --bayes_opt_truth_json, or\n"
      "--init_params_json; extra keys are tolerated):\n"
      "  {\"Q\": [[K x K]], \"mu\": [[K x d]],\n"
      "   \"Sigma\": [[[d x d], ... K]], \"sigma_diag\": [[K x d]],\n"
      "   \"pi\": [K]}\n";
}

bool parse_cli(int argc, char* argv[], cli_args& a) {
    if (argc < 2) return false;

    // Flag form is detected by leading "--".
    if (std::string(argv[1]).rfind("--", 0) == 0) {
        for (int i = 1; i < argc; ++i) {
            std::string s = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value after " + s);
                return argv[++i];
            };
            if      (s == "--tree")              a.tree_files.push_back(next());
            else if (s == "--obs")               a.obs_files.push_back(next());
            else if (s == "--k")                 a.k           = std::stoul(next());
            else if (s == "--max_iter")          a.max_iter    = std::stoi(next());
            else if (s == "--tol")               a.tol         = std::stod(next());
            else if (s == "--q_init_rate")       a.q_init_rate = std::stod(next());
            else if (s == "--ctmc_weight" || s == "--w_C")
                a.ctmc_weight = std::stod(next());
            else if (s == "--emission_weight" || s == "--w_E")
                a.emission_weight = std::stod(next());
            else if (s == "--pi_mode") {
                std::string m = next();
                if      (m == "stationary") a.fix_pi = false;
                else if (m == "free")       a.fix_pi = true;
                else throw std::runtime_error(
                    "--pi_mode must be 'stationary' (pi = stationary(Q)) "
                    "or 'free' (pi fit from root posteriors via M-step). "
                    "Got: '" + m + "'");
            }
            else if (s == "--fix_pi") {
                std::cerr << "warning: --fix_pi is deprecated; use "
                             "--pi_mode {stationary,free}. "
                             "--fix_pi 0 == --pi_mode stationary, "
                             "--fix_pi 1 == --pi_mode free.\n";
                a.fix_pi = std::stoi(next()) != 0;
            }
            else if (s == "--n_init")            a.n_init      = std::stoi(next());
            else if (s == "--verbose")           a.verbose          = true;
            else if (s == "--kmeans_init") {
                std::string m = next();
                if      (m == "seeding")  a.kmeans_standard = false;
                else if (m == "standard") a.kmeans_standard = true;
                else throw std::runtime_error(
                    "--kmeans_init must be 'seeding' or 'standard'. Got: '" + m + "'");
            }
            else if (s == "--kmeans_max_iter")   a.kmeans_max_iter = std::stoi(next());
            else if (s == "--kmeans_tol")        a.kmeans_tol      = std::stod(next());
            else if (s == "--fix_emission")      a.fix_emission     = true;
            else if (s == "--diag_covariance" ||
                     s == "--diagonal_covariance") {
                a.diagonal_covariance = true;
            }
            else if (s == "--parallel_restarts") {
                a.parallel_restarts = true;
                if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                    a.restart_threads = std::stoi(argv[++i]);
                }
            }
            else if (s == "--restart_threads") {
                a.restart_threads = std::stoi(next());
                a.parallel_restarts = true;
            }
            else if (s == "--fix_q")             a.fix_q_rate       = std::stod(next());
            else if (s == "--fix_q_json")        a.fix_q_json       = next();
            else if (s == "--fix_labels")        a.fix_labels_csv   = next();
            else if (s == "--bayes_opt")         a.bayes_opt        = true;
            else if (s == "--true_q_exit_rate")  a.true_q_exit_rate = std::stod(next());
            else if (s == "--true_simplex_scale") a.true_simplex_scale = std::stod(next());
            else if (s == "--bayes_opt_truth_json") a.bayes_opt_truth_json = next();
            else if (s == "--init_params_json") a.init_params_json = next();
            else if (s == "--init_emission_json") a.init_emission_json = next();
            else throw std::runtime_error("unknown flag: " + s);
        }
        if (a.tree_files.size() != a.obs_files.size())
            throw std::runtime_error("--tree and --obs must be paired (same count)");
        if (a.tree_files.empty())
            throw std::runtime_error("at least one --tree/--obs pair required");
        if (a.k == 0)
            throw std::runtime_error("--k is required");
        if (a.n_init < 1)
            throw std::runtime_error("--n_init must be >= 1");
        if (a.kmeans_max_iter < 0)
            throw std::runtime_error("--kmeans_max_iter must be >= 0");
        if (a.kmeans_tol < 0.0)
            throw std::runtime_error("--kmeans_tol must be >= 0");
        if (a.restart_threads < 0)
            throw std::runtime_error("--restart_threads must be >= 0");
        if (!std::isfinite(a.ctmc_weight) || a.ctmc_weight <= 0.0)
            throw std::runtime_error("--ctmc_weight/--w_C must be finite and > 0");
        if (!std::isfinite(a.emission_weight) || a.emission_weight <= 0.0)
            throw std::runtime_error(
                "--emission_weight/--w_E must be finite and > 0");

        // Mode-flag validation: at most one of the three primary mode flags.
        // --fix_q_json is an alternative form of --fix_q (treated as the same
        // "fix Q" mode for mutual-exclusion purposes).
        const bool fix_q_any = a.fix_q_rate.has_value() ||
                               a.fix_q_json.has_value();
        if (a.fix_q_rate && a.fix_q_json)
            throw std::runtime_error(
                "--fix_q and --fix_q_json are mutually exclusive");
        const int n_modes = (fix_q_any         ? 1 : 0)
                          + (a.fix_labels_csv  ? 1 : 0)
                          + (a.bayes_opt       ? 1 : 0);
        if (n_modes > 1)
            throw std::runtime_error(
                "--fix_q/--fix_q_json, --fix_labels, --bayes_opt are mutually exclusive");
        // --init_params_json: incompatible with --bayes_opt (which doesn't run
        // EM) and --fix_labels (which clamps μ, Σ from kmeans++ init —
        // semantically incompatible with reading them from JSON). Compatible
        // with --fix_q, --fix_q_json, and --fix_emission: in those cases the
        // JSON simply provides the initial values that are then either frozen
        // or updated by EM, depending on the freeze flags.
        const bool has_init_json =
            a.init_params_json.has_value() || a.init_emission_json.has_value();
        if (a.init_params_json && a.init_emission_json)
            throw std::runtime_error(
                "--init_params_json and --init_emission_json are aliases; use only one");
        if (has_init_json) {
            if (a.bayes_opt)
                throw std::runtime_error(
                    "--init_params_json is incompatible with --bayes_opt");
            if (a.bayes_opt_truth_json)
                throw std::runtime_error(
                    "--init_params_json is incompatible with --bayes_opt_truth_json");
            if (a.fix_labels_csv)
                throw std::runtime_error(
                    "--init_params_json is incompatible with --fix_labels");
        }
        if (a.bayes_opt) {
            // --bayes_opt_truth_json supersedes --true_q_exit_rate / --true_simplex_scale.
            if (a.bayes_opt_truth_json) {
                if (a.true_q_exit_rate || a.true_simplex_scale)
                    throw std::runtime_error(
                        "--bayes_opt_truth_json cannot be combined with "
                        "--true_q_exit_rate / --true_simplex_scale");
            } else {
                if (!a.true_q_exit_rate)
                    throw std::runtime_error(
                        "--bayes_opt requires --true_q_exit_rate (or --bayes_opt_truth_json)");
                if (!a.true_simplex_scale)
                    throw std::runtime_error(
                        "--bayes_opt requires --true_simplex_scale (or --bayes_opt_truth_json)");
                if (a.k != 4)
                    throw std::runtime_error(
                        "--bayes_opt without --bayes_opt_truth_json requires --k 4 "
                        "(simplex hardcoded)");
            }
        }
        return true;
    }

    // Legacy positional: <tree> <obs> <k> [max_iter] [tol] [q_init_rate] [fix_pi] [n_init]
    // No variant modes supported on this path — use flag form for those.
    if (argc < 4) return false;
    a.tree_files = { argv[1] };
    a.obs_files  = { argv[2] };
    a.k          = std::stoul(argv[3]);
    if (argc > 4) a.max_iter    = std::stoi(argv[4]);
    if (argc > 5) a.tol         = std::stod(argv[5]);
    if (argc > 6) a.q_init_rate = std::stod(argv[6]);
    if (argc > 7) a.fix_pi      = std::stoi(argv[7]) != 0;
    if (argc > 8) a.n_init      = std::stoi(argv[8]);
    if (a.n_init < 1)
        throw std::runtime_error("n_init must be >= 1");
    return true;
}

std::size_t resolve_restart_worker_count(const cli_args& A,
                                         std::size_t restart_job_count) {
    if (restart_job_count == 0) return 0;
    if (!A.parallel_restarts || restart_job_count == 1) return 1;

    const unsigned int hw = std::thread::hardware_concurrency();
    const std::size_t requested =
        (A.restart_threads > 0)
            ? static_cast<std::size_t>(A.restart_threads)
            : static_cast<std::size_t>(hw == 0 ? 1 : hw);
    return std::max<std::size_t>(
        1, std::min(requested, restart_job_count));
}

// ─────────────────────────────────────────────────────────────────────────────
// Labels CSV parser (used by --fix_labels)
// ─────────────────────────────────────────────────────────────────────────────
//
// Header: "cell,state". Each subsequent row is "<name>,<int>". The int is a
// 0-indexed cluster label and must be in [0, k).

std::unordered_map<std::string, int> parse_labels_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open labels CSV: " + path);

    std::unordered_map<std::string, int> out;
    std::string line;
    // Header — skip if it doesn't start with a digit-state row.
    if (!std::getline(in, line))
        throw std::runtime_error("Empty labels CSV: " + path);
    {
        std::stringstream ss(line);
        std::string name, state_str;
        if (std::getline(ss, name, ',') && std::getline(ss, state_str)) {
            // Try parsing the second column as an int — if it succeeds, it's
            // a data row, not a header. Otherwise treat as header.
            try {
                int s = std::stoi(state_str);
                out[name] = s;
            } catch (...) { /* header row — discard */ }
        }
    }
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string name, state_str;
        if (!std::getline(ss, name, ',') || !std::getline(ss, state_str))
            throw std::runtime_error(
                "Bad row in labels CSV (expected 'name,state'): " + line);
        out[name] = std::stoi(state_str);
    }
    return out;
}

// (make_uniform_exit_q and SIMPLEX_4 moved to file scope above the anon
// namespace so run_em_restart can use them.)

int main_impl(int argc, char* argv[]) {
    cli_args A;
    try {
        if (!parse_cli(argc, argv, A)) { print_usage(argv[0]); return 1; }
    } catch (const std::exception& e) {
        std::cerr << "CLI error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    const std::size_t k           = A.k;
    const int         max_iter    = A.max_iter;
    const double      tol         = A.tol;
    const double      q_init_rate = A.q_init_rate;
    const bool        fix_pi      = A.fix_pi;
    const int         n_init      = A.n_init;
    const std::size_t n_trees     = A.tree_files.size();
    const unsigned int detected_cpus = std::thread::hardware_concurrency();
    const std::size_t restart_job_count_for_print =
        A.bayes_opt ? 0 : 2 * static_cast<std::size_t>(n_init);
    const std::size_t restart_worker_count_for_print =
        resolve_restart_worker_count(A, restart_job_count_for_print);
    const std::string restart_worker_mode_for_print =
        A.bayes_opt
            ? std::string("no_em_restarts")
            : ((A.parallel_restarts && restart_job_count_for_print > 1)
                   ? std::string("parallel")
                   : std::string("sequential"));

    std::cerr << "── phyloGMM (phylocore impl) ────────────────────────────────\n";
    std::cerr << "n_trees  = " << n_trees << "\n";
    for (std::size_t i = 0; i < n_trees; ++i) {
        std::cerr << "  tree[" << i << "] = " << A.tree_files[i] << "\n";
        std::cerr << "  obs [" << i << "] = " << A.obs_files[i]  << "\n";
    }
    std::cerr << "k        = " << k         << "\n";
    std::cerr << "max_iter = " << max_iter  << ", tol = " << tol << "\n";
    std::cerr << "fix_pi   = " << fix_pi    << ", n_init = " << n_init << "\n";
    std::cerr << "kmeans_init = "
              << (A.kmeans_standard ? "standard" : "seeding")
              << " (lloyd_max_iter = " << A.kmeans_max_iter
              << ", lloyd_tol = " << A.kmeans_tol << ")\n";
    std::cerr << "covariance = "
              << (A.diagonal_covariance ? "diagonal" : "full") << "\n";
    std::cerr << "parallel_restarts = "
              << (A.parallel_restarts ? "true" : "false");
    if (A.parallel_restarts) {
        std::cerr << " (threads = "
                  << (A.restart_threads > 0
                          ? std::to_string(A.restart_threads)
                          : std::string("hardware"))
                  << ")";
    }
    std::cerr << "\n";
    std::cerr << "available_cpus = "
              << (detected_cpus == 0
                      ? std::string("unknown")
                      : std::to_string(detected_cpus))
              << "\n";
    std::cerr << "restart_jobs = " << restart_job_count_for_print << "\n";
    std::cerr << "restart_workers = "
              << restart_worker_count_for_print;
    if (restart_worker_mode_for_print == "no_em_restarts") {
        std::cerr << " (no EM restarts)";
    } else if (restart_worker_mode_for_print == "parallel") {
        std::cerr << " (parallel)";
    } else {
        std::cerr << " (sequential)";
    }
    std::cerr << "\n";
    {
        const double effective_rate =
            (q_init_rate > 0.0) ? q_init_rate : 1.0 / static_cast<double>(k);
        std::cerr << "q_init_rate = " << q_init_rate
                  << " (effective rate = " << effective_rate;
        if (k >= 2) {
            std::cerr << ", off-diag entry = "
                      << effective_rate / static_cast<double>(k - 1);
        } else {
            std::cerr << ", k=1: no off-diag entries (Q = [[0]])";
        }
        std::cerr << ")\n";
    }

    // ── Parse all trees ────────────────────────────────────────────────────
    // Each tree is parsed independently; leaf taxa_ids start at 0 within each
    // tree. We then globally offset taxa_ids below so observed_data is a
    // single concatenated vector across trees.
    std::vector<tree> tree_storage;
    tree_storage.reserve(n_trees);
    for (const auto& tf : A.tree_files) {
        auto parsed = parse_newick_trees(tf);
        if (parsed.empty()) {
            std::cerr << "no trees parsed from " << tf << "\n";
            return 2;
        }
        tree_storage.push_back(std::move(parsed[0]));
    }
    std::vector<tree*> trees(n_trees);
    for (std::size_t i = 0; i < n_trees; ++i) trees[i] = &tree_storage[i];

    std::size_t total_leaves = 0;
    std::vector<std::size_t> n_leaves_per_tree(n_trees);
    for (std::size_t i = 0; i < n_trees; ++i) {
        n_leaves_per_tree[i] = trees[i]->num_leaves;
        total_leaves        += trees[i]->num_leaves;
        std::cerr << "Tree[" << i << "]: " << trees[i]->num_leaves
                  << " leaves, " << trees[i]->num_nodes << " nodes\n";
    }

    // ── Parse all observations and resolve a common dimension d ────────────
    std::vector<decltype(parse_observations_csv(""))> obs_csvs;
    obs_csvs.reserve(n_trees);
    std::size_t d = 0;
    for (std::size_t i = 0; i < n_trees; ++i) {
        auto oc = parse_observations_csv(A.obs_files[i]);
        if (i == 0) d = oc.d;
        else if (oc.d != d) {
            std::cerr << "ERROR: obs[" << i << "] has d=" << oc.d
                      << " but expected d=" << d
                      << " (all trees must share the same dimension)\n";
            return 3;
        }
        std::cerr << "Obs[" << i << "]: " << oc.by_name.size()
                  << " cells, d = " << oc.d << "\n";
        obs_csvs.push_back(std::move(oc));
    }

    // ── Align leaves across all trees with a single global taxa_id space ───
    // Each tree's leaves get global taxa_ids [offset, offset + tree.num_leaves)
    // where offset is the running sum of preceding trees' leaf counts.
    // We REWRITE each leaf's node.data to its new global ID so the existing
    // joint_optimization_problem machinery (which indexes observed_data by
    // node.data) works unchanged.
    std::vector<Eigen::VectorXd> observed_data(total_leaves,
                                                Eigen::VectorXd::Zero(d));
    std::vector<std::string> leaf_names(total_leaves);
    std::vector<int>         tree_index_per_leaf(total_leaves, 0);
    std::size_t offset = 0;
    for (std::size_t ti = 0; ti < n_trees; ++ti) {
        tree& t = *trees[ti];
        const auto& oc = obs_csvs[ti];
        std::size_t n_observed_in_tree = 0;
        for (int node_id : t.tree.nodes()) {
            if (t.tree.out_degree(node_id) != 0) continue;
            const std::string& name      = t.node_names[node_id];
            const std::size_t  local_id  = t.tree[node_id].data;
            const std::size_t  global_id = offset + local_id;
            t.tree[node_id].data = global_id;
            leaf_names[global_id]          = name;
            tree_index_per_leaf[global_id] = static_cast<int>(ti);
            auto it = oc.by_name.find(name);
            if (it != oc.by_name.end()) {
                observed_data[global_id] = it->second;
                ++n_observed_in_tree;
            }
        }
        std::cerr << "Aligned[" << ti << "]: " << n_observed_in_tree
                  << " / " << t.num_leaves << " leaves\n";
        if (n_observed_in_tree != t.num_leaves) {
            std::cerr << "ERROR: every leaf must be observed (tree " << ti
                      << ").\n";
            return 3;
        }
        offset += t.num_leaves;
    }
    const std::size_t n_observed = total_leaves;

    // ── Metadata ───────────────────────────────────────────────────────────
    meta_data metadata;
    metadata.num_sites         = 1;
    metadata.max_alphabet_size = k;
    metadata.alphabet_sizes    = { k };
    metadata.taxa_names        = std::move(leaf_names);

    // ── Hard-label clamping (--fix_labels) ─────────────────────────────────
    // Populated AFTER the global taxa_id alignment above, so the indices line
    // up with observed_data and metadata.taxa_names. Read by
    // compute_inside_log_likelihood at the leaf branch.
    if (A.fix_labels_csv) {
        metadata.hard_labels.assign(total_leaves, -1);
        const auto labels = parse_labels_csv(*A.fix_labels_csv);
        std::size_t n_matched = 0, n_oob = 0;
        for (std::size_t ti = 0; ti < n_trees; ++ti) {
            const tree& t = *trees[ti];
            for (int node_id : t.tree.nodes()) {
                if (t.tree.out_degree(node_id) != 0) continue;
                const std::string& name      = t.node_names[node_id];
                const std::size_t  global_id = t.tree[node_id].data;
                auto it = labels.find(name);
                if (it == labels.end()) continue;
                if (it->second < 0 || static_cast<std::size_t>(it->second) >= k) {
                    ++n_oob; continue;
                }
                metadata.hard_labels[global_id] = it->second;
                ++n_matched;
            }
        }
        std::cerr << "Hard labels: " << n_matched << " / " << total_leaves
                  << " leaves clamped (" << n_oob
                  << " out-of-range entries skipped)\n";
    }

    // ── Flag dispatch ──────────────────────────────────────────────────────
    // The four modes converge here:
    //   * --bayes_opt: bypass the restart-EM loop entirely. We build the true
    //     parameters into a single synthetic restart_record and pass it
    //     straight to compute_posteriors. records remains empty so the
    //     "per_restart" JSON array is [].
    //   * Everything else: run the standard restart-EM loop, propagating
    //     fix_q_rate / fix_emission into each run_em_restart call. --fix_labels
    //     additionally clamps leaf inside likelihoods via metadata.hard_labels.

    std::vector<restart_record> records;
    restart_record              best_storage;  // only used by --bayes_opt
    const restart_record*       best = nullptr;
    std::optional<std::size_t>  best_restart_index;

    if (A.bayes_opt) {
        std::cerr << "── --bayes_opt mode (no EM) ─────────────────────────────────\n";

        best_storage.label          = "bayes_opt";
        best_storage.seed           = -1;
        best_storage.iterations     = 0;
        best_storage.converged      = true;
        best_storage.log_likelihood = 0.0;  // filled by compute_posteriors

        if (A.bayes_opt_truth_json) {
            // True (Q, μ, Σ, π) loaded from JSON. Σ may be a full K×d×d
            // "Sigma" array or compact diagonal "sigma_diag"; π defaults to
            // uniform if absent.
            std::cerr << "bayes_opt_truth_json = " << *A.bayes_opt_truth_json << "\n";
            const truth_json tj = load_truth_json(*A.bayes_opt_truth_json);
            best_storage.Q = vec2d_to_matrix(tj.Q, k, k, "Q");
            validate_ctmc_q(best_storage.Q, "Q");
            best_storage.mu =
                vec2d_to_vector_list(tj.mu, k, d, "mu");
            best_storage.Sigma =
                sigma_from_json(tj, k, d, A.diagonal_covariance, 0.0,
                                "--bayes_opt_truth_json");
            best_storage.pi =
                normalized_pi_from_json(tj, k, "--bayes_opt_truth_json");
        } else {
            // Legacy convention: Q = uniform-exit-rate(true_q_exit_rate),
            // μ = true_simplex_scale × SIMPLEX_4 (k=4, d=3 hardcoded), Σ = I,
            // π = 1/k.
            if (d != 3) {
                std::cerr << "ERROR: --bayes_opt without --bayes_opt_truth_json "
                             "requires d=3 (got d=" << d
                          << "). The simplex μ is hardcoded for (k=4, d=3).\n";
                return 4;
            }
            std::cerr << "true_q_exit_rate   = " << *A.true_q_exit_rate << "\n";
            std::cerr << "true_simplex_scale = " << *A.true_simplex_scale << "\n";
            best_storage.Q = make_uniform_exit_q(k, *A.true_q_exit_rate);
            best_storage.mu.assign(k, Eigen::VectorXd::Zero(d));
            for (std::size_t i = 0; i < k; ++i)
                for (std::size_t j = 0; j < d; ++j)
                    best_storage.mu[i][j] =
                        *A.true_simplex_scale * SIMPLEX_4[i][j];
            best_storage.Sigma.assign(k, Eigen::MatrixXd::Identity(d, d));
            best_storage.pi = Eigen::VectorXd::Constant(
                static_cast<Eigen::Index>(k), 1.0 / static_cast<double>(k));
        }
        best = &best_storage;
    } else {
        records.reserve(2 * static_cast<std::size_t>(n_init));

        // --fix_q_json: load the K×K matrix once and pass to every restart.
        std::optional<Eigen::MatrixXd> fix_q_mat;
        if (A.fix_q_json) {
            std::cerr << "fix_q_json = " << *A.fix_q_json << "\n";
            const truth_json tj = load_truth_json(*A.fix_q_json);
            fix_q_mat = vec2d_to_matrix(tj.Q, k, k, "Q");
            validate_ctmc_q(*fix_q_mat, "--fix_q_json");
        }

        // --init_params_json / --init_emission_json: load once, share across
        // all restarts. mu and covariance are required; Q and pi are optional
        // (when absent, run_em_restart falls back to uniform-exit-rate Q and
        // uniform π respectively). With --n_init > 1, every restart uses the
        // same JSON init.
        const std::optional<std::string> init_json_path =
            A.init_params_json ? A.init_params_json : A.init_emission_json;
        const bool init_json_is_legacy_alias =
            A.init_emission_json.has_value();
        std::optional<init_emission_bundle> init_bundle;
        if (init_json_path) {
            std::cerr << (A.init_params_json
                              ? "init_params_json = "
                              : "init_emission_json = ")
                      << *init_json_path << "\n";
            const truth_json tj = load_truth_json(*init_json_path);
            if (tj.mu.empty())
                throw std::runtime_error(
                    "--init_params_json: required key 'mu' missing or empty");
            if (tj.Sigma.empty() && tj.sigma_diag.empty())
                throw std::runtime_error(
                    "--init_params_json: required key 'Sigma' or 'sigma_diag' "
                    "missing or empty");
            init_emission_bundle b;
            b.mu = vec2d_to_vector_list(tj.mu, k, d, "mu");
            b.Sigma = sigma_from_json(
                tj, k, d, A.diagonal_covariance,
                init_json_is_legacy_alias ? 1e-6 : 0.0,
                A.init_params_json ? "--init_params_json"
                                   : "--init_emission_json");
            if (!tj.Q.empty()) {
                b.Q = vec2d_to_matrix(tj.Q, k, k, "Q");
                validate_ctmc_q(*b.Q,
                                A.init_params_json ? "--init_params_json"
                                                   : "--init_emission_json");
            }
            if (tj.has_pi) {
                b.pi = normalized_pi_from_json(
                    tj, k, A.init_params_json ? "--init_params_json"
                                              : "--init_emission_json");
            }
            init_bundle = std::move(b);
        }
        const init_emission_bundle* init_ptr =
            init_bundle ? &(*init_bundle) : nullptr;

        std::vector<restart_job> jobs;
        jobs.reserve(2 * static_cast<std::size_t>(n_init));
        // k-means++ seeds {42, 43, ...} for reproducibility.
        for (int i = 0; i < n_init; ++i) {
            jobs.push_back(restart_job{
                jobs.size(), "kmeans++ #" + std::to_string(i),
                42 + i, true});
        }
        // random seeds {1000, 1001, ...} for reproducibility.
        for (int i = 0; i < n_init; ++i) {
            jobs.push_back(restart_job{
                jobs.size(), "random #" + std::to_string(i),
                1000 + i, false});
        }

        auto run_job = [&](const restart_job& job) {
            return run_em_restart(trees, observed_data, metadata, k, d,
                                  job.label,
                                  job.seed, job.use_kmeans_pp,
                                  A.kmeans_standard,
                                  A.kmeans_max_iter,
                                  A.kmeans_tol,
                                  fix_pi,
                                  tol, max_iter, q_init_rate,
                                  A.ctmc_weight,
                                  A.emission_weight,
                                  A.fix_q_rate,
                                  A.fix_emission
                                     || A.fix_labels_csv.has_value(),
                                  A.diagonal_covariance,
                                  fix_q_mat,
                                  init_ptr,
                                  A.verbose);
        };

        records.resize(jobs.size());
        if (!A.parallel_restarts || jobs.size() <= 1) {
            for (const auto& job : jobs) {
                const int ordinal =
                    job.use_kmeans_pp
                        ? (job.seed - 42 + 1)
                        : (job.seed - 1000 + 1);
                std::cerr << "  " << (job.use_kmeans_pp ? "kmeans++" : "random  ")
                          << " " << ordinal << "/" << n_init
                          << "  seed=" << job.seed << " ... " << std::flush;
                auto r = run_job(job);
                std::cerr << "log L = " << r.log_likelihood
                          << " (iters=" << r.iterations
                          << ", converged=" << r.converged << ")\n";
                records[job.index] = std::move(r);
            }
        } else {
            const std::size_t worker_count =
                resolve_restart_worker_count(A, jobs.size());
            std::cerr << "Running " << jobs.size()
                      << " restarts with " << worker_count
                      << " parallel workers\n";

            std::atomic<std::size_t> next_job{0};
            std::mutex log_mutex;
            std::exception_ptr first_error = nullptr;
            std::vector<std::thread> workers;
            workers.reserve(worker_count);

            auto worker = [&]() {
                while (true) {
                    const std::size_t j = next_job.fetch_add(1);
                    if (j >= jobs.size()) break;
                    const auto& job = jobs[j];
                    {
                        std::lock_guard<std::mutex> lock(log_mutex);
                        std::cerr << "  " << job.label
                                  << " seed=" << job.seed << " ... started\n";
                    }
                    try {
                        auto r = run_job(job);
                        {
                            std::lock_guard<std::mutex> lock(log_mutex);
                            std::cerr << "  " << job.label
                                      << " seed=" << job.seed
                                      << " ... log L = " << r.log_likelihood
                                      << " (iters=" << r.iterations
                                      << ", converged=" << r.converged << ")\n";
                            records[job.index] = std::move(r);
                        }
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(log_mutex);
                        if (!first_error) first_error = std::current_exception();
                        next_job.store(jobs.size());
                    }
                }
            };

            for (std::size_t i = 0; i < worker_count; ++i)
                workers.emplace_back(worker);
            for (auto& t : workers) t.join();
            if (first_error) std::rethrow_exception(first_error);
        }

        // Best by log_likelihood.
        best = &records[0];
        best_restart_index = 0;
        for (std::size_t i = 1; i < records.size(); ++i) {
            if (records[i].log_likelihood > best->log_likelihood) {
                best = &records[i];
                best_restart_index = i;
            }
        }

        std::cerr << "\nBest: " << best->label << " seed=" << best->seed
                  << " log L = " << best->log_likelihood << "\n";
    }

    // ── Posterior extraction at best params (one extra E-step pass) ────────
    // In --bayes_opt mode this is the ONLY E-step pass. In the EM modes the
    // log L returned here will match best->log_likelihood up to floating drift.
    auto post = compute_posteriors(
        trees, observed_data, metadata, k, d, fix_pi,
        A.diagonal_covariance, A.ctmc_weight, A.emission_weight, *best);

    // ── JSON output ────────────────────────────────────────────────────────
    std::cout.precision(10);
    std::cout << "{\n";
    // log_likelihood: prefer the post-pass value so --bayes_opt mode reports
    // a real number (best->log_likelihood is 0 there). In EM modes this is
    // numerically equivalent to best->log_likelihood up to one E-step's drift.
    std::cout << "  \"log_likelihood\": ";
    dump_json_number(std::cout, post.log_likelihood);
    std::cout << ",\n";
    std::cout << "  \"k\": " << k << ",\n";
    std::cout << "  \"d\": " << d << ",\n";
    std::cout << "  \"n_observed\": " << n_observed << ",\n";
    std::cout << "  \"n_leaves\": " << total_leaves << ",\n";
    std::cout << "  \"n_trees\": " << n_trees << ",\n";
    std::cout << "  \"runtime\": {\"available_cpus\": "
              << detected_cpus
              << ", \"restart_jobs\": " << restart_job_count_for_print
              << ", \"restart_workers\": "
              << restart_worker_count_for_print
              << ", \"restart_worker_mode\": \""
              << restart_worker_mode_for_print << "\"},\n";
    std::cout << "  \"kmeans_init\": {\"mode\": \""
              << (A.kmeans_standard ? "standard" : "seeding")
              << "\", \"lloyd_max_iter\": " << A.kmeans_max_iter
              << ", \"lloyd_tol\": " << A.kmeans_tol
              << "},\n";
    std::cout << "  \"covariance_mode\": \""
              << (A.diagonal_covariance ? "diagonal" : "full") << "\",\n";
    std::cout << "  \"objective_weights\": {\"ctmc\": "
              << A.ctmc_weight << ", \"emission\": "
              << A.emission_weight << "},\n";
    std::cout << "  \"parallel_restarts\": {\"enabled\": "
              << (A.parallel_restarts ? "true" : "false")
              << ", \"threads\": " << A.restart_threads << "},\n";
    std::cout << "  \"best_restart\": ";
    if (best_restart_index) {
        std::cout << "{\"index\": " << *best_restart_index
                  << ", \"label\": \"" << best->label
                  << "\", \"seed\": " << best->seed
                  << ", \"log_likelihood\": ";
        dump_json_number(std::cout, best->log_likelihood);
        std::cout << "},\n";
    } else {
        std::cout << "null,\n";
    }

    // mode: records which mode flags were active so callers can verify the
    // run was parameterized as intended. Objective weights are always shown;
    // other fields are omitted unless the corresponding flag was set.
    std::cout << "  \"mode\": {";
    {
        bool first = true;
        auto sep = [&]() { if (!first) std::cout << ", "; first = false; };
        sep(); std::cout << "\"ctmc_weight\": " << A.ctmc_weight;
        sep(); std::cout << "\"emission_weight\": " << A.emission_weight;
        if (A.fix_q_rate) {
            sep(); std::cout << "\"fix_q_rate\": " << *A.fix_q_rate;
        }
        if (A.fix_q_json) {
            sep();
            std::cout << "\"fix_q_json\": \"" << *A.fix_q_json << "\"";
        }
        if (A.fix_emission || A.fix_labels_csv || A.bayes_opt) {
            sep(); std::cout << "\"fix_emission\": true";
        }
        if (A.fix_labels_csv) {
            sep();
            std::cout << "\"fix_labels_csv\": \"" << *A.fix_labels_csv << "\"";
        }
        if (A.bayes_opt) {
            sep(); std::cout << "\"bayes_opt\": true";
            if (A.bayes_opt_truth_json) {
                sep();
                std::cout << "\"bayes_opt_truth_json\": \""
                          << *A.bayes_opt_truth_json << "\"";
            } else {
                sep(); std::cout << "\"true_q_exit_rate\": "   << *A.true_q_exit_rate;
                sep(); std::cout << "\"true_simplex_scale\": " << *A.true_simplex_scale;
            }
        }
        if (A.init_params_json) {
            sep();
            std::cout << "\"init_params_json\": \""
                      << *A.init_params_json << "\"";
        }
        if (A.init_emission_json) {
            sep();
            std::cout << "\"init_emission_json\": \""
                      << *A.init_emission_json << "\"";
        }
    }
    std::cout << "},\n";
    std::cout << "  \"n_leaves_per_tree\": [";
    for (std::size_t i = 0; i < n_leaves_per_tree.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << n_leaves_per_tree[i];
    }
    std::cout << "],\n";

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

    // ── Per-leaf + internal-node posteriors (normalized probabilities) ────
    // Row order for "posteriors" / "leaf_labels" matches "leaf_names" (which
    // matches taxa_id ordering). Internal-node rows are unordered w.r.t. the
    // tree; callers must use "internal_node_ids" to map row → phylogeny node.
    std::cout << "  \"leaf_names\": [";
    for (std::size_t i = 0; i < metadata.taxa_names.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "\"" << metadata.taxa_names[i] << "\"";
    }
    std::cout << "],\n";

    std::cout << "  \"posteriors\": [";
    for (std::size_t i = 0; i < post.leaf_posteriors.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "[";
        for (std::size_t j = 0; j < k; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << post.leaf_posteriors[i][j];
        }
        std::cout << "]";
    }
    std::cout << "],\n";

    std::cout << "  \"leaf_labels\": [";
    for (std::size_t i = 0; i < post.leaf_labels.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << post.leaf_labels[i];
    }
    std::cout << "],\n";

    std::cout << "  \"tree_index_per_leaf\": [";
    for (std::size_t i = 0; i < tree_index_per_leaf.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << tree_index_per_leaf[i];
    }
    std::cout << "],\n";

    std::cout << "  \"internal_node_posteriors\": [";
    for (std::size_t i = 0; i < post.internal_posteriors.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "[";
        for (std::size_t j = 0; j < k; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << post.internal_posteriors[i][j];
        }
        std::cout << "]";
    }
    std::cout << "],\n";

    std::cout << "  \"internal_node_ids\": [";
    for (std::size_t i = 0; i < post.internal_node_ids.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << post.internal_node_ids[i];
    }
    std::cout << "],\n";

    std::cout << "  \"tree_index_per_internal\": [";
    for (std::size_t i = 0; i < post.tree_index_per_internal.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << post.tree_index_per_internal[i];
    }
    std::cout << "],\n";

    std::cout << "  \"expected_transitions\": ";
    dump_matrix(std::cout, post.expected_transitions);
    std::cout << ",\n";

    std::cout << "  \"expected_dwell_times\": ";
    dump_vec(std::cout, post.expected_dwell_times);
    std::cout << ",\n";

    std::cout << "  \"per_restart\": [\n";
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        const bool is_best = best_restart_index && *best_restart_index == i;
        if (i > 0) std::cout << ",\n";
        std::cout << "    {\"label\":\"" << r.label
                  << "\",\"seed\":" << r.seed
                  << ",\"initial_log_likelihood\":";
        dump_json_number(std::cout, r.initial_log_likelihood);
        std::cout << ",\"log_likelihood\":";
        dump_json_number(std::cout, r.log_likelihood);
        std::cout << ",\"iterations\":" << r.iterations
                  << ",\"converged\":" << (r.converged ? "true" : "false")
                  << ",\"is_best\":" << (is_best ? "true" : "false");
        std::cout << ",\"log_likelihood_trace\":";
        dump_double_vector(std::cout, r.log_likelihood_trace);
        std::cout << ",\"complete_data_objective_trace\":";
        dump_complete_data_objective_trace(
            std::cout, r.complete_data_objective_trace);
        std::cout << ",\"initial\":{\"log_likelihood\":";
        dump_json_number(std::cout, r.initial_log_likelihood);
        std::cout << ",\"Q\":";
        dump_matrix(std::cout, r.initial_Q);
        std::cout << ",\"mu\":";
        dump_vector_list(std::cout, r.initial_mu);
        std::cout << ",\"Sigma\":";
        dump_matrix_list(std::cout, r.initial_Sigma);
        std::cout << ",\"pi\":";
        dump_vec(std::cout, r.initial_pi);
        std::cout << "}";
        std::cout << ",\"final\":{\"log_likelihood\":";
        dump_json_number(std::cout, r.log_likelihood);
        std::cout << ",\"Q\":";
        dump_matrix(std::cout, r.Q);
        std::cout << ",\"mu\":";
        dump_vector_list(std::cout, r.mu);
        std::cout << ",\"Sigma\":";
        dump_matrix_list(std::cout, r.Sigma);
        std::cout << ",\"pi\":";
        dump_vec(std::cout, r.pi);
        std::cout << "}}";
    }
    std::cout << "\n  ]\n";
    std::cout << "}\n";
    return 0;
}

int main(int argc, char* argv[]) {
    return main_impl(argc, argv);
}
