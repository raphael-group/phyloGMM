#ifndef PHYLOCORE_OPTIMIZE_DIRECT_H
#define PHYLOCORE_OPTIMIZE_DIRECT_H

// Header-only. Including this header pulls in the IPOPT C++ API. Your
// executable must:
//   1. Have IPOPT headers on its include path (e.g. `pkg-config --cflags ipopt`
//      adds `-I.../coin-or`). The IPOPT headers below are referenced WITHOUT
//      a `coin-or/` prefix because pkg-config exposes them on the include
//      path directly — matches LAML-Pro.
//   2. Link IPOPT (`pkg-config --libs ipopt` → `-lipopt`).
// Users who don't include this header don't need IPOPT at all.

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include "IpTNLP.hpp"

#include "estimation_results.h"
#include "joint_optimization_problem.h"

namespace phylocore {
namespace optimize {

/*!
 * @brief Options for optimize::direct.
 */
struct direct_options {
    double tol            = 1e-6;
    int    max_iter       = 200;
    bool   verbose        = false;
    double finite_diff_eps = 1e-6;   // step size for finite-diff gradient fallback

    // If true, falls back to central-difference finite-diff gradient when
    // problem.gradient() returns empty. Default: false — an analytic
    // gradient is required.
    //
    // Set true ONLY when prototyping a new model. Finite-diff has subtle
    // failure modes — e.g., the gtr_blens convergence failure caught during
    // phyloGMM characterization, where central differences with eps=1e-6
    // misbehaved as branches approached the 1e-8 lower bound, producing
    // gradients that poisoned IPOPT's quasi-Newton Hessian and drove the
    // objective in the wrong direction.
    //
    // For models with branch-length parameters, the canonical analytic
    // gradient lives in optimize::branch_length_gradient (header
    // optimize/branch_gradient.h).
    bool   finite_diff_fallback = false;
};

namespace detail {

/*!
 * @brief Internal IPOPT TNLP adapter. Templated on the user's problem type.
 *        Inline-defined since the whole module is header-only.
 *
 * Translates a joint_optimization_problem subclass into IPOPT's TNLP contract:
 *   - eval_f / eval_grad_f → problem.log_likelihood() / problem.gradient()
 *     (with finite-diff fallback when gradient() returns empty)
 *   - get_bounds_info → problem.bounds()
 *   - get_starting_point → problem.get_params()
 *   - Constraint callbacks are stubbed out for now (problem.equality_constraints
 *     and inequality_constraints are placeholder types). Will be wired in when
 *     the constraint helpers are implemented.
 *
 * IPOPT minimizes; we maximize log-likelihood; the adapter negates throughout.
 *
 * Uses limited-memory quasi-Newton Hessian approximation by default
 * (configured in optimize::direct), so eval_h returns false.
 *
 * If problem.set_params throws invalid_parameters_error during evaluation,
 * eval_f / eval_grad_f return false and IPOPT backtracks.
 */
template <typename Problem>
class TNLP_adapter : public Ipopt::TNLP {
public:
    TNLP_adapter(Problem& problem, const direct_options& opts)
        : problem_(problem), opts_(opts) {}

    bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                      Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag,
                      Ipopt::TNLP::IndexStyleEnum& index_style) override {
        auto params = problem_.get_params();
        n           = static_cast<Ipopt::Index>(params.size());
        m           = 0;  // TODO: wire equality/inequality_constraints later
        nnz_jac_g   = 0;
        nnz_h_lag   = 0;  // quasi-Newton Hessian approximation
        index_style = Ipopt::TNLP::C_STYLE;
        return true;
    }

    bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                         Ipopt::Index /*m*/, Ipopt::Number* /*g_l*/,
                         Ipopt::Number* /*g_u*/) override {
        constexpr Ipopt::Number IPOPT_INF = 2e19;
        auto bounds = problem_.bounds();
        if (bounds.empty()) {
            for (Ipopt::Index i = 0; i < n; ++i) { x_l[i] = -IPOPT_INF; x_u[i] = IPOPT_INF; }
        } else {
            if (static_cast<Ipopt::Index>(bounds.size()) != n) {
                throw std::runtime_error(
                    "optimize::direct: bounds().size() != get_params().size()");
            }
            for (Ipopt::Index i = 0; i < n; ++i) {
                x_l[i] = bounds[i].first;
                x_u[i] = bounds[i].second;
            }
        }
        return true;
    }

    bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                            bool /*init_z*/, Ipopt::Number* /*z_L*/,
                            Ipopt::Number* /*z_U*/, Ipopt::Index /*m*/,
                            bool /*init_lambda*/,
                            Ipopt::Number* /*lambda*/) override {
        if (init_x) {
            auto params = problem_.get_params();
            for (Ipopt::Index i = 0; i < n; ++i) x[i] = params[i];
        }
        return true;
    }

    bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                Ipopt::Number& obj_value) override {
        if (new_x && !set_params_from_x_(x, n)) return false;
        obj_value = -problem_.log_likelihood();   // IPOPT minimizes
        return true;
    }

    bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                     Ipopt::Number* grad_f) override {
        if (new_x && !set_params_from_x_(x, n)) return false;

        auto grad = problem_.gradient();
        if (!grad.empty()) {
            if (static_cast<Ipopt::Index>(grad.size()) != n) {
                throw std::runtime_error(
                    "optimize::direct: gradient().size() != get_params().size()");
            }
            for (Ipopt::Index i = 0; i < n; ++i) grad_f[i] = -grad[i];
            return true;
        }

        if (!opts_.finite_diff_fallback) {
            throw std::runtime_error(
                "optimize::direct: gradient() returned empty and "
                "finite_diff_fallback is false; provide an analytic "
                "gradient (e.g., via optimize::branch_length_gradient)");
        }

        // Finite-difference fallback (central differences).
        auto center = problem_.get_params();
        const double eps = opts_.finite_diff_eps;
        for (Ipopt::Index i = 0; i < n; ++i) {
            auto perturbed = center;
            perturbed[i] = center[i] + eps;
            if (!try_set_params_(perturbed)) return false;
            double f_plus = -problem_.log_likelihood();

            perturbed[i] = center[i] - eps;
            if (!try_set_params_(perturbed)) return false;
            double f_minus = -problem_.log_likelihood();

            grad_f[i] = (f_plus - f_minus) / (2.0 * eps);
        }
        // Restore original parameters.
        if (!try_set_params_(center)) return false;
        return true;
    }

    bool eval_g(Ipopt::Index /*n*/, const Ipopt::Number* /*x*/, bool /*new_x*/,
                Ipopt::Index /*m*/, Ipopt::Number* /*g*/) override {
        return true;  // no constraints in first cut
    }

    bool eval_jac_g(Ipopt::Index /*n*/, const Ipopt::Number* /*x*/, bool /*new_x*/,
                    Ipopt::Index /*m*/, Ipopt::Index /*nele_jac*/,
                    Ipopt::Index* /*iRow*/, Ipopt::Index* /*jCol*/,
                    Ipopt::Number* /*values*/) override {
        return true;  // no constraints in first cut
    }

    bool eval_h(Ipopt::Index /*n*/, const Ipopt::Number* /*x*/, bool /*new_x*/,
                Ipopt::Number /*obj_factor*/, Ipopt::Index /*m*/,
                const Ipopt::Number* /*lambda*/, bool /*new_lambda*/,
                Ipopt::Index /*nele_hess*/, Ipopt::Index* /*iRow*/,
                Ipopt::Index* /*jCol*/, Ipopt::Number* /*values*/) override {
        return false;  // use quasi-Newton Hessian approximation
    }

    void finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n,
                           const Ipopt::Number* x, const Ipopt::Number* /*z_L*/,
                           const Ipopt::Number* /*z_U*/, Ipopt::Index /*m*/,
                           const Ipopt::Number* /*g*/,
                           const Ipopt::Number* /*lambda*/,
                           Ipopt::Number obj_value,
                           const Ipopt::IpoptData* /*ip_data*/,
                           Ipopt::IpoptCalculatedQuantities* /*ip_cq*/) override {
        final_params_.assign(x, x + n);
        final_log_likelihood_ = -obj_value;
        solver_status_        = status;
    }

    // ── Accessors for the wrapper to retrieve results ────────────────────────
    const std::vector<double>& final_params()         const { return final_params_; }
    double                     final_log_likelihood() const { return final_log_likelihood_; }
    Ipopt::SolverReturn        solver_status()        const { return solver_status_; }

private:
    Problem&                problem_;
    const direct_options&   opts_;
    std::vector<double>     final_params_;
    double                  final_log_likelihood_ = 0.0;
    Ipopt::SolverReturn     solver_status_        = Ipopt::INTERNAL_ERROR;

    // Set params from IPOPT's x vector. Returns false on
    // invalid_parameters_error (IPOPT then backtracks).
    bool set_params_from_x_(const Ipopt::Number* x, Ipopt::Index n) {
        std::vector<double> params(x, x + n);
        return try_set_params_(params);
    }

    bool try_set_params_(const std::vector<double>& params) {
        try {
            problem_.set_params(params);
        } catch (const invalid_parameters_error&) {
            return false;
        }
        return true;
    }
};

} // namespace detail

/*!
 * @brief Direct MLE via IPOPT.
 *
 * Maximizes problem.log_likelihood() over problem.get_params(). Supports
 * problem.bounds() and requires problem.gradient() to return a non-empty
 * analytic gradient — falling back to central-difference finite-diff only
 * when opts.finite_diff_fallback is explicitly set to true (intended for
 * prototyping; finite-diff has subtle failure modes near parameter
 * bounds).
 *
 * On return, the problem is left at the optimized parameter state.
 *
 * @throws std::runtime_error if IPOPT fails to initialize, if
 *         problem.gradient() returns empty while opts.finite_diff_fallback
 *         is false, or if IPOPT terminates abnormally (status worse than
 *         Solved_To_Acceptable_Level).
 */
template <typename Problem>
estimation_results direct(Problem& problem, direct_options opts = {}) {
    using namespace Ipopt;

    // Guard against overlapping taxa across trees — the inside pass would
    // otherwise double-count the shared observations and produce wrong
    // likelihoods / gradients without any warning.
    problem.validate_disjoint_taxa();

    SmartPtr<detail::TNLP_adapter<Problem>> adapter =
        new detail::TNLP_adapter<Problem>(problem, opts);
    SmartPtr<TNLP> tnlp = adapter;

    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    // Gradient-based scaling normalizes the objective gradient so the L-BFGS
    // initial step (no curvature info, identity Hessian) does not overshoot
    // when ||grad f|| is large in absolute terms — critical for likelihood
    // surfaces where the gradient can easily reach O(10^2)-O(10^3).
    // Drop nlp_scaling_max_gradient from its 100 default to 1 so the
    // scaling kicks in even for moderately large gradients.
    app->Options()->SetStringValue("nlp_scaling_method", "gradient-based");
    app->Options()->SetNumericValue("nlp_scaling_max_gradient", 1.0);
    app->Options()->SetNumericValue("tol", opts.tol);
    app->Options()->SetIntegerValue("max_iter", opts.max_iter);
    if (!opts.verbose) {
        app->Options()->SetIntegerValue("print_level", 0);
        app->Options()->SetStringValue("sb", "yes");  // suppress IPOPT banner
    }

    ApplicationReturnStatus init_status = app->Initialize();
    if (init_status != Solve_Succeeded) {
        throw std::runtime_error("optimize::direct: IPOPT failed to initialize");
    }

    ApplicationReturnStatus solve_status = app->OptimizeTNLP(tnlp);

    estimation_results results;
    results.params         = adapter->final_params();
    results.log_likelihood = adapter->final_log_likelihood();
    auto stats = app->Statistics();
    results.num_iterations = IsValid(stats) ? stats->IterationCount() : 0;
    results.converged      = (solve_status == Solve_Succeeded);

    // Restore the problem to the optimized state so that posterior accessors
    // and downstream consumers see the optimum.
    if (!results.params.empty()) {
        problem.set_params(results.params);
        problem.log_likelihood();
    }

    // Snapshot the trees (copies, not pointers).
    results.trees.reserve(problem.trees.size());
    for (const auto* t : problem.trees) results.trees.push_back(*t);

    if (solve_status != Solve_Succeeded &&
        solve_status != Solved_To_Acceptable_Level) {
        throw std::runtime_error(
            "optimize::direct: IPOPT terminated with status " +
            std::to_string(static_cast<int>(solve_status)));
    }

    return results;
}

} // namespace optimize
} // namespace phylocore

#endif // PHYLOCORE_OPTIMIZE_DIRECT_H
