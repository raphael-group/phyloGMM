#ifndef PHYLOCORE_OPTIMIZE_EM_H
#define PHYLOCORE_OPTIMIZE_EM_H

// Header-only. No IPOPT dependency. The EM driver is purely the convergence
// loop: it orchestrates inside/edge_inside/outside passes, then defers to the
// user's compute_suff_stats() and m_step() methods (Pattern B duck typing —
// the user adds these directly to their joint_optimization_problem subclass;
// there is no separate em_extension class to inherit from).
//
// Modes #3 / #4 (EM with iterative joint M-step) call optimize::direct
// internally from the user's m_step; including those incurs an IPOPT
// dependency at the consumer-executable level. Modes #5 (EM with closed-form
// M-step, e.g. phyloGMM) requires no IPOPT — only this header is needed.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "../phylogeny.h"
#include "estimation_results.h"

namespace phylocore {
namespace optimize {

/*!
 * @brief Options for optimize::em.
 *
 * Multi-restart support is intentionally not exposed in this first cut. If
 * you want multi-restart, wrap optimize::em(problem) in a user-controlled
 * loop and track the best result yourself. See the deferred TODO in
 * docs/parameter_estimation.md.
 */
struct em_options {
    // Relative log-L improvement threshold. EM stops when
    //   (new_llh - prev_llh) / |prev_llh| < rel_tol.
    // Matches LAML-Pro's convergence criterion (laml_em.cxx:1186). Using a
    // relative tol scales sensibly with dataset size: tighter for small log-L
    // problems, looser for large ones. The old absolute-tol default of 1e-6
    // was equivalent to rel_tol ≈ 1e-6/|log-L|, which became hopelessly tight
    // for any non-trivial dataset (e.g., 5600× tighter than LAML-Pro on a
    // 1024-leaf k=4 fixture, where log-L ≈ -558).
    double rel_tol           = 1e-5;
    int    max_iter          = 100;
    bool   verbose           = false;

    // EM theoretically guarantees log-likelihood is non-decreasing each
    // iteration. Numerically it can decrease — usually a sign that
    // compute_suff_stats and m_step disagree about which parameters are
    // being updated, or that an inner finite-diff solver in m_step
    // backtracked too far. When true, emit a warning to stderr.
    bool   warn_on_decrease  = true;
};

/*!
 * @brief EM convergence loop.
 *
 * Per iteration:
 *   1. Run edge_inside and outside passes for each tree (using buffers owned
 *      by the joint_optimization_problem). The inside pass + per-tree
 *      log_total_likelihood are populated by problem.log_likelihood().
 *   2. Call problem.compute_suff_stats() — user override that aggregates
 *      library-generic posteriors into model-specific sufficient statistics.
 *   3. Call problem.m_step(suff_stats) — user override that updates
 *      get_params() parameters (closed-form OR sub-call to optimize::direct).
 *   4. Recompute log-likelihood; check convergence on |Δlog-L| < tol.
 *
 * On return:
 *   - problem is left at whatever parameter state the final m_step produced
 *     (we do NOT track a "best so far" across iterations — log-L should be
 *     approximately monotone; warn_on_decrease surfaces deviations).
 *   - estimation_results.params reflects problem.get_params() at exit.
 *   - estimation_results.trees holds snapshot copies of problem.trees at exit.
 *
 * The Problem type is duck-typed:
 *   - Must inherit phylocore::optimize::joint_optimization_problem<...>
 *   - Must have `compute_suff_stats()` returning some user-defined type
 *   - Must have `m_step(const suff_stats&)`
 * Missing methods produce a template instantiation error.
 */
template <typename Problem>
estimation_results em(Problem& p, em_options opts = {}) {
    // Guard against overlapping taxa across trees — silent over-weighting of
    // shared observations in suff_stats would otherwise produce wrong results.
    p.validate_disjoint_taxa();

    // Initial log-likelihood. Also populates the per-tree inside buffer and
    // log_total_likelihood_per_tree_ on the problem (consumed by the edge /
    // outside passes and posterior accessors below).
    double prev_llh = p.log_likelihood();

    int  iter_count = 0;
    bool converged  = false;

    for (int iter = 0; iter < opts.max_iter; ++iter) {
        // E-step: edge_inside + outside passes for each tree. (Inside was
        // populated by the most recent log_likelihood() call.)
        for (std::size_t i = 0; i < p.trees.size(); ++i) {
            phylocore::compute_edge_inside_log_likelihood(
                *p.stm, *p.trees[i],
                p.inside_buffer(i), p.edge_inside_buffer(i),
                *p.node_data[i]);
            phylocore::compute_outside_log_likelihood(
                *p.stm, *p.trees[i],
                p.edge_inside_buffer(i), p.outside_buffer(i),
                *p.node_data[i]);
        }

        // User-overridden aggregation of library-generic posteriors into
        // model-specific sufficient statistics.
        auto suff = p.compute_suff_stats();

        // User-overridden parameter update.
        p.m_step(suff);

        // Recompute log-likelihood at new parameters. This also re-runs the
        // inside pass so the next iteration's E-step has fresh buffers.
        double new_llh = p.log_likelihood();
        ++iter_count;

        if (opts.verbose) {
            std::cerr << "[optimize::em] iter " << iter_count
                      << ": log L = " << new_llh
                      << "  (delta = " << (new_llh - prev_llh) << ")\n";
        }

        if (opts.warn_on_decrease && new_llh < prev_llh - 1e-9) {
            std::cerr << "[optimize::em] WARNING: log-L decreased at iter "
                      << iter_count << ": " << prev_llh
                      << " -> " << new_llh
                      << "  (delta = " << (new_llh - prev_llh) << ")\n";
        }

        // Relative-improvement convergence test. prev_llh is finite-negative
        // for any non-trivial dataset, so |prev_llh| > 0. We require a
        // *non-negative* improvement smaller than rel_tol — i.e. the iterate
        // genuinely plateaued. Non-positive deltas keep iterating (with
        // warn_on_decrease above surfacing the regression) until either a
        // genuine plateau or max_iter caps the loop. The earlier behaviour
        // (rel_delta < rel_tol for any sign) silently declared early-stop on
        // the first step that overshot, producing fits that were not at a
        // local optimum.
        double rel_delta = (new_llh - prev_llh) / std::abs(prev_llh);
        if (rel_delta >= 0.0 && rel_delta < opts.rel_tol) {
            prev_llh   = new_llh;
            converged  = true;
            break;
        }
        prev_llh = new_llh;
    }

    estimation_results results;
    results.params         = p.get_params();
    results.log_likelihood = prev_llh;
    results.num_iterations = iter_count;
    results.converged      = converged;
    results.trees.reserve(p.trees.size());
    for (const auto* t : p.trees) results.trees.push_back(*t);
    return results;
}

} // namespace optimize
} // namespace phylocore

#endif // PHYLOCORE_OPTIMIZE_EM_H
