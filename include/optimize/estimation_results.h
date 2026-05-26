#ifndef PHYLOCORE_OPTIMIZE_ESTIMATION_RESULTS_H
#define PHYLOCORE_OPTIMIZE_ESTIMATION_RESULTS_H

#include <vector>

#include "../phylogeny.h"

namespace phylocore {
namespace optimize {

/*!
 * @brief Output of a single optimization run.
 *
 * Returned by optimize::direct (IPOPT-based MLE) and optimize::em
 * (EM convergence loop). Pure data carrier — no methods, no invariants
 * beyond field semantics.
 *
 * The trees vector holds snapshot copies (not pointers) of the trees at
 * the optimized parameters, so results outlive any subsequent mutation
 * of the original joint_optimization_problem.
 *
 * Posteriors are intentionally NOT exposed here. After optimize::em /
 * optimize::direct returns, the underlying joint_optimization_problem
 * is left at the final parameter state; users access posteriors directly
 * via problem.node_posterior(tree_idx, node_id, site_c) (and the related
 * edge_posterior / root_posterior / leaf_posterior accessors) rather than
 * duplicating that machinery on the results object.
 */
struct estimation_results {
    /*!
     * @brief Optimized parameter vector. Matches the layout the user's
     *        get_params() / set_params() agreed on.
     */
    std::vector<double> params;

    /*!
     * @brief Final log-likelihood value at the optimized parameters.
     */
    double log_likelihood;

    /*!
     * @brief Number of iterations taken by the optimizer.
     */
    int num_iterations;

    /*!
     * @brief True if the optimizer converged; false if it hit max_iter
     *        (or otherwise terminated without satisfying the convergence
     *        criterion).
     */
    bool converged;

    /*!
     * @brief Snapshot copies of the trees at the optimized parameters
     *        (branch lengths and topology). Copies — not pointers — so
     *        results remain valid after the source problem is mutated
     *        or destroyed.
     */
    std::vector<phylocore::tree> trees;
};

} // namespace optimize
} // namespace phylocore

#endif // PHYLOCORE_OPTIMIZE_ESTIMATION_RESULTS_H
