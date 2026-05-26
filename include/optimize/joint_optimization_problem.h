#ifndef PHYLOCORE_OPTIMIZE_JOINT_OPTIMIZATION_PROBLEM_H
#define PHYLOCORE_OPTIMIZE_JOINT_OPTIMIZATION_PROBLEM_H

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "../meta_data.h"
#include "../phylogeny.h"
#include "../state_space.h"

namespace phylocore {
namespace optimize {

/*!
 * @brief Exception thrown by set_params when a parameter vector violates a
 *        model invariant (e.g., probabilities don't sum to 1, covariance not
 *        positive semi-definite, etc.).
 *
 * IPOPT's C++ API translates exceptions during evaluation callbacks into
 * "evaluation failed" returns so the optimizer can backtrack to a feasible
 * point instead of crashing.
 */
class invalid_parameters_error : public std::runtime_error {
public:
    static constexpr std::size_t NO_INDEX = static_cast<std::size_t>(-1);

    explicit invalid_parameters_error(const std::string& msg,
                                      std::size_t parameter_index = NO_INDEX)
        : std::runtime_error(msg), parameter_index_(parameter_index) {}

    std::size_t parameter_index() const noexcept { return parameter_index_; }

private:
    std::size_t parameter_index_;
};

// Placeholder constraint types. Concrete signatures will be defined when
// optimize::direct is implemented and the IPOPT translation layer is built.
// Higher-level constraint helpers (ultrametric_constraint, simplex_constraint,
// min_value_constraint, ...) will produce values of these types.
struct equality_constraint   {};
struct inequality_constraint {};

/*!
 * @brief Abstract base for an optimization problem over (model, observations, trees).
 *
 * Supports a set of trees that share model parameters (e.g., a single Q across
 * loci, or shared ν/φ across replicates) while possibly having independent
 * per-tree branch lengths. Single-tree is the size-1 special case.
 *
 * The user provides pointers to the model components (stm, obs), the global
 * observation data (observed_data, metadata), and the per-tree state (trees,
 * node_data). The user does NOT allocate any likelihood buffers — the library
 * lazily allocates them internally on first use.
 *
 * Required overrides:
 *   - get_params() : pack the currently-being-estimated parameters into a flat
 *                    vector. Things held fixed do NOT appear in this vector.
 *   - set_params() : unpack the flat vector back into model state; refresh any
 *                    caches (typically by calling stm->build() / obs->build()).
 *                    Throw invalid_parameters_error on invalid input.
 *
 * Optional overrides (library defaults provided):
 *   - log_likelihood()           — defaults to summed inside log-Ls across trees.
 *   - bounds()                   — defaults to unbounded (empty vector).
 *   - gradient()                 — defaults to empty (caller uses finite-diff).
 *   - equality_constraints()     — defaults to none.
 *   - inequality_constraints()   — defaults to none.
 *
 * For EM, the user additionally adds (directly on their subclass — Pattern B
 * duck typing, NOT via inheritance from em_extension):
 *   - using suff_stats_type      = ...   // model-defined
 *   - suff_stats_type compute_suff_stats(self, const standard_posteriors&)
 *   - void              m_step(self, const suff_stats_type&)
 *
 * See docs/parameter_estimation.md for the full contract and the five
 * optimization modes (Direct MLE finite-diff / analytic, EM joint M-step
 * finite-diff / analytic, EM closed-form M-step).
 *
 * @tparam Stm       Concrete state_transition_model subclass.
 * @tparam ObsModel  Concrete observation_model subclass.
 * @tparam O         Per-leaf observation type (e.g., std::vector<int> for
 *                   discrete, Eigen::VectorXd for continuous).
 * @tparam D         Per-node data type. Defaults to std::monostate for models
 *                   that need no per-node state.
 */
template <typename Stm,
          typename ObsModel,
          typename O,
          typename D = std::monostate>
class joint_optimization_problem {
public:
    // ── User-provided (the library does not own; do not free) ────────────────

    Stm*      stm = nullptr;
    ObsModel* obs = nullptr;

    // Global, shared across all trees. The inside-likelihood code looks up
    // each leaf's observation via the taxa_id stored in the digraph node's
    // data field — decoupled from tree-local node_id. So a single
    // observed_data and a single metadata cover the whole dataset, including
    // cases where leaves move between trees during topology search.
    const std::vector<O>* observed_data = nullptr;
    const meta_data*      metadata      = nullptr;

    // Per-tree state (indices kept in sync across these two vectors).
    std::vector<tree*>             trees;
    std::vector<std::vector<D>*>   node_data;

    virtual ~joint_optimization_problem() = default;

    // ── REQUIRED overrides ───────────────────────────────────────────────────

    /*!
     * @brief Pack the parameters being estimated into a flat vector.
     *
     * For a multi-parameter / multi-tree problem the user is responsible for
     * the layout: e.g., [Q_off_diag..., μ..., Σ..., π..., blens_tree_0...,
     * blens_tree_1...]. Things held fixed are NOT included.
     */
    virtual std::vector<double> get_params() const = 0;

    /*!
     * @brief Unpack the flat vector back into model state.
     *
     * Must refresh any caches (typically by calling stm->build() and/or
     * obs->build()). Throws invalid_parameters_error if the parameters violate
     * a model invariant; the optimizer translates this into an "evaluation
     * failed" return and may backtrack.
     */
    virtual void set_params(const std::vector<double>& params) = 0;

    // ── OPTIONAL overrides (library defaults) ────────────────────────────────

    /*!
     * @brief Total log-likelihood. Default: sum, over each tree, of the
     *        per-site inside log-likelihoods.
     *
     * Override to add prior terms (MAP-style estimation) or regularizers.
     */
    virtual double log_likelihood() {
        ensure_inside_buffers_allocated_();
        double total = 0.0;
        for (std::size_t i = 0; i < trees.size(); ++i) {
            auto per_site = phylocore::compute_inside_log_likelihood(
                *stm, *obs, *observed_data, *metadata,
                *trees[i], *inside_buffers_[i], *node_data[i]);
            for (double v : per_site) total += v;
        }
        return total;
    }

    /*!
     * @brief Per-parameter (lower, upper) bounds. Default: empty (unbounded).
     *
     * If non-empty, size must equal get_params().size().
     */
    virtual std::vector<std::pair<double, double>> bounds() const { return {}; }

    /*!
     * @brief Analytic gradient of log_likelihood w.r.t. get_params(). Default:
     *        empty — callers (optimize::direct) interpret empty as "use
     *        finite-difference gradient." Overriding lifts the problem from
     *        mode #1/#3 (slow finite-diff) to mode #2/#4 (fast analytic).
     */
    virtual std::vector<double> gradient() { return {}; }

    virtual std::vector<equality_constraint>   equality_constraints()   const { return {}; }
    virtual std::vector<inequality_constraint> inequality_constraints() const { return {}; }

protected:
    // ── Library-internal buffer access (for use by optimize:: algorithms) ────
    //
    // The inside buffer is lazily allocated on first log_likelihood() call.
    // Edge-inside and outside buffers will be allocated on demand by
    // optimize::em (or any algorithm that needs them); we defer allocating
    // them here to avoid paying the memory cost for Direct-MLE-only problems.

    likelihood_buffer& inside_buffer(std::size_t tree_idx) {
        ensure_inside_buffers_allocated_();
        return *inside_buffers_[tree_idx];
    }

    std::size_t num_trees() const noexcept { return trees.size(); }

private:
    std::vector<std::unique_ptr<likelihood_buffer>> inside_buffers_;

    void ensure_inside_buffers_allocated_() {
        if (inside_buffers_.size() == trees.size()) return;
        inside_buffers_.clear();
        if (trees.empty()) return;

        const std::size_t num_sites = stm->sites_latent_states.size();
        const std::size_t max_alpha = std::max_element(
            stm->sites_latent_states.begin(),
            stm->sites_latent_states.end(),
            [](const state_space& a, const state_space& b) {
                return a.alphabet_size < b.alphabet_size;
            })->alphabet_size;

        inside_buffers_.reserve(trees.size());
        for (tree* t : trees) {
            inside_buffers_.push_back(std::make_unique<likelihood_buffer>(
                num_sites, max_alpha, t->num_nodes));
        }
    }
};

} // namespace optimize
} // namespace phylocore

#endif // PHYLOCORE_OPTIMIZE_JOINT_OPTIMIZATION_PROBLEM_H
