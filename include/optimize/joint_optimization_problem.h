#ifndef PHYLOCORE_OPTIMIZE_JOINT_OPTIMIZATION_PROBLEM_H
#define PHYLOCORE_OPTIMIZE_JOINT_OPTIMIZATION_PROBLEM_H

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
 * @brief Abstract base encapsulating everything one optimization run needs.
 *
 * Two roles in one object:
 *
 * 1. User-facing contract (the user subclasses and supplies):
 *      Required: get_params(), set_params()
 *      Optional: log_likelihood(), bounds(), gradient(),
 *                equality_constraints(), inequality_constraints()
 *      For EM:   using suff_stats_type = ...; compute_suff_stats(self),
 *                m_step(self, suff_stats)   (added directly to the subclass;
 *                                            Pattern B duck typing — no
 *                                            inheritance from em_extension)
 *
 * 2. Library-managed E-step state:
 *      Per-tree likelihood_buffers for inside, edge_inside, outside passes
 *      (lazily allocated, sized per-tree).
 *      Per-tree log_total_likelihood (populated by log_likelihood()).
 *      Lazy posterior accessors — node_posterior, edge_posterior,
 *      root_posterior, leaf_posterior — that compute on demand from the
 *      buffers. These are what the user's compute_suff_stats reads from.
 *
 * Supports a set of trees that share model parameters (e.g., a single Q
 * across loci, or shared ν/φ across replicates) while possibly having
 * independent per-tree branch lengths. Single-tree is the size-1 special
 * case. The inside-likelihood code looks up each leaf's observation via the
 * taxa_id stored in the digraph node's data field — decoupled from
 * tree-local node_id — so observed_data and metadata are single global
 * vectors, not per-tree.
 *
 * See docs/parameter_estimation.md for the five optimization modes and
 * the full design rationale.
 *
 * @tparam Stm       Concrete state_transition_model subclass.
 * @tparam ObsModel  Concrete observation_model subclass.
 * @tparam O         Per-leaf observation type (e.g., std::vector<int> for
 *                   discrete, Eigen::VectorXd for continuous).
 * @tparam D         Per-node data type. Defaults to std::monostate for
 *                   models that need no per-node state.
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

    const std::vector<O>* observed_data = nullptr;
    const meta_data*      metadata      = nullptr;

    std::vector<tree*>             trees;
    std::vector<std::vector<D>*>   node_data;

    virtual ~joint_optimization_problem() = default;

    // ── REQUIRED overrides ───────────────────────────────────────────────────

    /*!
     * @brief Pack the parameters being estimated into a flat vector.
     *
     * For multi-parameter / multi-tree problems the user controls the layout
     * (e.g., [Q_off_diag..., μ..., Σ..., π..., blens_tree_0...,
     * blens_tree_1...]). Things held fixed are NOT included.
     */
    virtual std::vector<double> get_params() const = 0;

    /*!
     * @brief Unpack the flat vector back into model state.
     *
     * Must refresh any caches (typically by calling stm->build() and/or
     * obs->build()). Throws invalid_parameters_error if the parameters
     * violate a model invariant; the optimizer translates this into an
     * "evaluation failed" return and may backtrack.
     */
    virtual void set_params(const std::vector<double>& params) = 0;

    // ── OPTIONAL overrides (library defaults) ────────────────────────────────

    /*!
     * @brief Total log-likelihood. Default: sum, over each tree, of the
     *        per-site inside log-likelihoods. Also populates per-tree
     *        log_total_likelihood storage as a side effect (consumed by
     *        posterior accessors).
     *
     * Override to add prior terms (MAP-style estimation) or regularizers —
     * but if you override, call this base implementation first (or otherwise
     * populate log_total_L_per_tree_) so posterior accessors keep working.
     */
    virtual double log_likelihood() {
        ensure_inside_buffers_allocated_();
        log_p_y_per_tree_per_site_.assign(trees.size(), {});
        double total = 0.0;
        for (std::size_t i = 0; i < trees.size(); ++i) {
            auto per_site = phylocore::compute_inside_log_likelihood(
                *stm, *obs, *observed_data, *metadata,
                *trees[i], *inside_buffers_[i], *node_data[i]);
            for (double v : per_site) total += v;
            log_p_y_per_tree_per_site_[i] = std::move(per_site);
        }
        return total;
    }

    /*!
     * @brief Per-parameter (lower, upper) bounds. Default: empty (unbounded).
     *        When non-empty, size must equal get_params().size().
     */
    virtual std::vector<std::pair<double, double>> bounds() const { return {}; }

    /*!
     * @brief Analytic gradient of log_likelihood w.r.t. get_params(). Default:
     *        empty — callers interpret empty as "use finite-difference
     *        gradient." Overriding lifts the problem from modes #1/#3 (slow
     *        finite-diff) to modes #2/#4 (fast analytic).
     */
    virtual std::vector<double> gradient() { return {}; }

    virtual std::vector<equality_constraint>   equality_constraints()   const { return {}; }
    virtual std::vector<inequality_constraint> inequality_constraints() const { return {}; }

    // ── Library-managed E-step state ─────────────────────────────────────────
    //
    // Buffer accessors. Lazily allocated on first call. Public for use by
    // optimize::* algorithms (which run the three traversals on them); user
    // code generally doesn't touch these directly.

    likelihood_buffer& inside_buffer(std::size_t tree_idx) {
        ensure_inside_buffers_allocated_();
        return *inside_buffers_[tree_idx];
    }

    likelihood_buffer& edge_inside_buffer(std::size_t tree_idx) {
        ensure_edge_inside_buffers_allocated_();
        return *edge_inside_buffers_[tree_idx];
    }

    likelihood_buffer& outside_buffer(std::size_t tree_idx) {
        ensure_outside_buffers_allocated_();
        return *outside_buffers_[tree_idx];
    }

    // ── Posterior accessors ──────────────────────────────────────────────────
    //
    // Valid only after the EM driver has run the three traversal passes
    // (inside via log_likelihood(); edge_inside and outside via
    // phylocore::compute_edge_inside_log_likelihood and
    // phylocore::compute_outside_log_likelihood on the buffers above).
    //
    // All values are in LOG SPACE. Caller applies exp() when needed.

    /*!
     * @brief log γ_v(i) = log P(z_v = i | y) at site c, node v.
     *
     * Returns a vector of length alphabet_size(c). Valid for any node v in
     * tree tree_idx (leaves, internal, root).
     */
    std::vector<double> node_posterior(
        std::size_t tree_idx, std::size_t node_id, std::size_t site_c)
    {
        const std::size_t alpha =
            stm->sites_latent_states[site_c].alphabet_size;
        std::vector<double> result(alpha);
        const auto& in  = inside_buffer(tree_idx);
        const auto& out = outside_buffer(tree_idx);
        const double L  = log_p_y_per_tree_per_site_[tree_idx][site_c];
        for (std::size_t i = 0; i < alpha; ++i) {
            result[i] = in(site_c, node_id, i)
                      + out(site_c, node_id, i)
                      - L;
        }
        return result;
    }

    /*!
     * @brief log γ_e(i, j) = log P(z_parent = i, z_child = j | y) at site c
     *        for the edge from parent w to child_node_id.
     *
     * Returns a k × k matrix (in log space). Multifurcation-supporting:
     * M_{w→u}(i) = outside(w, i) + Σ_{siblings z of u} edge_inside(z, i).
     *
     * Throws std::invalid_argument if child_node_id is the root (no
     * incoming edge in the phylogeny sense).
     */
    std::vector<std::vector<double>> edge_posterior(
        std::size_t tree_idx, std::size_t child_node_id, std::size_t site_c)
    {
        const tree& t = *trees[tree_idx];
        if (t.tree.in_degree(child_node_id) == 0) {
            throw std::invalid_argument(
                "edge_posterior: node has no parent (is root)");
        }

        const std::size_t alpha =
            stm->sites_latent_states[site_c].alphabet_size;
        const auto& in  = inside_buffer(tree_idx);
        const auto& ei  = edge_inside_buffer(tree_idx);
        const auto& out = outside_buffer(tree_idx);
        const double L  = log_p_y_per_tree_per_site_[tree_idx][site_c];

        const std::size_t parent_id =
            t.tree.predecessors(child_node_id)[0];

        // M_{w→u}(i) = outside(w, i) + Σ_{siblings z of u} edge_inside(z, i)
        std::vector<double> M(alpha);
        for (std::size_t i = 0; i < alpha; ++i) {
            M[i] = out(site_c, parent_id, i);
        }
        for (auto sibling_id : t.tree.successors(parent_id)) {
            if (static_cast<std::size_t>(sibling_id) == child_node_id) continue;
            for (std::size_t i = 0; i < alpha; ++i) {
                M[i] += ei(site_c, sibling_id, i);
            }
        }

        // γ_e(i, j) = M(i) + log P^c(t_u)_{ij} + inside(u, j) - log_total_L
        const double branch_length = t.branch_lengths[child_node_id];
        std::vector<std::vector<double>> result(
            alpha, std::vector<double>(alpha));
        for (std::size_t i = 0; i < alpha; ++i) {
            for (std::size_t j = 0; j < alpha; ++j) {
                const double log_P_ij = stm->compute_log_transition_probability(
                    (*node_data[tree_idx])[child_node_id],
                    site_c, branch_length, i, j);
                result[i][j] = M[i] + log_P_ij
                             + in(site_c, child_node_id, j)
                             - L;
            }
        }
        return result;
    }

    /*!
     * @brief Convenience: log γ_root(i) at site c. Same as node_posterior
     *        called on the tree's root_id.
     */
    std::vector<double> root_posterior(
        std::size_t tree_idx, std::size_t site_c)
    {
        return node_posterior(tree_idx, trees[tree_idx]->root_id, site_c);
    }

    /*!
     * @brief Convenience: log γ_ℓ(i) at site c for a leaf. Same as
     *        node_posterior; named for clarity in compute_suff_stats code
     *        that aggregates leaf responsibilities (e.g., phyloGMM).
     */
    std::vector<double> leaf_posterior(
        std::size_t tree_idx, std::size_t leaf_node_id, std::size_t site_c)
    {
        return node_posterior(tree_idx, leaf_node_id, site_c);
    }

    /*!
     * @brief Per-site marginal log-likelihood. log p(y_c | tree_idx).
     *
     * This is what posterior accessors use for normalization, since
     * inside(c, v, i) + outside(c, v, i) = log P(z_v = i, y_c) is a
     * per-site joint, not joint across sites. Populated by log_likelihood().
     */
    double log_p_y(std::size_t tree_idx, std::size_t site_c) const {
        return log_p_y_per_tree_per_site_[tree_idx][site_c];
    }

    /*!
     * @brief Per-tree total log-likelihood: sum over sites of log p(y_c | tree).
     *        Populated by log_likelihood() (computed on demand from the
     *        per-site values).
     */
    double log_total_likelihood(std::size_t tree_idx) const {
        double s = 0.0;
        for (double v : log_p_y_per_tree_per_site_[tree_idx]) s += v;
        return s;
    }

    /*!
     * @brief Throws std::invalid_argument if any taxa_id appears in more than
     *        one tree.
     *
     * The library's design assumes disjoint taxa_ids across trees — the same
     * taxon should not appear in multiple trees (since that would mean the
     * same observation is counted multiple times in suff_stats, producing
     * silently wrong results). Optimization entry points (optimize::em /
     * optimize::direct) call this at the top of the main loop to surface the
     * mistake with a clear message instead of returning garbage.
     */
    void validate_disjoint_taxa() const {
        std::unordered_map<std::size_t, std::size_t> taxa_to_tree;
        for (std::size_t ti = 0; ti < trees.size(); ++ti) {
            for (std::size_t v = 0; v < trees[ti]->num_nodes; ++v) {
                if (trees[ti]->tree.out_degree(static_cast<int>(v)) != 0) continue;  // leaves only
                const std::size_t taxa_id = trees[ti]->tree[static_cast<int>(v)].data;
                auto it = taxa_to_tree.find(taxa_id);
                if (it != taxa_to_tree.end() && it->second != ti) {
                    throw std::invalid_argument(
                        "joint_optimization_problem: taxa_id " + std::to_string(taxa_id) +
                        " appears in trees " + std::to_string(it->second) + " AND " +
                        std::to_string(ti) + ". Library assumes disjoint taxa across trees.");
                }
                taxa_to_tree[taxa_id] = ti;
            }
        }
    }

private:
    std::vector<std::unique_ptr<likelihood_buffer>> inside_buffers_;
    std::vector<std::unique_ptr<likelihood_buffer>> edge_inside_buffers_;
    std::vector<std::unique_ptr<likelihood_buffer>> outside_buffers_;
    // Per-tree, per-site log marginal: log_p_y_per_tree_per_site_[tree][site] = log p(y_site | tree).
    // Populated by log_likelihood(). Consumed by node_posterior, edge_posterior,
    // and external callers via log_p_y(tree_idx, site_c).
    std::vector<std::vector<double>>                log_p_y_per_tree_per_site_;

    // Compute (num_sites, max_alpha) from the model — shared by all three
    // buffer allocators.
    std::pair<std::size_t, std::size_t> buffer_dims_() const {
        const std::size_t num_sites = stm->sites_latent_states.size();
        const std::size_t max_alpha = std::max_element(
            stm->sites_latent_states.begin(),
            stm->sites_latent_states.end(),
            [](const state_space& a, const state_space& b) {
                return a.alphabet_size < b.alphabet_size;
            })->alphabet_size;
        return {num_sites, max_alpha};
    }

    void allocate_buffers_(
        std::vector<std::unique_ptr<likelihood_buffer>>& target)
    {
        target.clear();
        if (trees.empty()) return;
        const auto [num_sites, max_alpha] = buffer_dims_();
        target.reserve(trees.size());
        for (tree* t : trees) {
            target.push_back(std::make_unique<likelihood_buffer>(
                num_sites, max_alpha, t->num_nodes));
        }
    }

    void ensure_inside_buffers_allocated_() {
        if (inside_buffers_.size() != trees.size()) {
            allocate_buffers_(inside_buffers_);
        }
    }
    void ensure_edge_inside_buffers_allocated_() {
        if (edge_inside_buffers_.size() != trees.size()) {
            allocate_buffers_(edge_inside_buffers_);
        }
    }
    void ensure_outside_buffers_allocated_() {
        if (outside_buffers_.size() != trees.size()) {
            allocate_buffers_(outside_buffers_);
        }
    }
};

} // namespace optimize
} // namespace phylocore

#endif // PHYLOCORE_OPTIMIZE_JOINT_OPTIMIZATION_PROBLEM_H
