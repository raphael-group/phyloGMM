#ifndef PHYLOCORE_TOPOLOGY_SEARCH_H
#define PHYLOCORE_TOPOLOGY_SEARCH_H

/*!
 * @file topology_search.h
 * @brief Topology search utilities for phylogenetic trees.
 *
 * Provides NNI and SPR tree move primitives, a scoring layer over any
 * joint_optimization_problem, and two complete topology search strategies
 * (hill-climbing and simulated annealing).
 *
 * This header is fully self-contained; it has no dependency on the phylocore
 * core library includes beyond `phylogeny.h`. All NNI/SPR logic is declared
 * under phylocore::topology_search.
 *
 * ## Protocol (duck typing, no inheritance required)
 *
 * Problem must expose:
 *   std::vector<tree*>          trees             (public member; index 0 is
 *                                                  the tree being searched)
 *   double                      log_likelihood()  (reads directly from trees[0]
 *                                                  on every call — no stale
 *                                                  cache between topology edits)
 *   std::vector<double>         get_params() const
 *   void                        set_params(const std::vector<double>&)
 *
 * The scoring and search functions modify *p.trees[0] directly (in place) and
 * revert any rejected changes.  branch_lengths and the parameter vector are
 * also saved and restored on rejection so no Problem state leaks.
 */

#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

#include "phylogeny.h"

namespace phylocore {
namespace topology_search {

// ── NNI ──────────────────────────────────────────────────────────────────────

/*!
 * @brief A Nearest-Neighbor Interchange move that swaps the subtrees rooted
 *        at u and v.
 */
struct nni {
    int u;
    int v;
};

/*!
 * @brief Apply an NNI move to t in place.
 *
 * Returns {parent(u), parent(v)} before the swap so the caller can revert
 * via revert_nni.
 */
inline std::pair<int, int> apply_nni(tree& t, const nni& m)
{
    int pu = t.tree.predecessors(m.u)[0];
    int pv = t.tree.predecessors(m.v)[0];
    t.tree.remove_edge(pu, m.u);
    t.tree.remove_edge(pv, m.v);
    t.tree.add_edge(pu, m.v);
    t.tree.add_edge(pv, m.u);
    return {pu, pv};
}

/*!
 * @brief Revert an NNI move using the {pu, pv} pair returned by apply_nni.
 */
inline void revert_nni(tree& t, const nni& m, int pu, int pv)
{
    t.tree.remove_edge(pu, m.v);
    t.tree.remove_edge(pv, m.u);
    t.tree.add_edge(pu, m.u);
    t.tree.add_edge(pv, m.v);
}

/*!
 * @brief Enumerate every NNI move in the neighborhood of t.
 *
 * Skips the root and leaf nodes; each internal edge contributes two moves
 * (one per child of the internal node).
 */
inline std::vector<nni> compute_nni_neighborhood(const tree& t)
{
    std::vector<nni> moves;
    for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
        if (node_id == t.root_id || t.tree.out_degree(node_id) == 0) continue;
        int p_id = t.tree.predecessors(node_id)[0];
        int w_id = t.tree.successors(p_id)[0];
        if (w_id == static_cast<int>(node_id)) w_id = t.tree.successors(p_id)[1];
        for (int u_id : t.tree.successors(node_id))
            moves.push_back({w_id, u_id});
    }
    return moves;
}

/*!
 * @brief Apply one NNI move chosen uniformly at random from the neighborhood
 *        of t using the given integer seed. Returns the move that was applied.
 *
 * Precondition: the NNI neighborhood of t is non-empty.
 */
inline nni apply_rand_nni(tree& t, int seed)
{
    std::mt19937 gen(static_cast<unsigned>(seed));
    std::vector<nni> moves = compute_nni_neighborhood(t);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(moves.size()) - 1);
    nni chosen = moves[dist(gen)];
    apply_nni(t, chosen);
    return chosen;
}

// ── SPR ──────────────────────────────────────────────────────────────────────

/*!
 * @brief A Subtree Pruning and Regrafting move on a rooted binary tree.
 *
 * Prunes the subtree rooted at s from its parent p (grandparent g, sibling
 * sib). After the prune, g connects directly to sib. p is reused as the new
 * internal node at the regraft site, replacing the edge u → v with
 * u → p → {s, v}.
 */
struct spr {
    int s;    // subtree root being pruned
    int p;    // parent of s (reused as new internal node at regraft site)
    int g;    // grandparent of s
    int sib;  // sibling of s
    int u;    // regraft edge: parent endpoint
    int v;    // regraft edge: child endpoint
};

/*!
 * @brief Apply an SPR move to t in place.
 */
inline void apply_spr(tree& t, const spr& m)
{
    t.tree.remove_edge(m.g, m.p);
    t.tree.remove_edge(m.p, m.sib);
    t.tree.remove_edge(m.p, m.s);
    t.tree.add_edge(m.g, m.sib);
    t.tree.remove_edge(m.u, m.v);
    t.tree.add_edge(m.u, m.p);
    t.tree.add_edge(m.p, m.s);
    t.tree.add_edge(m.p, m.v);
}

/*!
 * @brief Revert an SPR move applied by apply_spr.
 */
inline void revert_spr(tree& t, const spr& m)
{
    t.tree.remove_edge(m.u, m.p);
    t.tree.remove_edge(m.p, m.s);
    t.tree.remove_edge(m.p, m.v);
    t.tree.add_edge(m.u, m.v);
    t.tree.remove_edge(m.g, m.sib);
    t.tree.add_edge(m.g, m.p);
    t.tree.add_edge(m.p, m.sib);
    t.tree.add_edge(m.p, m.s);
}

namespace detail {

inline void collect_subtree(const tree& t, int root_id,
                             std::unordered_set<int>& out)
{
    std::queue<int> q;
    q.push(root_id);
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        if (!out.insert(cur).second) continue;
        for (int child : t.tree.successors(cur)) q.push(child);
    }
}

} // namespace detail

/*!
 * @brief Enumerate all valid SPR moves for a rooted binary tree. O(N²) moves.
 *
 * Excludes the identity move (regrafting s back onto its original parent edge).
 */
inline std::vector<spr> compute_spr_neighborhood(const tree& t)
{
    std::vector<spr> moves;
    for (size_t s_id = 0; s_id < t.num_nodes; ++s_id) {
        if (s_id == t.root_id) continue;
        int p_id = t.tree.predecessors(s_id)[0];
        if (p_id == static_cast<int>(t.root_id)) continue;
        int g_id  = t.tree.predecessors(p_id)[0];
        auto p_ch = t.tree.successors(p_id);
        int sib_id = (p_ch[0] == static_cast<int>(s_id)) ? p_ch[1] : p_ch[0];

        std::unordered_set<int> subtree;
        detail::collect_subtree(t, static_cast<int>(s_id), subtree);
        subtree.insert(p_id);

        for (size_t u_id = 0; u_id < t.num_nodes; ++u_id) {
            if (subtree.count(u_id)) continue;
            for (int v_id : t.tree.successors(u_id)) {
                if (subtree.count(v_id)) continue;
                if (static_cast<int>(u_id) == g_id && v_id == sib_id) continue;
                moves.push_back({static_cast<int>(s_id), p_id, g_id, sib_id,
                                 static_cast<int>(u_id), v_id});
            }
        }
    }
    return moves;
}

// ── Results and options ───────────────────────────────────────────────────────

/*!
 * @brief Result returned by topology search strategies.
 */
struct topology_evaluation_result {
    tree                best_tree;
    double              log_likelihood  = 0.0;
    int                 iterations      = 0;
    std::vector<double> log_likelihoods;
};

/*!
 * @brief Options shared by all search strategies.
 */
struct search_strategy_options {
    int    max_iter          = 100;
    int    num_threads       = 1;
    bool   is_ultrametric    = false;
    double min_branch_length = 1e-6;
    double initial_phi       = 1.0;
    double initial_nu        = 1.0;
};

// ── Topology scoring ──────────────────────────────────────────────────────────

/*!
 * @brief Score @p move against a caller-provided copy of the tree.
 *
 * @p t is consumed: the move is applied to it and the copy is discarded.
 * p.trees[0] is temporarily redirected to &t for the log_likelihood() call
 * and restored before returning, so no Problem state leaks.
 *
 * @tparam Problem  Any type satisfying the protocol in the file header.
 */
template <typename Problem>
double score_nni_move(tree t, const nni& move, Problem& p)
{
    apply_nni(t, move);
    tree* orig   = p.trees[0];
    p.trees[0]   = &t;
    double score = p.log_likelihood();
    p.trees[0]   = orig;
    return score;
}

/*!
 * @brief Score every NNI in the neighborhood of @p t; return the best result.
 *
 * @p t is a copy of the base topology. A fresh copy is made per move and
 * discarded after scoring — no revert needed. Does not permanently apply the
 * best move; best_tree in the result is a separate copy with the best move
 * applied.
 */
template <typename Problem>
topology_evaluation_result score_nni_neighborhood(tree t, Problem& p)
{
    tree* orig        = p.trees[0];
    p.trees[0]        = &t;
    double current_ll = p.log_likelihood();
    p.trees[0]        = orig;

    auto moves = compute_nni_neighborhood(t);

    topology_evaluation_result result;
    result.log_likelihood = current_ll;
    result.best_tree      = t;
    result.iterations     = 1;
    result.log_likelihoods.reserve(moves.size());

    nni  best_move{};
    bool found_better = false;

    for (const auto& m : moves) {
        // score_nni_move takes t by value — copies t, applies move, discards
        double score = score_nni_move(t, m, p);
        result.log_likelihoods.push_back(score);
        if (score > result.log_likelihood) {
            result.log_likelihood = score;
            best_move             = m;
            found_better          = true;
        }
    }

    if (found_better) {
        result.best_tree = t;  // t was never mutated during the loop
        apply_nni(result.best_tree, best_move);
    }

    return result;
}

/*!
 * @brief Score @p move against a caller-provided copy of the tree.
 *
 * @p t is consumed: the move is applied and the copy is discarded.
 * p.trees[0] is temporarily redirected to &t for the log_likelihood() call
 * and restored before returning.
 */
template <typename Problem>
double score_spr_move(tree t, const spr& move, Problem& p)
{
    apply_spr(t, move);
    tree* orig   = p.trees[0];
    p.trees[0]   = &t;
    double score = p.log_likelihood();
    p.trees[0]   = orig;
    return score;
}

// ── Search strategies ─────────────────────────────────────────────────────────

/*!
 * @brief Hill-climbing topology search over p.trees[0] using NNI moves.
 *
 * Each round scores the full NNI neighborhood and applies the best improving
 * move. Terminates when no move improves the score or max_iter is reached.
 *
 * @tparam Problem  Must satisfy the protocol in the file header.
 */
template <typename Problem>
topology_evaluation_result hill_climbing(Problem& p,
                                         search_strategy_options opts = {})
{
    tree&  current_tree = *p.trees[0];
    double current_ll   = p.log_likelihood();

    topology_evaluation_result result;
    result.best_tree      = current_tree;
    result.log_likelihood = current_ll;
    result.log_likelihoods.push_back(current_ll);

    for (int iter = 0; iter < opts.max_iter; ++iter) {
        auto nbr = score_nni_neighborhood(current_tree, p);
        if (nbr.log_likelihood > current_ll) {
            current_tree          = nbr.best_tree;
            current_ll            = nbr.log_likelihood;
            result.best_tree      = current_tree;
            result.log_likelihood = current_ll;
            result.log_likelihoods.push_back(current_ll);
            result.iterations     = iter + 1;
        } else {
            break;
        }
    }

    return result;
}

/*!
 * @brief Simulated annealing topology search over p.trees[0] using NNI moves.
 *
 * Temperature schedule: T = T0 · alpha^(no_accepts), T0 = 0.1, alpha = 0.99.
 * Temperature decreases with each accepted move (not with each iteration),
 * preserving exploration early in the search. A proposed move is accepted with
 * probability min(1, exp(Δ / T)). On rejection the NNI move, branch lengths,
 * and model parameters are all restored.
 *
 * Adapted from LAML-Pro/include/topology_search.h::simulated_annealing.
 *
 * @tparam Problem  Must satisfy the protocol in the file header (including
 *                  get_params() and set_params()).
 */
template <typename Problem>
topology_evaluation_result simulated_annealing(Problem& p,
                                               search_strategy_options opts = {})
{
    constexpr double alpha = 0.99;
    constexpr double T0    = 0.1;

    tree&  current_tree = *p.trees[0];
    double current_ll   = p.log_likelihood();

    std::mt19937                     gen(std::random_device{}());
    std::uniform_real_distribution<> uni(0.0, 1.0);

    std::vector<double> log_likelihoods;
    log_likelihoods.push_back(current_ll);

    std::size_t no_accepts = 0;
    int         iteration  = 0;

    while (iteration < opts.max_iter) {
        std::vector<nni> neighborhood = compute_nni_neighborhood(current_tree);
        if (neighborhood.empty()) break;

        std::vector<double>          weights(neighborhood.size(), 1.0);
        std::discrete_distribution<> sampler(weights.begin(), weights.end());
        nni sampled_move = neighborhood[sampler(gen)];

        auto saved_blens  = current_tree.branch_lengths;
        auto saved_params = p.get_params();

        auto [pu, pv]  = apply_nni(current_tree, sampled_move);
        double move_ll = p.log_likelihood();
        double delta   = move_ll - current_ll;

        double T        = T0 * std::pow(alpha, static_cast<double>(no_accepts));
        double proposal = std::exp(delta / T);

        if (delta > 0.0 || proposal > uni(gen)) {
            current_ll = move_ll;
            ++no_accepts;
        } else {
            revert_nni(current_tree, sampled_move, pu, pv);
            current_tree.branch_lengths = std::move(saved_blens);
            p.set_params(saved_params);
        }

        log_likelihoods.push_back(current_ll);
        ++iteration;
    }

    return {current_tree, current_ll, iteration, std::move(log_likelihoods)};
}

} // namespace topology_search
} // namespace phylocore

#endif // PHYLOCORE_TOPOLOGY_SEARCH_H
