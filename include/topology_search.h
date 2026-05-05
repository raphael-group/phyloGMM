#ifndef PHYLOCORE_TOPOLOGY_SEARCH_H
#define PHYLOCORE_TOPOLOGY_SEARCH_H

#include <atomic>
#include <functional>
#include <queue>
#include <random>
#include <thread>
#include <unordered_set>
#include <variant>
#include <vector>

#include "phylogeny.h"

namespace phylocore {

// ─── NNI ─────────────────────────────────────────────────────────────────────

/*!
 * @brief A Nearest-Neighbor Interchange move that swaps the subtrees rooted at u and v.
 */
struct nni {
    int u;
    int v;
};

inline std::pair<int,int> apply_nni(tree& t, const nni& m) {
    int pu = t.tree.predecessors(m.u)[0];
    int pv = t.tree.predecessors(m.v)[0];
    t.tree.remove_edge(pu, m.u);
    t.tree.remove_edge(pv, m.v);
    t.tree.add_edge(pu, m.v);
    t.tree.add_edge(pv, m.u);
    return {pu, pv};
}

inline void revert_nni(tree& t, const nni& m, int pu, int pv) {
    t.tree.remove_edge(pu, m.v);
    t.tree.remove_edge(pv, m.u);
    t.tree.add_edge(pu, m.u);
    t.tree.add_edge(pv, m.v);
}

inline std::vector<nni> compute_nni_neighborhood(const tree& t) {
    std::vector<nni> moves;
    for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
        if (node_id == t.root_id || t.tree.out_degree(node_id) == 0) continue;
        int p_id = t.tree.predecessors(node_id)[0];
        int w_id = t.tree.successors(p_id)[0];
        if (w_id == (int)node_id) w_id = t.tree.successors(p_id)[1];
        for (int u_id : t.tree.successors(node_id))
            moves.push_back({w_id, u_id});
    }
    return moves;
}

template <typename D>
double evaluate_single_nni(
    const std::function<double(tree&, D&)>& scoring_fn,
    tree& t, D& model, const nni& move)
{
    auto [pu, pv] = apply_nni(t, move);
    double score  = scoring_fn(t, model);
    revert_nni(t, move, pu, pv);
    return score;
}

template <typename D>
std::vector<std::pair<nni, double>>
evaluate_nnis(
    const std::function<double(tree&, D&)>& scoring_fn,
    tree& t, D& model,
    const std::vector<nni>& moves)
{
    std::vector<std::pair<nni, double>> out;
    out.reserve(moves.size());
    for (const auto& m : moves)
        out.emplace_back(m, evaluate_single_nni(scoring_fn, t, model, m));
    return out;
}

/*!
 * @brief Evaluates every NNI in the neighborhood of initial_tree.
 *
 * Moves are partitioned evenly across `threads` worker threads (sequential if threads <= 1).
 * scoring_fn must not modify branch lengths or model parameters.
 */
template <typename D>
std::vector<std::pair<nni, double>> evaluate_nni_neighborhood(
    const std::function<double(tree&, D&)>& scoring_fn,
    const tree& initial_tree,
    const D& initial_model,
    int threads = 8)
{
    tree t   = initial_tree;
    D model  = initial_model;
    std::vector<nni> moves = compute_nni_neighborhood(t);

    if (threads <= 1) return evaluate_nnis(scoring_fn, t, model, moves);

    std::vector<std::vector<nni>>                     thread_moves(threads);
    std::vector<std::vector<std::pair<nni, double>>>  thread_results(threads);

    for (size_t i = 0; i < moves.size(); ++i)
        thread_moves[i % threads].push_back(moves[i]);

    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i) {
        pool.emplace_back([&, i]() {
            tree ti   = t;
            D modeli  = model;
            thread_results[i] = evaluate_nnis(scoring_fn, ti, modeli, thread_moves[i]);
        });
    }
    for (auto& th : pool) th.join();

    std::vector<std::pair<nni, double>> out;
    for (const auto& r : thread_results)
        out.insert(out.end(), r.begin(), r.end());
    return out;
}

inline void stochastically_perturb_tree(tree& t, int nni_count, std::mt19937& gen) {
    std::uniform_int_distribution<int> dist(0, t.num_nodes - 1);
    for (int i = 0; i < nni_count; ++i) {
        int node_id = dist(gen);
        if (node_id == (int)t.root_id || t.tree.out_degree(node_id) == 0) continue;
        int p_id = t.tree.predecessors(node_id)[0];
        int w_id = t.tree.successors(p_id)[0];
        if (w_id == node_id) w_id = t.tree.successors(p_id)[1];
        auto children = t.tree.successors(node_id);
        int u_id = children[dist(gen) % children.size()];
        nni m{w_id, u_id};
        apply_nni(t, m);
    }
}

// ─── SPR ─────────────────────────────────────────────────────────────────────

/*!
 * @brief A Subtree Pruning and Regrafting move on a rooted binary tree.
 *
 * Prunes the subtree rooted at s from its parent p (grandparent g, sibling sib).
 * After the prune, g connects directly to sib; p is reused as the new internal
 * node at the regraft site, replacing the edge u → v with u → p → {s, v}.
 */
struct spr {
    int s;    // subtree root being pruned
    int p;    // parent of s (reused as new internal node at regraft site)
    int g;    // grandparent of s
    int sib;  // sibling of s
    int u;    // regraft edge: parent endpoint
    int v;    // regraft edge: child endpoint
};

inline void apply_spr(tree& t, const spr& m) {
    t.tree.remove_edge(m.g, m.p);
    t.tree.remove_edge(m.p, m.sib);
    t.tree.remove_edge(m.p, m.s);
    t.tree.add_edge(m.g, m.sib);
    t.tree.remove_edge(m.u, m.v);
    t.tree.add_edge(m.u, m.p);
    t.tree.add_edge(m.p, m.s);
    t.tree.add_edge(m.p, m.v);
}

inline void revert_spr(tree& t, const spr& m) {
    t.tree.remove_edge(m.u, m.p);
    t.tree.remove_edge(m.p, m.s);
    t.tree.remove_edge(m.p, m.v);
    t.tree.add_edge(m.u, m.v);
    t.tree.remove_edge(m.g, m.sib);
    t.tree.add_edge(m.g, m.p);
    t.tree.add_edge(m.p, m.sib);
    t.tree.add_edge(m.p, m.s);
}

inline void collect_subtree(const tree& t, int root_id, std::unordered_set<int>& out) {
    std::queue<int> q;
    q.push(root_id);
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        if (!out.insert(cur).second) continue;
        for (int child : t.tree.successors(cur)) q.push(child);
    }
}

/*!
 * @brief Enumerates all valid SPR moves for a rooted binary tree. O(N²) moves.
 */
inline std::vector<spr> compute_spr_neighborhood(const tree& t) {
    std::vector<spr> moves;
    for (size_t s_id = 0; s_id < t.num_nodes; ++s_id) {
        if (s_id == t.root_id) continue;
        int p_id = t.tree.predecessors(s_id)[0];
        if (p_id == (int)t.root_id) continue;
        int g_id  = t.tree.predecessors(p_id)[0];
        auto p_ch = t.tree.successors(p_id);
        int sib_id = (p_ch[0] == (int)s_id) ? p_ch[1] : p_ch[0];

        std::unordered_set<int> subtree;
        collect_subtree(t, (int)s_id, subtree);
        subtree.insert(p_id);

        for (size_t u_id = 0; u_id < t.num_nodes; ++u_id) {
            if (subtree.count(u_id)) continue;
            for (int v_id : t.tree.successors(u_id)) {
                if (subtree.count(v_id)) continue;
                if ((int)u_id == g_id && v_id == sib_id) continue; // identity
                moves.push_back({(int)s_id, p_id, g_id, sib_id, (int)u_id, v_id});
            }
        }
    }
    return moves;
}

template <typename D>
double evaluate_single_spr(
    const std::function<double(tree&, D&)>& scoring_fn,
    tree& t, D& model, const spr& move)
{
    apply_spr(t, move);
    double llh = scoring_fn(t, model);
    revert_spr(t, move);
    return llh;
}

} // namespace phylocore

#endif // PHYLOCORE_TOPOLOGY_SEARCH_H
