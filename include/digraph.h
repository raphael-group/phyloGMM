#ifndef PHYLOCORE_DIGRAPH_H
#define PHYLOCORE_DIGRAPH_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace phylocore {

template <class T>
class vertex {
public:
    int id;
    T data;
    vertex() {};
    vertex(int id, T data) : id(id), data(data) {};
};

/*!
 * @brief An adjacency list representation of a directed graph.
 *
 * Vertices are numbered 0 to N-1; each carries arbitrary data of type T.
 *
 * @tparam T The type of data stored in each vertex.
 */
template <class T>
class digraph {
private:
    int id_counter = 0;

    std::vector<std::vector<int>> succ;
    std::vector<std::vector<int>> pred;
    std::vector<vertex<T>> vertices;

public:
    int add_vertex(T data) {
        vertex<T> v(id_counter, data);
        vertices.push_back(v);
        succ.push_back({});
        pred.push_back({});
        id_counter++;
        return v.id;
    }

    void add_edge(int u, int v) {
        succ[u].push_back(v);
        pred[v].push_back(u);
    }

    void remove_edge(int u, int v) {
        succ[u].erase(std::remove(succ[u].begin(), succ[u].end(), v), succ[u].end());
        pred[v].erase(std::remove(pred[v].begin(), pred[v].end(), u), pred[v].end());
    }

    size_t size() const { return vertices.size(); }

    std::vector<int> nodes() const {
        std::vector<int> out;
        out.reserve(id_counter);
        for (int i = 0; i < id_counter; ++i) out.push_back(i);
        return out;
    }

    std::vector<std::pair<int, int>> edges() const {
        std::vector<std::pair<int, int>> out;
        for (size_t u = 0; u < succ.size(); ++u)
            for (int v : succ[u])
                out.emplace_back(u, v);
        return out;
    }

    vertex<T>& operator[](int u)             { return vertices[u]; }
    const vertex<T>& operator[](int u) const { return vertices[u]; }

    const std::vector<int>& predecessors(int u) const { return pred[u]; }
    const std::vector<int>& successors(int u)   const { return succ[u]; }

    bool contains(int u) const {
        return u >= 0 && static_cast<size_t>(u) < vertices.size();
    }

    size_t in_degree(int u)  const { return pred[u].size(); }
    size_t out_degree(int u) const { return succ[u].size(); }

    std::vector<int> preorder_traversal(int root) const {
        std::vector<int> order;
        std::stack<int> stk;
        stk.push(root);
        while (!stk.empty()) {
            int i = stk.top(); stk.pop();
            order.push_back(i);
            for (int k : this->successors(i)) stk.push(k);
        }
        return order;
    }

    std::vector<int> postorder_traversal(int root) const {
        std::stack<int> stk;
        stk.push(root);
        std::vector<int> order;
        std::vector<bool> visited(id_counter, false);
        while (!stk.empty()) {
            int i = stk.top(); stk.pop();
            if (visited[i]) continue;
            if (this->out_degree(i) == 0) {
                visited[i] = true;
                order.push_back(i);
                continue;
            }
            const auto& children = this->successors(i);
            bool all_done = true;
            for (int k : children) {
                if (!visited[k]) {
                    if (all_done) stk.push(i);
                    stk.push(k);
                    all_done = false;
                }
            }
            if (!all_done) continue;
            visited[i] = true;
            order.push_back(i);
        }
        return order;
    }

    friend bool ancestor(const digraph<T>& g, int u, int v) {
        if (u == v) return true;
        for (int w : g.succ[u])
            if (ancestor(g, w, v)) return true;
        return false;
    }
};

} // namespace phylocore

#endif // PHYLOCORE_DIGRAPH_H
