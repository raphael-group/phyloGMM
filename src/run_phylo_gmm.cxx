#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "csv.hpp"
#include "em_optimization.h"
#include "io.h"
#include "models/phylo_gmm_model.h"

using namespace phylocore;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <tree_file> <obs_csv> <k> [max_iter] [tol]\n";
        return 1;
    }

    std::string tree_file = argv[1];
    std::string obs_file  = argv[2];
    size_t      k         = std::stoul(argv[3]);
    int         max_iter  = (argc > 4) ? std::stoi(argv[4]) : 100;
    double      tol       = (argc > 5) ? std::stod(argv[5]) : 1e-6;

    // ── Parse tree ────────────────────────────────────────────────────────────
    auto trees = parse_newick_trees(tree_file);
    tree& t = trees[0];
    std::cerr << "Tree: " << t.num_leaves << " leaves, "
              << t.num_nodes << " nodes\n";

    // ── Parse observations CSV ────────────────────────────────────────────────
    // columns: cell, x0, x1, ..., x{d-1}
    std::unordered_map<std::string, std::vector<double>> obs_map;
    size_t d = 0;
    {
        csv::CSVReader reader(obs_file);
        auto col_names = reader.get_col_names();
        d = col_names.size() - 1;
        for (auto& row : reader) {
            std::string cell = row[0].get<std::string>();
            std::vector<double> vals(d);
            for (size_t j = 0; j < d; ++j)
                vals[j] = row[j + 1].get<double>();
            obs_map[std::move(cell)] = std::move(vals);
        }
    }
    std::cerr << "Observations: " << obs_map.size() << " cells, d=" << d << "\n";

    // ── Align leaves: build Y (n_leaves × d) and is_observed mask ───────────
    // Indexed by leaf.data (0..num_leaves-1), consistent with taxa_id in model
    Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(t.num_leaves, d);
    std::vector<bool> is_observed(t.num_leaves, false);

    size_t n_observed = 0;
    for (int node_id : t.tree.nodes()) {
        if (t.tree.out_degree(node_id) != 0) continue;
        const std::string& name = t.node_names[node_id];
        auto it = obs_map.find(name);
        if (it != obs_map.end()) {
            size_t leaf_data = t.tree[node_id].data;
            for (size_t j = 0; j < d; ++j)
                Y(leaf_data, j) = it->second[j];
            is_observed[leaf_data] = true;
            ++n_observed;
        }
    }
    std::cerr << "Aligned: " << n_observed << " observed leaves out of "
              << t.num_leaves << "\n";

    if (n_observed == 0) {
        std::cerr << "Error: no observed leaves aligned — check cell names\n";
        return 1;
    }

    // ── Build Y_obs for k-means init (observed rows only) ────────────────────
    Eigen::MatrixXd Y_obs(n_observed, d);
    {
        size_t row = 0;
        for (size_t ell = 0; ell < t.num_leaves; ++ell)
            if (is_observed[ell])
                Y_obs.row(row++) = Y.row(ell);
    }

    // ── Initialize model via k-means on observed data ─────────────────────────
    std::cerr << "Running k-means++ init (k=" << k << ")...\n";
    phylo_gmm_model model = phylo_gmm_model::init_kmeans(k, Y_obs);
    model.Y           = Y;
    model.is_observed = is_observed;

    // ── EM ────────────────────────────────────────────────────────────────────
    std::vector<phylo_gmm_node_data> node_data(t.num_nodes);
    std::cerr << "Running EM (max_iter=" << max_iter << ", tol=" << tol << ")...\n";

    double ll_final = em_optimize(t, model, node_data, tol, max_iter);

    // ── Emit JSON to stdout ───────────────────────────────────────────────────
    std::cout << "{\n";
    std::cout << "  \"log_likelihood\": " << ll_final << ",\n";
    std::cout << "  \"k\": " << k << ",\n";
    std::cout << "  \"d\": " << d << ",\n";
    std::cout << "  \"n_observed\": " << n_observed << ",\n";
    std::cout << "  \"n_leaves\": " << t.num_leaves << ",\n";

    // Q matrix
    std::cout << "  \"Q\": [";
    for (size_t i = 0; i < k; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "[";
        for (size_t j = 0; j < k; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << model.Q(i, j);
        }
        std::cout << "]";
    }
    std::cout << "],\n";

    // mu
    std::cout << "  \"mu\": [";
    for (size_t i = 0; i < k; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "[";
        for (size_t j = 0; j < d; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << model.mu[i][j];
        }
        std::cout << "]";
    }
    std::cout << "],\n";

    // sigma diagonal std devs
    std::cout << "  \"sigma_diag\": [";
    for (size_t i = 0; i < k; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "[";
        for (size_t j = 0; j < d; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::sqrt(model.Sigma[i](j, j));
        }
        std::cout << "]";
    }
    std::cout << "],\n";

    // pi
    std::cout << "  \"pi\": [";
    for (size_t i = 0; i < k; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << model.pi[i];
    }
    std::cout << "]\n";
    std::cout << "}\n";

    return 0;
}
