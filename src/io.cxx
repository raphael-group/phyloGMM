#include "io.h"

#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "csv.hpp"
#include "extern/compact_tree.h"

namespace phylocore {

// ─── Newick I/O ──────────────────────────────────────────────────────────────

static tree parse_newick_tree_file(const std::string& fname) {
    compact_tree ct(fname);

    size_t num_nodes  = ct.get_num_nodes();
    size_t num_leaves = ct.get_num_leaves();
    std::vector<int> name_map(num_nodes, -1);

    digraph<size_t> g;
    std::vector<double> branch_lengths(num_nodes);
    std::vector<std::string> node_names(num_nodes);

    size_t leaf_idx     = 0;
    size_t internal_idx = num_leaves;
    for (size_t i = 0; i < num_nodes; ++i) {
        int j = (ct.get_children(i).empty()) ? (int)leaf_idx++ : (int)internal_idx++;
        int id = g.add_vertex((size_t)j);
        branch_lengths[id] = ct.get_edge_length(i);
        if (branch_lengths[id] == 0.0) branch_lengths[id] = 1.0;
        node_names[id] = ct.get_label(i);
        name_map[i]    = id;
    }

    int root_id = name_map[0];
    for (size_t i = 0; i < num_nodes; ++i)
        for (auto child : ct.get_children(i))
            g.add_edge(name_map[i], name_map[child]);

    if (root_id == -1) throw std::runtime_error("Root node not found in tree");
    return {num_leaves, num_nodes, (size_t)root_id, g, branch_lengths, node_names};
}

std::vector<tree> parse_newick_trees(const std::string& fname) {
    std::ifstream file(fname);
    if (!file) throw std::runtime_error("Cannot open tree file: " + fname);

    std::vector<tree> trees;
    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        ++line_num;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        std::string tmp = "/tmp/phylocore_newick_" + std::to_string(line_num) + ".nwk";
        { std::ofstream out(tmp); if (!out) throw std::runtime_error("Cannot create temp file: " + tmp); out << line << "\n"; }
        trees.push_back(parse_newick_tree_file(tmp));
        std::remove(tmp.c_str());
    }
    if (trees.empty()) throw std::runtime_error("No Newick strings found in file: " + fname);
    return trees;
}

std::string write_newick_tree(const tree& t) {
    std::function<std::string(size_t)> rec = [&](size_t node) -> std::string {
        std::string result;
        auto children = t.tree.successors(node);
        if (!children.empty()) {
            result += "(";
            for (size_t i = 0; i < children.size(); ++i) {
                if (i > 0) result += ",";
                result += rec(children[i]);
            }
            result += ")";
        }
        if (!t.node_names[node].empty()) result += t.node_names[node];
        result += ":" + std::to_string(t.branch_lengths[node]);
        return result;
    };
    return rec(t.root_id) + ";";
}

// ─── Character matrix ─────────────────────────────────────────────────────────

std::pair<std::vector<std::string>, std::vector<std::vector<int>>>
parse_character_matrix(const std::string& filename) {
    std::vector<std::string> taxa_names;
    std::vector<std::vector<int>> matrix;
    try {
        csv::CSVReader reader(filename);
        for (auto& row : reader) {
            taxa_names.push_back(row[0].get<std::string>());
            std::vector<int> states;
            for (size_t i = 1; i < row.size(); ++i) {
                if (row[i].is_null() || row[i].get<std::string>() == "?")
                    states.push_back(-1);
                else
                    states.push_back(row[i].get<int>());
            }
            matrix.push_back(std::move(states));
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse character matrix: " + std::string(e.what()));
    }
    if (taxa_names.empty()) throw std::runtime_error("No data found in character matrix file");
    return {taxa_names, matrix};
}

// ─── process_phylogeny_data ──────────────────────────────────────────────────

phylogeny_data process_phylogeny_data(
    std::vector<tree>& trees,
    const std::string& character_matrix_file)
{
    if (trees.empty()) throw std::runtime_error("process_phylogeny_data: no trees provided");

    auto [file_taxa_names, raw_matrix] = parse_character_matrix(character_matrix_file);

    // Map file taxa name → row index
    std::unordered_map<std::string, size_t> file_taxa_idx;
    for (size_t i = 0; i < file_taxa_names.size(); ++i)
        file_taxa_idx[file_taxa_names[i]] = i;

    // Collect leaf names across trees; validate disjoint sets
    std::unordered_map<std::string, size_t> leaf_to_tree;
    std::vector<std::pair<size_t, size_t>> leaf_global_order; // (tree_idx, node_id)
    for (size_t k = 0; k < trees.size(); ++k) {
        for (size_t node_id = 0; node_id < trees[k].num_nodes; ++node_id) {
            if (trees[k].tree.out_degree(node_id) != 0) continue;
            const std::string& name = trees[k].node_names[node_id];
            if (leaf_to_tree.count(name))
                throw std::runtime_error("Leaf '" + name + "' appears in multiple trees");
            leaf_to_tree[name] = k;
            leaf_global_order.emplace_back(k, node_id);
        }
    }

    size_t total_leaves   = leaf_global_order.size();
    size_t num_characters = raw_matrix.empty() ? 0 : raw_matrix[0].size();

    // Build aligned global matrix; assign leaf node.data
    std::vector<std::vector<int>> global_matrix(total_leaves,
                                                 std::vector<int>(num_characters, -1));
    std::vector<std::string> global_taxa_names(total_leaves);

    for (size_t i = 0; i < total_leaves; ++i) {
        auto [k, node_id] = leaf_global_order[i];
        const std::string& name = trees[k].node_names[node_id];
        if (!file_taxa_idx.count(name))
            throw std::runtime_error("Leaf '" + name + "' not found in character matrix");
        global_taxa_names[i] = name;
        global_matrix[i]     = raw_matrix[file_taxa_idx.at(name)];
        trees[k].tree[node_id].data = i;
    }

    // Assign globally-offset IDs to internal nodes
    size_t internal_counter = total_leaves;
    for (auto& t : trees)
        for (size_t node_id = 0; node_id < t.num_nodes; ++node_id)
            if (t.tree.out_degree(node_id) != 0)
                t.tree[node_id].data = internal_counter++;

    // Compute max_alphabet_size from observed states
    int max_state = 0;
    for (const auto& row : global_matrix)
        for (int s : row)
            if (s > max_state) max_state = s;

    return {
        num_characters,
        static_cast<size_t>(max_state + 1),
        global_taxa_names,
        global_matrix
    };
}

} // namespace phylocore
