#include "io.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
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

    // Per-process, per-call temp path so concurrent phylocore processes can't
    // collide on a hard-coded /tmp/phylocore_newick_<line>.nwk. Honor TMPDIR
    // when set; otherwise fall back to /tmp. Include the PID + a process-local
    // monotonic counter so multiple parse_newick_trees() calls within one
    // process are also unique.
    const char* tmpdir_env = std::getenv("TMPDIR");
    std::string tmpdir = (tmpdir_env && *tmpdir_env) ? tmpdir_env : "/tmp";
    if (!tmpdir.empty() && tmpdir.back() == '/') tmpdir.pop_back();
    static std::atomic<int> call_counter{0};
    int call_id = call_counter.fetch_add(1);
    int pid = static_cast<int>(::getpid());

    std::vector<tree> trees;
    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        ++line_num;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        std::string tmp = tmpdir + "/phylocore_newick_pid" + std::to_string(pid)
                          + "_c" + std::to_string(call_id)
                          + "_l" + std::to_string(line_num) + ".nwk";
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

// ─── process_input_data ──────────────────────────────────────────────────────

template <>
process_input_result<std::vector<int>>
process_input_data<std::vector<int>>(
    std::vector<tree>& trees,
    const std::string& observations_file)
{
    if (trees.empty()) throw std::runtime_error("process_input_data: no trees provided");

    auto [file_taxa_names, raw_matrix] = parse_character_matrix(observations_file);

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

    size_t total_leaves = leaf_global_order.size();
    size_t num_sites    = raw_matrix.empty() ? 0 : raw_matrix[0].size();

    // Build aligned per-leaf observation rows; assign leaf node.data
    std::vector<std::vector<int>> observed_data(total_leaves,
                                                std::vector<int>(num_sites, -1));
    std::vector<std::string> taxa_names(total_leaves);

    for (size_t i = 0; i < total_leaves; ++i) {
        auto [k, node_id] = leaf_global_order[i];
        const std::string& name = trees[k].node_names[node_id];
        if (!file_taxa_idx.count(name))
            throw std::runtime_error("Leaf '" + name + "' not found in observations file");
        taxa_names[i]    = name;
        observed_data[i] = raw_matrix[file_taxa_idx.at(name)];
        trees[k].tree[node_id].data = i;
    }

    // Assign globally-offset IDs to internal nodes
    size_t internal_counter = total_leaves;
    for (auto& t : trees)
        for (size_t node_id = 0; node_id < t.num_nodes; ++node_id)
            if (t.tree.out_degree(node_id) != 0)
                t.tree[node_id].data = internal_counter++;

    // Per-site alphabet sizes: max observed state + 1, floored at 1.
    std::vector<size_t> alphabet_sizes(num_sites, 1);
    for (const auto& row : observed_data)
        for (size_t j = 0; j < num_sites; ++j)
            if (row[j] >= 0 && static_cast<size_t>(row[j]) + 1 > alphabet_sizes[j])
                alphabet_sizes[j] = static_cast<size_t>(row[j]) + 1;

    size_t max_alphabet_size = *std::max_element(
        alphabet_sizes.begin(), alphabet_sizes.end());

    return process_input_result<std::vector<int>>{
        std::move(observed_data),
        meta_data{
            num_sites,
            max_alphabet_size,
            std::move(alphabet_sizes),
            std::move(taxa_names)
        }
    };
}

} // namespace phylocore
