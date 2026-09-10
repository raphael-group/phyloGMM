#include <iostream>
#include <random>

#include "simulator.h"

int main() {
    std::mt19937 rng(20260909);

    for (std::size_t num_leaves : {2U, 3U, 8U}) {
        const auto tree =
            phylocore::tests::phylo_gmm::simulate_random_binary_tree(
                num_leaves, rng);
        const double root_length = tree.branch_lengths.at(tree.root_id);
        if (root_length != 0.0) {
            std::cerr << "FAIL: phyloGMM simulator assigned root stem length "
                      << root_length << " for " << num_leaves << " leaves\n";
            return 1;
        }
    }

    std::cout << "PASS: phyloGMM simulator uses no root stem\n";
    return 0;
}
