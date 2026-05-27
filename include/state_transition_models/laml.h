#ifndef PHYLOCORE_MODELS_LAML_H
#define PHYLOCORE_MODELS_LAML_H

#include <string>
#include <vector>

#include "../state_transition_model.h"
#include "laml_data.h"

namespace phylocore {

/*!
 * @brief LAML (Lineage Analysis via Maximum Likelihood) substitution model.
 *
 * Models CRISPR editing data as a non-reversible CTMC over states:
 *   index 0  → missing/silenced state  (-1 in external convention)
 *   index 1  → unedited state          ( 0 in external convention)
 *   index 2+ → mutated states          ( 1, 2, ... in external convention)
 *
 * Parameters (stored in the inherited parameters vector):
 *   parameters[0] = nu         (mutation rate)
 *   parameters[1] = phi        (dropout probability)
 *   parameters[2] = timescale
 *
 * Usage:
 *   1. Construct with character matrix or observation matrix, priors, and parameters.
 *   2. Call initialize_data() to build per-node caches from the tree.
 *   3. Pass the model and node_data to the standard phylocore likelihood traversals.
 */
class laml_model : public state_transition_model<laml_data> {
public:
    std::vector<std::vector<int>>                  character_matrix;   // [leaf_id][character]
    std::vector<std::vector<std::vector<double>>>  observation_matrix; // [leaf_id][character][state]
    std::vector<std::vector<double>>               mutation_priors;    // [character][state]
    std::vector<std::vector<double>>               log_mutation_priors;
    std::string data_type; // "character-matrix" | "observation-matrix"
    bool   ultrametric;
    double min_branch_length;

    laml_model(
        const std::vector<std::vector<int>>&                 character_matrix,
        const std::vector<std::vector<std::vector<double>>>& observation_matrix,
        const std::vector<std::vector<double>>&              mutation_priors,
        double nu,
        double phi,
        const std::string& data_type,
        bool   ultrametric       = false,
        double min_branch_length = 0.01,
        double timescale         = 1.0
    );

    void compute_log_pmatrix_vector_product(
        laml_data& d, size_t character, double branch_length,
        const std::vector<double>& log_vector, std::vector<double>& result
    ) const override;

    void compute_log_pmatrix_transpose_vector_product(
        laml_data& d, size_t character, double branch_length,
        const std::vector<double>& log_vector, std::vector<double>& result
    ) const override;

    /*!
     * @brief Build per-node caches from a tree and a shared scratch buffer.
     *
     * @param t      The tree whose branch lengths are used.
     * @param buffer Scratch vector (size ≥ max alphabet size) shared across nodes.
     */
    std::vector<laml_data> initialize_data(
        const tree& t,
        std::vector<double>* buffer
    ) const;
};

} // namespace phylocore

#endif // PHYLOCORE_MODELS_LAML_H
