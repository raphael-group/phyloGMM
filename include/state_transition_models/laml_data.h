#ifndef PHYLOCORE_STATE_TRANSITION_MODELS_LAML_DATA_H
#define PHYLOCORE_STATE_TRANSITION_MODELS_LAML_DATA_H

#include <vector>

namespace phylocore {

/*!
 * @brief Per-node cached log-space scalars for the LAML / PMM CTMC.
 *
 * All quantities are constant for a given branch length, so we cache them
 * once per node and reuse across characters in the inside / outside traversal.
 *
 *   v1 = log(1 - exp(-b))                where b = branch_length
 *   v2 = log(1 - exp(-nu * b))
 *
 * The two emission-related scalars (log_phi, log_one_minus_phi) are
 * independent of branch length but live here too so that the categorical
 * observation model can pick them up via the per-node data D, matching
 * LAML-Pro's existing laml_data layout 1:1.
 *
 * `buffer` is a pointer to a shared scratch vector (one shared across all
 * nodes for a given traversal) used by the optimised matvec to avoid
 * per-call allocation. Must have size >= max alphabet size at any site.
 *
 * Mirrors `struct laml_data` in
 *   /Users/gc3045/Desktop/git/LAML-Pro/include/models/laml.h
 * to keep the math (and any future joint optimisation problem) bit-for-bit
 * comparable across the two codebases.
 */
struct laml_data {
    std::vector<double>* buffer = nullptr;
    double log_phi             = 0.0;
    double log_one_minus_phi   = 0.0;
    double v1                  = 0.0;  // log(1 - exp(-b))
    double v2                  = 0.0;  // log(1 - exp(-nu * b))

    laml_data() = default;
    laml_data(std::vector<double>* buffer,
              double log_phi, double log_one_minus_phi,
              double v1, double v2)
        : buffer(buffer), log_phi(log_phi),
          log_one_minus_phi(log_one_minus_phi),
          v1(v1), v2(v2) {}
};

} // namespace phylocore

#endif // PHYLOCORE_STATE_TRANSITION_MODELS_LAML_DATA_H
