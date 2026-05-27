#ifndef PHYLOCORE_STATE_TRANSITION_MODELS_PMM_CTMC_H
#define PHYLOCORE_STATE_TRANSITION_MODELS_PMM_CTMC_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "../constants.h"
#include "../math_utilities.h"
#include "../phylogeny.h"  // for tree (needed by initialize_data)
#include "../state_transition_model.h"
#include "laml_data.h"

namespace phylocore {

/*!
 * @brief PMM (Probabilistic Mixture Model) CTMC over CRISPR-edit characters.
 *
 * State space (size k + 2 for k mutation categories at a given character):
 *   index 0  → silenced / dropped-out  (external label: -1 / missing)
 *   index 1  → unedited                (external label: 0)
 *   index 2+ → mutated to category 2..k+1
 *
 * Sparse closed-form transition kernel (no eigendecomposition needed):
 *   Let b = branch_length, nu = parameters[0].
 *   v1 = log(1 - exp(-b)), v2 = log(1 - exp(-nu*b))   (cached per node)
 *
 *   From unedited (1):
 *     P(1 -> 1)(b)        = exp(-(1 + nu) * b)
 *     P(1 -> silenced)(b) = 1 - exp(-nu * b)            [= exp(v2)]
 *     P(1 -> mutated α)(b) = q_c[α] * (1 - exp(-b)) * exp(-nu*b)
 *                          = exp(log q_c[α] + v1 - nu*b)
 *   From mutated α (>=2):
 *     P(α -> α)(b)        = exp(-nu * b)
 *     P(α -> silenced)(b) = 1 - exp(-nu * b)            [= exp(v2)]
 *     all other transitions are impossible.
 *   From silenced (0):
 *     P(0 -> 0)(b)        = 1; absorbing
 *
 * Parameters layout (state_transition_parameters):
 *   [0] = nu          (silencing rate)
 *   [1] = phi         (per-leaf dropout / silencing emission probability;
 *                      lives on the obs model conceptually, but LAML-Pro
 *                      ships it here so per-node laml_data can cache
 *                      log(phi) and log(1-phi))
 *   [2] = timescale   (unused at the math level; preserved for parity
 *                      with LAML-Pro's `laml_model::parameters` layout)
 *
 * Per-character mutation priors (q_c) live in mutation_priors[c]; entry
 * mutation_priors[c][α] gives the prior probability that a mutation event
 * at character c produces alphabet-state α+2 (so the prior vector at
 * character c has length alphabet_size - 2).
 *
 * The root prior is fixed: probability 1 on the unedited state (index 1),
 * 0 elsewhere. LAML's likelihood includes a non-trivial branch leading
 * INTO the root (it carries the silencing-rate ν contribution), so this
 * model has a meaningful root branch — see has_meaningful_root_branch.
 *
 * @tparam D Per-node data type. Defaults to laml_data; the optimised
 *           matvec assumes D exposes v1, v2, buffer (and the categorical
 *           observation model further assumes log_phi, log_one_minus_phi).
 */
template <typename D = laml_data>
class pmm_ctmc_model : public state_transition_model<D> {
public:
    using state_transition_model<D>::sites_latent_states;
    using state_transition_model<D>::root_prior;
    using state_transition_model<D>::state_transition_parameters;
    using state_transition_model<D>::has_optimized_functions;

    // ── Data layout ──────────────────────────────────────────────────────────

    std::vector<std::vector<double>> mutation_priors;     // [character][alpha]; raw
    std::vector<std::vector<double>> log_mutation_priors; // [character][alpha]; log

    // LAML's likelihood includes the branch leading into the root (it carries the
    // silencing rate ν contribution), so for pmm_ctmc_model has_meaningful_root_branch
    // is set to true in the constructor body (the base class attribute defaults to false).

    // ── Construction ──────────────────────────────────────────────────────────

    /*!
     * @param alphabet_sizes Per-character alphabet size = k_c + 2 where k_c is
     *                       the number of mutation categories at character c.
     * @param mutation_priors_in [character][alpha] with alpha in 0..alphabet_sizes[c] - 3
     *                       giving the prior probability of mutating to alphabet
     *                       state alpha + 2 at that character.
     * @param nu             Silencing rate.
     * @param phi            Per-leaf dropout / silencing emission probability.
     * @param timescale      Scale factor on branch lengths; preserved for
     *                       LAML-Pro parity (this class does not consume it).
     */
    pmm_ctmc_model(
        const std::vector<size_t>&              alphabet_sizes,
        const std::vector<std::vector<double>>& mutation_priors_in,
        double                                  nu,
        double                                  phi,
        double                                  timescale = 1.0)
        : mutation_priors(mutation_priors_in)
    {
        if (alphabet_sizes.empty())
            throw std::invalid_argument(
                "pmm_ctmc_model: alphabet_sizes must be non-empty");
        if (mutation_priors_in.size() != alphabet_sizes.size())
            throw std::invalid_argument(
                "pmm_ctmc_model: mutation_priors size " +
                std::to_string(mutation_priors_in.size()) +
                " != alphabet_sizes size " +
                std::to_string(alphabet_sizes.size()));

        state_transition_parameters = {nu, phi, timescale};

        sites_latent_states.resize(alphabet_sizes.size());
        for (size_t c = 0; c < alphabet_sizes.size(); ++c) {
            if (alphabet_sizes[c] < 2)
                throw std::invalid_argument(
                    "pmm_ctmc_model: alphabet_sizes[" + std::to_string(c) +
                    "] = " + std::to_string(alphabet_sizes[c]) + " < 2 (need silenced + unedited at minimum)");
            sites_latent_states[c].alphabet_size = alphabet_sizes[c];
            sites_latent_states[c].missing_index = 0; // index 0 = silenced
        }

        log_mutation_priors.resize(mutation_priors.size());
        for (size_t c = 0; c < mutation_priors.size(); ++c) {
            log_mutation_priors[c].resize(mutation_priors[c].size());
            for (size_t a = 0; a < mutation_priors[c].size(); ++a) {
                log_mutation_priors[c][a] = (mutation_priors[c][a] == 0.0)
                    ? NEGATIVE_INFINITY
                    : std::log(mutation_priors[c][a]);
            }
        }

        // Root prior: unedited (index 1) with probability 1; rest zero.
        root_prior.resize(sites_latent_states.size());
        for (size_t c = 0; c < sites_latent_states.size(); ++c) {
            root_prior[c].assign(sites_latent_states[c].alphabet_size, 0.0);
            root_prior[c][1] = 1.0;
        }

        // We provide an optimised matvec — switch on the fast path.
        this->has_optimized_functions = true;

        // The root branch carries the silencing rate ν contribution in LAML,
        // so include it in branch-length optimization (gtr_blens-style problems).
        this->has_meaningful_root_branch = true;
    }

    // ── Required scalar primitive ────────────────────────────────────────────
    //
    // log P(alpha -> beta along a branch of length b) at this character.
    // Derived directly from the closed-form kernel above.

    double compute_log_transition_probability(
        D& /*data*/,
        size_t character,
        double branch_length,
        size_t alpha,
        size_t beta
    ) const override {
        const double nu = state_transition_parameters[0];
        const size_t k  = sites_latent_states[character].alphabet_size;
        if (alpha >= k || beta >= k) return NEGATIVE_INFINITY;

        // State 0 = silenced (absorbing).
        if (alpha == 0) return (beta == 0) ? 0.0 : NEGATIVE_INFINITY;

        // State 1 = unedited.
        if (alpha == 1) {
            if (beta == 0)      return std::log1p(-std::exp(-nu * branch_length));     // 1 - e^{-nu b}
            if (beta == 1)      return -(1.0 + nu) * branch_length;                    // e^{-(1+nu)b}
            // beta >= 2: mutated category beta-2
            return log_mutation_priors[character][beta - 2]
                 + std::log1p(-std::exp(-branch_length))                               // 1 - e^{-b}
                 - nu * branch_length;
        }

        // State alpha >= 2 (mutated): can only self-transition or become silenced.
        if (beta == 0)      return std::log1p(-std::exp(-nu * branch_length));
        if (beta == alpha)  return -nu * branch_length;
        return NEGATIVE_INFINITY;
    }

    // ── Optimised matvec (forward + transpose, dispatched via the bool) ──────
    //
    // result[i] = log Σ_j P(t)_{i,j}  · exp(log_v[j])         (transpose=false)
    // result[i] = log Σ_j P(t)_{j,i}  · exp(log_v[j])         (transpose=true)
    //
    // Mirrors LAML-Pro::laml_model::compute_log_pmatrix_vector_product and
    // compute_log_pmatrix_transpose_vector_product in
    //   /Users/gc3045/Desktop/git/LAML-Pro/src/models/laml.cxx

    void compute_log_pmatrix_vector_product(
        D&                          data,
        size_t                      character,
        double                      branch_length,
        const std::vector<double>&  log_vector,
        std::vector<double>&        result,
        bool                        transpose
    ) const override {
        const size_t alphabet_size = sites_latent_states[character].alphabet_size;
        const double nu = state_transition_parameters[0];

        std::vector<double>& tmp = *(data.buffer);

        if (!transpose) {
            // FORWARD: result[i] = log Σ_j P_{i,j} · exp(log_v[j])
            std::copy(log_vector.begin(),
                      log_vector.begin() + alphabet_size,
                      tmp.begin());

            // i = 0 (silenced, absorbing): only j = 0 contributes.
            result[0] = log_vector[0];

            // i = 1 (unedited row of P): result[1] = log Σ_j P(1 -> j) · v[j].
            //   j = 0 (silenced):       P(1 -> 0) = 1 - e^{-νb} = exp(v2)
            //   j = 1 (self):           P(1 -> 1) = e^{-(1+ν)b}
            //   j >= 2 (mutated cat):   P(1 -> j) = q[j-2] · (1 - e^{-b}) · e^{-νb}
            tmp[0] += data.v2;
            tmp[1] += -branch_length * (1.0 + nu);
            for (size_t j = 2; j < alphabet_size; ++j) {
                tmp[j] += log_mutation_priors[character][j - 2]
                        - branch_length * nu
                        + data.v1;
            }
            result[1] = log_sum_exp(tmp.begin(), tmp.begin() + alphabet_size);

            // i >= 2 (mutated row): result[i] = log Σ_j P(i -> j) · v[j].
            //   Only two non-zero columns:
            //     j = i        (self):       P(i -> i) = e^{-νb}
            //     j = 0  (→ silenced):       P(i -> 0) = 1 - e^{-νb} = exp(v2)
            //   (LAML-Pro stashes "log_v[0] + v2" in tmp2[1] — same idea.)
            double tmp2[2];
            for (size_t i = 2; i < alphabet_size; ++i) {
                tmp2[0] = log_vector[i] - branch_length * nu;
                tmp2[1] = log_vector[0] + data.v2;
                result[i] = log_sum_exp(tmp2, tmp2 + 2);
            }
            return;
        }

        // TRANSPOSE: result[i] = log Σ_j P_{j,i} · exp(log_v[j])

        // i = 0 (silenced): receives from j = 0 (self), j = 1 (P(1->0) = 1-e^{-nu b}),
        // j >= 2 (P(α->0) = 1-e^{-nu b}). Both use v2.
        tmp[0] = log_vector[0];
        for (size_t j = 1; j < alphabet_size; ++j)
            tmp[j] = log_vector[j] + data.v2;
        result[0] = log_sum_exp(tmp.begin(), tmp.begin() + alphabet_size);

        // i = 1 (unedited): only j = 1 reaches unedited (P(1->1) = e^{-(1+nu)b}).
        result[1] = -branch_length * (1.0 + nu) + log_vector[1];

        // i >= 2 (mutated): from j = 1 (mutation event) or j = i (self).
        double tmp2[2];
        for (size_t i = 2; i < alphabet_size; ++i) {
            tmp2[0] = log_vector[1]
                    + log_mutation_priors[character][i - 2]
                    - branch_length * nu
                    + data.v1;
            tmp2[1] = log_vector[i] - branch_length * nu;
            result[i] = log_sum_exp(tmp2, tmp2 + 2);
        }
    }

    // ── Per-node cache ───────────────────────────────────────────────────────
    /*!
     * @brief Build the per-node laml_data cache for a tree.
     *
     * Computes v1 = log(1 - exp(-b_v)) and v2 = log(1 - exp(-nu * b_v))
     * for every node v in the tree, plus the (branch-length-independent)
     * log(phi) / log(1-phi) scalars consumed by the categorical observation
     * model.
     *
     * @param t       Tree whose branch_lengths are used.
     * @param buffer  Scratch buffer shared across nodes; size >= max alphabet.
     */
    std::vector<D> initialize_data(
        const tree& t,
        std::vector<double>* buffer
    ) const {
        const double nu  = state_transition_parameters[0];
        const double phi = state_transition_parameters[1];
        const double log_phi             = std::log(phi);
        const double log_one_minus_phi   = std::log(1.0 - phi);

        std::vector<D> result(t.num_nodes);
        for (size_t i = 0; i < t.num_nodes; ++i) {
            const double b = t.branch_lengths[i];
            result[i] = D(
                buffer,
                log_phi,
                log_one_minus_phi,
                std::log(1.0 - std::exp(-b)),
                std::log(1.0 - std::exp(-nu * b))
            );
        }
        return result;
    }
};

} // namespace phylocore

#endif // PHYLOCORE_STATE_TRANSITION_MODELS_PMM_CTMC_H
