#include "state_transition_models/laml.h"
#include "phylogeny.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "constants.h"
#include "math_utilities.h"

namespace phylocore {

// ── initialize_data ───────────────────────────────────────────────────────────

std::vector<laml_data> laml_model::initialize_data(
    const tree& t,
    std::vector<double>* buffer
) const {
    std::vector<laml_data> result(t.num_nodes);
    for (size_t i = 0; i < t.num_nodes; ++i) {
        result[i] = laml_data(
            buffer,
            std::log(parameters[1]),
            std::log(1.0 - parameters[1]),
            std::log(1.0 - std::exp(-t.branch_lengths[i])),
            std::log(1.0 - std::exp(-parameters[0] * t.branch_lengths[i]))
        );
    }
    return result;
}

// ── Constructor ───────────────────────────────────────────────────────────────

laml_model::laml_model(
    const std::vector<std::vector<int>>&                 character_matrix,
    const std::vector<std::vector<std::vector<double>>>& observation_matrix,
    const std::vector<std::vector<double>>&              mutation_priors,
    double nu,
    double phi,
    const std::string& data_type,
    bool   ultrametric,
    double min_branch_length,
    double timescale
) : character_matrix(character_matrix),
    observation_matrix(observation_matrix),
    mutation_priors(mutation_priors),
    data_type(data_type),
    ultrametric(ultrametric),
    min_branch_length(min_branch_length)
{
    parameters = {nu, phi, timescale};

    if (data_type == "character-matrix") {
        sites_latent_states.resize(character_matrix[0].size());
        for (size_t i = 0; i < sites_latent_states.size(); i++) {
            int max_state = 0;
            for (size_t j = 0; j < character_matrix.size(); j++)
                max_state = std::max(max_state, character_matrix[j][i]);
            sites_latent_states[i].alphabet_size = static_cast<size_t>(max_state) + 2;
        }
    } else {
        sites_latent_states.resize(observation_matrix[0].size());
        for (size_t i = 0; i < sites_latent_states.size(); i++) {
            size_t max_states = 0;
            for (size_t j = 0; j < observation_matrix.size(); j++)
                max_states = std::max(max_states, observation_matrix[j][i].size());
            sites_latent_states[i].alphabet_size = max_states + 1; // +1 for silenced state
        }
    }

    log_mutation_priors.resize(mutation_priors.size());
    for (size_t i = 0; i < mutation_priors.size(); i++) {
        log_mutation_priors[i].resize(mutation_priors[i].size());
        for (size_t j = 0; j < mutation_priors[i].size(); j++) {
            log_mutation_priors[i][j] = (mutation_priors[i][j] == 0.0)
                ? -1e12
                : std::log(mutation_priors[i][j]);
        }
    }

    // Root prior: probability 1 on the unedited state (index 1); 0 elsewhere.
    root_prior.resize(sites_latent_states.size());
    for (size_t c = 0; c < sites_latent_states.size(); ++c) {
        root_prior[c].assign(sites_latent_states[c].alphabet_size, 0.0);
        root_prior[c][1] = 1.0;
    }
}

// ── Transition matrix-vector products ────────────────────────────────────────

void laml_model::compute_log_pmatrix_vector_product(
    laml_data& d,
    size_t character,
    double branch_length,
    const std::vector<double>& log_vector,
    std::vector<double>& result
) const {
    size_t alphabet_size = sites_latent_states[character].alphabet_size;
    double nu = parameters[0];

    std::vector<double>& tmp = *(d.buffer);
    std::copy(log_vector.begin(), log_vector.begin() + alphabet_size, tmp.begin());

    // State 0 = missing: absorbing (missing stays missing)
    result[0] = log_vector[0];

    // State 1 = unedited: receives from all states
    tmp[0] += d.v2;
    tmp[1] += -branch_length * (1.0 + nu);
    for (size_t j = 2; j < alphabet_size; j++)
        tmp[j] += log_mutation_priors[character][j - 2] - branch_length * nu + d.v1;
    result[1] = log_sum_exp(tmp.begin(), tmp.begin() + alphabet_size);

    // States 2+ = mutated: self-transition and transition from missing
    double tmp2[] = {0.0, 0.0};
    for (size_t i = 2; i < alphabet_size; i++) {
        tmp2[0] = log_vector[i] - branch_length * nu;
        tmp2[1] = log_vector[0] + d.v2;
        result[i] = log_sum_exp(tmp2, tmp2 + 2);
    }
}

void laml_model::compute_log_pmatrix_transpose_vector_product(
    laml_data& d,
    size_t character,
    double branch_length,
    const std::vector<double>& log_vector,
    std::vector<double>& result
) const {
    size_t alphabet_size = sites_latent_states[character].alphabet_size;
    double nu = parameters[0];

    std::vector<double>& tmp = *(d.buffer);

    // State 0 = missing: receives from all states (non-missing can become silenced)
    tmp[0] = log_vector[0];
    for (size_t i = 1; i < alphabet_size; i++)
        tmp[i] = log_vector[i] + d.v2;
    result[0] = log_sum_exp(tmp.begin(), tmp.begin() + alphabet_size);

    // State 1 = unedited: self-transition only
    result[1] = -branch_length * (1.0 + nu) + log_vector[1];

    // States 2+ = mutated: from unedited (via mutation) and self-transition
    double tmp2[] = {0.0, 0.0};
    for (size_t i = 2; i < alphabet_size; i++) {
        tmp2[0] = log_vector[1] + log_mutation_priors[character][i - 2] - branch_length * nu + d.v1;
        tmp2[1] = log_vector[i] - branch_length * nu;
        result[i] = log_sum_exp(tmp2, tmp2 + 2);
    }
}

} // namespace phylocore
