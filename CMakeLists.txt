cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG   "-g --coverage")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")

project(phylocore VERSION 0.1)

# ── Third-party dependencies ──────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(csv-parser
    GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git
    GIT_TAG        master
)
set(CSV_CXX_STANDARD 17 CACHE STRING "" FORCE)
FetchContent_MakeAvailable(csv-parser)

FetchContent_Declare(Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
)
set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen)

# ── Library ───────────────────────────────────────────────────────────────────
add_library(phylocore_core STATIC
    src/io.cxx
    src/state_transition_models/reversible_ctmc.cxx
    src/state_transition_models/ctmc.cxx
    src/observation_models/gmm.cxx
    # src/state_transition_models/laml.cxx              # disabled: pending migration (Task #9)
)

target_include_directories(phylocore_core PUBLIC include ${csv-parser_SOURCE_DIR}/include)
target_link_libraries(phylocore_core PUBLIC Eigen3::Eigen)

# ── IPOPT (required by consumers that include optimize/direct.h) ────────────
# Pulled in per-executable rather than added to phylocore_core.
find_package(PkgConfig REQUIRED)
pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt>=3)

# ── Driver: run_gtr ──────────────────────────────────────────────────────────
add_executable(run_gtr src/run_gtr.cxx)
target_link_libraries(run_gtr PRIVATE phylocore_core csv PkgConfig::IPOPT)
target_compile_definitions(run_gtr
    PRIVATE PHYLOCORE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

# ── Driver: run_gtr_blens ────────────────────────────────────────────────────
# Branch-length-only GTR MLE via optimize::direct (IPOPT). Demonstrates the
# "narrow get_params" pattern: same problem class template as run_gtr, just a
# smaller parameter vector. IPOPT needed since this uses optimize::direct.
add_executable(run_gtr_blens src/run_gtr_blens.cxx)
target_link_libraries(run_gtr_blens PRIVATE phylocore_core csv PkgConfig::IPOPT)
target_compile_definitions(run_gtr_blens
    PRIVATE PHYLOCORE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

# ── Driver: run_gtr_em ───────────────────────────────────────────────────────
# Hobolth-Jensen EM driver. optimize/em.h is header-only and does NOT depend
# on IPOPT (Mode #5 in docs/parameter_estimation.md), so this target links
# only against phylocore_core.
add_executable(run_gtr_em src/run_gtr_em.cxx)
target_link_libraries(run_gtr_em PRIVATE phylocore_core csv)
target_compile_definitions(run_gtr_em
    PRIVATE PHYLOCORE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

# ── Driver: run_phylo_gmm ────────────────────────────────────────────────────
# phyloGMM EM driver. Closed-form M-step (Mode #5) — no IPOPT dependency.
add_executable(run_phylo_gmm src/run_phylo_gmm.cxx)
target_link_libraries(run_phylo_gmm PRIVATE phylocore_core)
target_compile_definitions(run_phylo_gmm
    PRIVATE PHYLOCORE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

# ── Driver: run_phylo_gmm_real ───────────────────────────────────────────────
# Real-data phyloGMM EM driver. Mirrors the CLI of the OLD phylo_gmm-branch
# `run_phylo_gmm` so the same fixtures can be fed to both implementations and
# the outputs diffed (cross-implementation comparison, Task #33). Separate
# target from `run_phylo_gmm` because the existing target embeds the synthetic
# validation (and we don't want to regress its CTest behaviour).
add_executable(run_phylo_gmm_real src/run_phylo_gmm_real.cxx)
target_link_libraries(run_phylo_gmm_real PRIVATE phylocore_core csv)
target_compile_definitions(run_phylo_gmm_real
    PRIVATE PHYLOCORE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

# ── Driver: run_laml_likelihood ──────────────────────────────────────────────
# Smoke driver for the new pmm_ctmc_model + pmm_categorical_observation_model.
# Likelihood evaluation only; no parameter optimisation → no IPOPT dependency.
add_executable(run_laml_likelihood src/run_laml_likelihood.cxx)
target_link_libraries(run_laml_likelihood PRIVATE phylocore_core csv)
target_compile_definitions(run_laml_likelihood
    PRIVATE PHYLOCORE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)
add_test(NAME test_run_laml_likelihood COMMAND run_laml_likelihood)

# ── Driver: run_laml ─────────────────────────────────────────────────────────
# LAML EM (Mode #3): inside/edge_inside/outside E-step + IPOPT-based M-step on
# the auxiliary Q-function over (ν, φ, branches). Links IPOPT because the
# M-step calls optimize::direct.
add_executable(run_laml src/run_laml.cxx)
target_link_libraries(run_laml PRIVATE phylocore_core csv PkgConfig::IPOPT)
target_compile_definitions(run_laml
    PRIVATE PHYLOCORE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)
add_test(NAME test_run_laml COMMAND run_laml)

# ── Driver: run_laml_pro ─────────────────────────────────────────────────────
# LAML-Pro EM (Mode #3): same shape as run_laml but with NATIVE parameters
# (no logit/log reparameterization) + explicit bounds via optimize::direct.
# Ultrametric constraints are deferred until the equality_constraint API lands
# (TODO in src/run_laml_pro.cxx).
add_executable(run_laml_pro src/run_laml_pro.cxx)
target_link_libraries(run_laml_pro PRIVATE phylocore_core csv PkgConfig::IPOPT)
target_compile_definitions(run_laml_pro
    PRIVATE PHYLOCORE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)
add_test(NAME test_run_laml_pro COMMAND run_laml_pro)

# Cross-check variant: run on the LAML-Pro n250 fixture if available.
# Loads PHYLOCORE_LAML_FIXTURE_DIR via test property. Skipped at CI time if
# the directory does not exist.
add_test(NAME test_run_laml_pro_fixture COMMAND run_laml_pro)
set_tests_properties(test_run_laml_pro_fixture PROPERTIES
    ENVIRONMENT "PHYLOCORE_LAML_FIXTURE_DIR=/Users/gc3045/Desktop/git/LAML-Pro/examples/n250_m30_character_matrix"
    TIMEOUT 600
)

# ── Tests / drivers (disabled until consumer fixes land) ─────────────────────
# test_gtr, test_phylo_gmm, test_laml, run_phylo_gmm all need
# to be updated for the new observation_model<D, O> + observed_data flow.

# ── Test: GTR vs RAxML-NG reference ──────────────────────────────────────────
add_executable(test_gtr_raxml
    tests/gtr_likelihood_raxml_validation/test_gtr_raxml.cxx
)
target_link_libraries(test_gtr_raxml PRIVATE phylocore_core csv)
target_compile_definitions(test_gtr_raxml
    PRIVATE PHYLOCORE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

# ── Smoke test: optimize::direct (IPOPT) ─────────────────────────────────────
add_executable(direct_smoke tests/optimize/direct_smoke.cxx)
target_link_libraries(direct_smoke PRIVATE phylocore_core PkgConfig::IPOPT)

# ── Test infra: phyloGMM synthetic data simulator ────────────────────────────
add_executable(simulate_phylo_gmm tests/phylo_gmm/simulate.cxx)
target_link_libraries(simulate_phylo_gmm PRIVATE phylocore_core)

# ── Unit tests (CTest harness) ───────────────────────────────────────────────
enable_testing()

function(add_phylocore_unit_test name)
    add_executable(${name} tests/unit/${name}.cxx)
    target_link_libraries(${name} PRIVATE phylocore_core)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_phylocore_unit_test(test_log_p_y_accessor)
add_phylocore_unit_test(test_node_posterior_normalization)
add_phylocore_unit_test(test_inside_likelihood_jc)
add_phylocore_unit_test(test_pmm_ctmc_inside_likelihood)

# ── phyloGMM random-init characterisation test ───────────────────────────────
# Tracks EM behaviour under RANDOM init only (no k-means++). The add_phylocore_unit_test
# macro hard-codes tests/unit/ as the source dir, so we add this test inline.
add_executable(test_random_init_phylo_gmm tests/phylo_gmm/test_random_init_phylo_gmm.cxx)
target_link_libraries(test_random_init_phylo_gmm PRIVATE phylocore_core)
add_test(NAME test_random_init_phylo_gmm COMMAND test_random_init_phylo_gmm)

# ── Python bindings (optional) ────────────────────────────────────────────────
# Uncomment when bindings are added:
# find_package(pybind11 REQUIRED)
# add_subdirectory(python/)
