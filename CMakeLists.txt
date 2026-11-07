cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG   "-g --coverage")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")

project(phylogmm VERSION 1.0)

# GCC 8's libstdc++ does not implement floating-point std::from_chars.
# csv-parser/classify_scalar can fall back to a compatible parser when this
# macro is defined, which keeps builds working on older HPC/server toolchains.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
  add_compile_definitions(CLASSIFY_SCALAR_DISABLE_STD_FLOAT_FROM_CHARS)
endif()

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
# Eigen registers its own test suite when BUILD_TESTING is on; keep only phyloGMM's tests.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen)

# ── Library ───────────────────────────────────────────────────────────────────
add_library(phylocore_core STATIC
    src/io.cxx
    src/state_transition_models/reversible_ctmc.cxx
    src/state_transition_models/ctmc.cxx
    src/state_transition_models/melt.cxx
    src/observation_models/gmm.cxx
    src/observation_models/melt.cxx
)

target_include_directories(phylocore_core PUBLIC include ${csv-parser_SOURCE_DIR}/include)
target_link_libraries(phylocore_core PUBLIC Eigen3::Eigen)

# ── run_phylo_gmm ─────────────────────────────────────────────────────────────
# phyloGMM EM on one or more Newick trees + CSV observations.
add_executable(run_phylo_gmm src/run_phylo_gmm.cxx)
target_link_libraries(run_phylo_gmm PRIVATE phylocore_core csv)
target_compile_definitions(run_phylo_gmm
    PRIVATE PHYLOCORE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

# ── Tests (ctest) ─────────────────────────────────────────────────────────────
# Build and run with: cmake --build build -j 2 && (cd build && ctest)
enable_testing()

# Synthetic-data simulator used by the phyloGMM tests.
add_executable(simulate_phylo_gmm tests/phylo_gmm/simulate.cxx)
target_link_libraries(simulate_phylo_gmm PRIVATE phylocore_core)

function(add_phylocore_unit_test name)
    add_executable(${name} tests/unit/${name}.cxx)
    target_link_libraries(${name} PRIVATE phylocore_core)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_phylocore_unit_test(test_log_p_y_accessor)
add_phylocore_unit_test(test_node_posterior_normalization)
add_phylocore_unit_test(test_inside_likelihood_jc)
add_phylocore_unit_test(test_pmm_ctmc_inside_likelihood)
add_phylocore_unit_test(test_disjoint_taxa_check)

function(add_phylogmm_test name)
    add_executable(${name} tests/phylo_gmm/${name}.cxx)
    target_link_libraries(${name} PRIVATE phylocore_core)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# The simulator samples the root from pi with no incoming root stem.
add_phylogmm_test(test_phylo_gmm_simulator_root)
# fix_Q / fix_emission / stationary-pi flag combinations.
add_phylogmm_test(test_phylo_gmm_flags)
# Sufficient statistics summed over several trees equal the per-tree sums.
add_phylogmm_test(test_multi_tree_aggregation)
# Relabelling states leaves the likelihood unchanged (needs the robust
# matrix exponential for a reducible Q; see ctmc.h).
add_phylogmm_test(test_ctmc_permutation_invariance)

# Command-line tests of run_phylo_gmm (--init_params_json, objective weights,
# root stem handling).
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    foreach(t test_run_phylo_gmm_init_params test_run_phylo_gmm_weights test_run_phylo_gmm_root_stem)
        add_test(NAME ${t}
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/phylo_gmm/${t}.py
                         $<TARGET_FILE:run_phylo_gmm>)
    endforeach()
endif()
