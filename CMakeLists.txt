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

# ── Driver: run_gtr_em ───────────────────────────────────────────────────────
# Hobolth-Jensen EM driver. optimize/em.h is header-only and does NOT depend
# on IPOPT (Mode #5 in docs/parameter_estimation.md), so this target links
# only against phylocore_core.
add_executable(run_gtr_em src/run_gtr_em.cxx)
target_link_libraries(run_gtr_em PRIVATE phylocore_core csv)
target_compile_definitions(run_gtr_em
    PRIVATE PHYLOCORE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
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

# ── Python bindings (optional) ────────────────────────────────────────────────
# Uncomment when bindings are added:
# find_package(pybind11 REQUIRED)
# add_subdirectory(python/)
