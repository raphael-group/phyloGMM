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
    src/state_transition_models/gtr.cxx
    # src/state_transition_models/phylo_gmm_model.cxx  # disabled: pending migration (Task #9)
    # src/state_transition_models/laml.cxx              # disabled: pending migration (Task #9)
)

target_include_directories(phylocore_core PUBLIC include ${csv-parser_SOURCE_DIR}/include)
target_link_libraries(phylocore_core PUBLIC Eigen3::Eigen)

# ── Driver: run_gtr ──────────────────────────────────────────────────────────
add_executable(run_gtr src/run_gtr.cxx)
target_link_libraries(run_gtr PRIVATE phylocore_core csv)

# ── Tests / drivers (disabled until consumer fixes land) ─────────────────────
# test_gtr, test_gtr_raxml, test_phylo_gmm, test_laml, run_phylo_gmm all need
# to be updated for the new observation_model<D, O> + observed_data flow.

# ── Python bindings (optional) ────────────────────────────────────────────────
# Uncomment when bindings are added:
# find_package(pybind11 REQUIRED)
# add_subdirectory(python/)
