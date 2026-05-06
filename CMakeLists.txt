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
    src/models/gtr.cxx
    src/models/phylo_gmm_model.cxx
    src/models/laml.cxx
)

target_include_directories(phylocore_core PUBLIC include ${csv-parser_SOURCE_DIR}/include)
target_link_libraries(phylocore_core PUBLIC Eigen3::Eigen)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/test_gtr.cxx)
    add_executable(test_gtr src/test_gtr.cxx src/models/gtr.cxx)
    target_include_directories(test_gtr PRIVATE include)
    target_link_libraries(test_gtr PRIVATE Eigen3::Eigen)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/gtr_likelihood_raxml_validation/test_gtr_raxml.cxx)
    add_executable(test_gtr_raxml tests/gtr_likelihood_raxml_validation/test_gtr_raxml.cxx src/models/gtr.cxx)
    target_include_directories(test_gtr_raxml PRIVATE include)
    target_link_libraries(test_gtr_raxml PRIVATE Eigen3::Eigen)
    target_compile_definitions(test_gtr_raxml PRIVATE
        PHYLOCORE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    # Generate fixtures first: python3 tests/gtr_likelihood_raxml_validation/generate_fixture.py
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/phylo_gmm/test_phylo_gmm.cxx)
    add_executable(test_phylo_gmm
        tests/phylo_gmm/test_phylo_gmm.cxx
        src/models/gtr.cxx
        src/models/phylo_gmm_model.cxx)
    target_include_directories(test_phylo_gmm PRIVATE include)
    target_link_libraries(test_phylo_gmm PRIVATE Eigen3::Eigen)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/laml/test_laml.cxx)
    add_executable(test_laml
        tests/laml/test_laml.cxx
        src/models/laml.cxx)
    target_include_directories(test_laml PRIVATE include)
    target_link_libraries(test_laml PRIVATE Eigen3::Eigen)
endif()

# ── Driver: run phylo_gmm on a tree + observations CSV ───────────────────────
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/run_phylo_gmm.cxx)
    add_executable(run_phylo_gmm
        src/run_phylo_gmm.cxx
        src/io.cxx
        src/models/gtr.cxx
        src/models/phylo_gmm_model.cxx)
    target_include_directories(run_phylo_gmm PRIVATE include ${csv-parser_SOURCE_DIR}/include)
    target_link_libraries(run_phylo_gmm PRIVATE Eigen3::Eigen csv)
endif()

# ── Python bindings (optional) ────────────────────────────────────────────────
# Uncomment when bindings are added:
# find_package(pybind11 REQUIRED)
# add_subdirectory(python/)
