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
FetchContent_MakeAvailable(csv-parser)

# ── Library ───────────────────────────────────────────────────────────────────
add_subdirectory(src/)

# ── Python bindings (optional) ────────────────────────────────────────────────
# Uncomment when bindings are added:
# find_package(pybind11 REQUIRED)
# add_subdirectory(python/)
