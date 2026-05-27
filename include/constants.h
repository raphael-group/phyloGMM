#ifndef PHYLOCORE_CONSTANTS_H
#define PHYLOCORE_CONSTANTS_H

namespace phylocore {

// Finite sentinels — used instead of std::numeric_limits<double>::infinity()
// because the release build is compiled with -ffast-math (which implies
// -ffinite-math-only) under which the compiler is allowed to assume there are
// no infinities. Raw infinity() comparisons can silently produce wrong results.
constexpr double NEGATIVE_INFINITY = -1e8;  // "very low" log-probability sentinel
constexpr double POSITIVE_INFINITY =  1e8;  // "very high" worst-so-far / max sentinel

constexpr double BRANCH_LENGTH_LB  = 1e-6;
constexpr double BRANCH_LENGTH_UB  = 1e4;

// LAML-Pro parameter bounds (mirrors /Users/gc3045/Desktop/git/LAML-Pro/include/constants.h)
constexpr double NU_LB  = 1e-6;
constexpr double NU_UB  = 1e3;
constexpr double PHI_LB = 1e-6;
constexpr double PHI_UB = 1.0 - 1e-6;

} // namespace phylocore

#endif // PHYLOCORE_CONSTANTS_H
