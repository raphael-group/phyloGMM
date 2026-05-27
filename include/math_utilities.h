#ifndef PHYLOCORE_MATH_UTILITIES_H
#define PHYLOCORE_MATH_UTILITIES_H

#include <array>
#include <cmath>
#include <algorithm>
#include <limits>

#include "extern/gcem.hpp"
#include "constants.h"

/*
 * Set to 1 to enable lookup-table approximations for exp/log.
 * Slightly alters likelihood values but can improve throughput.
 */
#define PHYLOCORE_FAST_LOG 0
#define PHYLOCORE_FAST_EXP 0

namespace phylocore {

constexpr int    TABLE_SIZE = 1000000;
constexpr double EXP_MIN    = -25.0;
constexpr double EXP_MAX    = +25.0;
constexpr double LOG_MIN    = 1e-7;
constexpr double LOG_MAX    = 5.0;

constexpr std::array<double, TABLE_SIZE + 1> make_exp_table() {
    std::array<double, TABLE_SIZE + 1> table{};
    for (int i = 0; i <= TABLE_SIZE; ++i) {
        double x = EXP_MIN + ((EXP_MAX - EXP_MIN) * i / TABLE_SIZE);
        table[i] = gcem::exp(x);
    }
    return table;
}

constexpr std::array<double, TABLE_SIZE + 1> make_log_table() {
    std::array<double, TABLE_SIZE + 1> table{};
    for (int i = 0; i <= TABLE_SIZE; ++i) {
        double x = LOG_MIN + ((LOG_MAX - LOG_MIN) * i / TABLE_SIZE);
        table[i] = gcem::log(x);
    }
    return table;
}

inline auto EXP_TABLE = make_exp_table();
inline auto LOG_TABLE = make_log_table();

inline double fast_exp(double x) {
    if (x <= EXP_MIN) return 0.0;
    if (x >= EXP_MAX) return 1e7;
    int idx = static_cast<int>((x - EXP_MIN) * TABLE_SIZE / (EXP_MAX - EXP_MIN));
    return EXP_TABLE[idx];
}

inline double fast_log(double x) {
    if (x <= LOG_MIN) return -16.118;
    if (x >= LOG_MAX) return 0.0;
    int idx = static_cast<int>((x - LOG_MIN) * TABLE_SIZE / (LOG_MAX - LOG_MIN));
    return LOG_TABLE[idx];
}

template <typename Iter>
typename std::iterator_traits<Iter>::value_type
log_sum_exp(Iter begin, Iter end) {
    using VT = typename std::iterator_traits<Iter>::value_type;
    if (begin == end) return VT{};

    auto max_val = *std::max_element(begin, end);
    if (max_val <= NEGATIVE_INFINITY) return max_val;  // finite sentinel; -ffast-math-safe

    VT sum{};
    for (auto it = begin; it != end; ++it) {
#if PHYLOCORE_FAST_EXP
        sum += fast_exp(*it - max_val);
#else
        sum += std::exp(*it - max_val);
#endif
    }

#if PHYLOCORE_FAST_LOG
    return max_val + fast_log(sum);
#else
    return max_val + std::log(sum);
#endif
}

} // namespace phylocore

#endif // PHYLOCORE_MATH_UTILITIES_H
