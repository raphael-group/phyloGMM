#ifndef PHYLOCORE_BRANCH_OPTIMIZATION_H
#define PHYLOCORE_BRANCH_OPTIMIZATION_H

#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "constants.h"
#include "phylogeny.h"

namespace phylocore {

namespace detail {

/*
 * Brent's method: finds x in [a, b] minimizing f(x).
 * Combines golden-section search with parabolic interpolation.
 */
inline double brent(
    const std::function<double(double)>& f,
    double a, double b,
    double tol     = 1e-8,
    int    max_iter = 100)
{
    constexpr double gold = 0.3819660112501051; // (3 - sqrt(5)) / 2

    double x = a + gold * (b - a);
    double w = x, v = x;
    double fx = f(x), fw = fx, fv = fx;
    double d = 0.0, e = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        double mid  = 0.5 * (a + b);
        double tol1 = tol * std::abs(x) + 1e-10;
        double tol2 = 2.0 * tol1;

        if (std::abs(x - mid) <= tol2 - 0.5 * (b - a)) return x;

        bool parabolic = false;
        if (std::abs(e) > tol1) {
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p; else q = -q;
            r = e; e = d;
            if (std::abs(p) < std::abs(0.5 * q * r) &&
                p > q * (a - x) && p < q * (b - x)) {
                d = p / q;
                double u = x + d;
                if ((u - a) < tol2 || (b - u) < tol2)
                    d = (x < mid) ? tol1 : -tol1;
                parabolic = true;
            }
        }
        if (!parabolic) {
            e = (x < mid) ? b - x : a - x;
            d = gold * e;
        }

        double u  = x + (std::abs(d) >= tol1 ? d : (d > 0 ? tol1 : -tol1));
        double fu = f(u);

        if (fu <= fx) {
            if (u < x) b = x; else a = x;
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x)       { v = w; fv = fw; w = u; fw = fu; }
            else if (fu <= fv || v == x || v == w) { v = u; fv = fu; }
        }
    }
    return x;
}

} // namespace detail

/*!
 * @brief Optimize all branch lengths via coordinate descent + Brent's method.
 *
 * Each cycle iterates over every node in postorder, holding all other branch
 * lengths fixed and finding the optimal length for that branch via a 1D line
 * search. Repeats until the total log-likelihood improvement falls below tol.
 *
 * scoring_fn must not modify branch lengths or model parameters — only read them.
 *
 * @param t          Tree to optimize (branch lengths modified in place)
 * @param scoring_fn Function returning total log-likelihood for a given tree/model
 * @param model      Model passed to scoring_fn
 * @param tol        Stop when a full cycle improves log-likelihood by less than this
 * @param max_cycles Maximum number of coordinate-descent cycles
 * @return           Log-likelihood at the optimized branch lengths
 */
template <typename D>
double optimize_branch_lengths(
    tree& t,
    const std::function<double(tree&, D&)>& scoring_fn,
    D&     model,
    double tol        = 1e-6,
    int    max_cycles = 100)
{
    double llh = scoring_fn(t, model);

    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        for (auto node_id : t.tree.postorder_traversal(t.root_id)) {
            double opt = detail::brent(
                [&](double b) {
                    t.branch_lengths[node_id] = b;
                    return -scoring_fn(t, model); // brent minimizes; we maximize llh
                },
                BRANCH_LENGTH_LB,
                BRANCH_LENGTH_UB
            );
            t.branch_lengths[node_id] = opt;
        }

        double new_llh = scoring_fn(t, model);
        if (new_llh - llh < tol) { llh = new_llh; break; }
        llh = new_llh;
    }

    return llh;
}

} // namespace phylocore

#endif // PHYLOCORE_BRANCH_OPTIMIZATION_H
