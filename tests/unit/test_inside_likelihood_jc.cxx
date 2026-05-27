// Unit test: compute_inside_log_likelihood on a 2-leaf JC tree.
//
// Verifies the per-site log-likelihood against the closed-form analytic
// answer for Jukes-Cantor. The JC rate matrix (after the
// reversible_ctmc_model normalization that sets expected substitutions per
// unit time to 1) has off-diagonal entries 1/3 and diagonal -1, so:
//
//   P_αα(t) = 1/4 + 3/4 · exp(-4t/3)
//   P_αβ(t) = 1/4 - 1/4 · exp(-4t/3)        (α ≠ β)
//
// On a tree {leaf0 ─ root ─ leaf1} with branch lengths t and stationary
// root prior π_i = 1/4:
//
//   L(s0, s1) = Σ_i (1/4) · P_{i,s0}(t) · P_{i,s1}(t)
//
// We check the per-site marginal log p(y) computed by log_p_y(0, 0) against
// the analytic formula for both the matching case (both leaves "A") and the
// mismatched case (leaves "A" / "C").
//
// This is the most basic sanity check on the inside-pass implementation; it
// runs in a few microseconds and rules out a whole class of bugs (wrong root
// prior application, incorrect transition probabilities, off-by-one in the
// postorder).

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "jc_fixture.h"

namespace {

double jc_analytic_log_likelihood(int s0, int s1, double t) {
    const double e    = std::exp(-4.0 * t / 3.0);
    const double diag = 0.25 + 0.75 * e;
    const double off  = 0.25 - 0.25 * e;
    double L = 0.0;
    for (int i = 0; i < 4; ++i) {
        const double p_i_s0 = (i == s0) ? diag : off;
        const double p_i_s1 = (i == s1) ? diag : off;
        L += 0.25 * p_i_s0 * p_i_s1;
    }
    return std::log(L);
}

} // namespace

int main() {
    using namespace phylocore::tests;

    const double t       = 0.1;
    const double TOL     = 1e-10;

    // ── Case 1: both leaves observe A (state 0) ─────────────────────────────
    {
        auto f = jc_fixture::make(/*obs0=*/0, /*obs1=*/0, /*branch=*/t);
        const double total = f.problem->log_likelihood();
        const double per_site = f.problem->log_p_y(0, 0);
        const double expected = jc_analytic_log_likelihood(0, 0, t);
        expect_near(per_site, expected, TOL,
                    "log_p_y per-site (both A)");
        // For a single-site problem the total equals the per-site marginal.
        expect_near(total, expected, TOL,
                    "log_likelihood total (both A)");
    }

    // ── Case 2: leaves observe A (0) and C (1) ──────────────────────────────
    {
        auto f = jc_fixture::make(/*obs0=*/0, /*obs1=*/1, /*branch=*/t);
        f.problem->log_likelihood();
        const double per_site = f.problem->log_p_y(0, 0);
        const double expected = jc_analytic_log_likelihood(0, 1, t);
        expect_near(per_site, expected, TOL,
                    "log_p_y per-site (A, C)");
    }

    // ── Case 3: longer branch (t = 0.5) to exercise different regime ────────
    {
        auto f = jc_fixture::make(/*obs0=*/2, /*obs1=*/3, /*branch=*/0.5);
        f.problem->log_likelihood();
        const double per_site = f.problem->log_p_y(0, 0);
        const double expected = jc_analytic_log_likelihood(2, 3, 0.5);
        expect_near(per_site, expected, TOL,
                    "log_p_y per-site (G, T at t=0.5)");
    }

    std::printf("test_inside_likelihood_jc: PASS\n");
    return 0;
}
