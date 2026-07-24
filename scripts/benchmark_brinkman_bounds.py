"""How well do simple mixing rules predict a mixed subresolution/resolved duct?

For a duct split into two regions (series = stacked along the flow z; parallel =
side by side across x) we compare the solver's effective permeability against
candidate estimators, using both the bulk input K and the solver's OWN
single-region K_eff (the latter cancels the duct convention offsets):

    harmonic  = weighted harmonic mean  (Wiener lower / exact series for Darcy)
    arithmetic= weighted arithmetic mean (Wiener upper / exact parallel for Darcy)

Goal: see which estimator is tight for (a) subres+subres (Darcy, both should be
exact for the matching arrangement) and (b) subres+open (only semi-analytical).
Exploration only -- prints a table, no assertions.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyflowsolver.brinkmanSolver import BrinkmanSolver

_OPEN = 1.0e12
W = H = 16
D = 16


def keff(Kfield):
    vol = np.ones((W, H, D), dtype=np.float64)
    if np.isscalar(Kfield):
        Kfield = np.full(vol.shape, Kfield, dtype=np.float64)
    s = BrinkmanSolver(vol, permeability=Kfield, scale=1.0,
                       fast_laplacian_guess=False, predictor="implicit",
                       max_iterations=12000, target_error=1e-10)
    s.solve()
    return s.effective_permeability()


def wmean_harm(vals, fracs):
    return 1.0 / sum(f / v for v, f in zip(vals, fracs))


def wmean_arith(vals, fracs):
    return sum(f * v for v, f in zip(vals, fracs))


def report(label, s_mixed, kA, kB, sA, sB, fracs):
    print(f"\n{label}: K_eff = {s_mixed:.5g}")
    for tag, a, b in (("bulk-K", kA, kB), ("single-region K_eff", sA, sB)):
        h = wmean_harm((a, b), fracs)
        ar = wmean_arith((a, b), fracs)
        print(f"   {tag:20s} harmonic={h:.5g} ({_re(s_mixed, h):+.1%})   "
              f"arithmetic={ar:.5g} ({_re(s_mixed, ar):+.1%})   "
              f"bracketed={'yes' if min(h, ar) <= s_mixed <= max(h, ar)*1.0001 else 'NO'}")


def _re(x, ref):
    return (x - ref) / ref if ref else np.nan


def run_case(name, kA, kB):
    print(f"\n{'='*70}\nCASE {name}:  K_A={kA:g}  K_B={kB:g}")
    sA, sB = keff(kA), keff(kB)
    print(f"single-region K_eff: A={sA:.5g}  B={sB:.5g}")

    # series: stack along z (both regions span the full width)
    dA = D // 2
    Kf = np.full((W, H, D), kA); Kf[:, :, dA:] = kB
    fr = (dA / D, (D - dA) / D)
    report("SERIES (stacked z)", keff(Kf), kA, kB, sA, sB, fr)

    # parallel: side by side across x
    wA = W // 2
    Kf = np.full((W, H, D), kA); Kf[wA:, :, :] = kB
    fr = (wA / W, (W - wA) / W)
    report("PARALLEL (side x)", keff(Kf), kA, kB, sA, sB, fr)


if __name__ == "__main__":
    # (a) subres + subres, Darcy limit -- expect series~harmonic, parallel~arith.
    run_case("subres+subres (mild contrast)", 0.02, 0.006)
    run_case("subres+subres (high contrast)", 0.05, 0.001)
    # (b) subres + open (resolved) -- semi-analytical.
    run_case("subres+open (mild)", 0.05, _OPEN)
    run_case("subres+open (strong drag)", 0.005, _OPEN)
