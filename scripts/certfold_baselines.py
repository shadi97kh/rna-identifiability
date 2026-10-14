#!/usr/bin/env python3
"""CERTIFIED FOLDING, THE BASELINES. What the certificate's accuracy test compares against.

The accuracy test asks whether a certified pair is more often right than a pair some cheaper
signal is confident about. This module produces those cheaper signals, one sequence at a time:
SCORES (a number per base pair) and PREDICTIONS (a set of pairs). It never reads a reference
structure, never takes one as an argument, and computes no accuracy of any kind: accuracy on
TS0, the held-out set and bpRNA-new belongs after the registration, and the scores are written
here so that nothing about them can depend on it.

THE MODEL is pinned exactly as everywhere else in this repository: Turner 2004 in ViennaRNA 2.7.2
at 37 C, dangles 0, special hairpins off, through thermo.vienna_ref.reference_md() and nothing
else. ViennaRNA energies are integer dcal/mol; every shift and bound here is in kcal/mol.

THE BASELINES
  pair_probabilities    the Turner 2004 base-pair probability p_ij, from the plain compound
  mea_structures        ViennaRNA's maximum-expected-accuracy structure at several gammas
  gamma_centroid_pairs  Hamada et al. 2009's gamma-centroid estimator, from p_ij
  envelope_draws        parameter tables drawn uniformly from the published envelope (the
                        certificate's own primary set), as kcal/mol shifts over B["cols"]
  noise_draws           parameter tables with independent Gaussian noise on every entry,
                        the setting of Layton and Bundschuh 2005
  pair_frequencies      for a list of such shifts, how often each pair is in the MFE

The last three make the sampling baseline: the certificate says a pair survives EVERY table in
the set; the frequency says how many of a few hundred random tables it survives. They differ in
kind (for-all against most), and the test is whether that difference buys accuracy.

GLOBAL STATE. pair_frequencies loads parameter tables, which is process-global in ViennaRNA
(thermo/vienna_ref.py hazard 3, and the two last-model caches turner_par.py measured). Each table
goes through turner_par.loaded, which rebuilds both caches on entry and restores Turner 2004 on
exit; the compound each table is first read on is checked with params_match, the generating-set
rows whose energy the table moves most are checked against counts . theta, and every MFE's
ViennaRNA energy is checked against counts . theta as well. Nothing here is thread-safe; shard
by process.

SEEDS. Every draw uses numpy.random.default_rng(SeedSequence([SEED, tag])), SEED = 20261013 as in
scripts/certfold.py (a test holds the two equal). Tags in use there: 1, 2, 3, 10, 11, 12; in
scripts/certfold_data.py: 20, 21; in scripts/certfold_metrics.py: 40. Tags here: 30 =
envelope_draws, 31 = noise_draws (defaults; a different tag gives a different, equally valid
draw, recorded in the draw's record). A tag must be a plain int >= 0 (_rng).

Guards raise GuardFailure and are never repaired.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import math
import numbers
import os
import pathlib
import sys
import tempfile
import time
from collections.abc import Sequence

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.turner_typeb import GuardFailure  # noqa: E402
from scripts.typeb_common import INF_DCAL  # noqa: E402

SEED = 20261013                  # scripts/certfold.py's SEED; a test holds the two equal
SEED_TAG_ENVELOPE = 30           # envelope_draws
SEED_TAG_NOISE = 31              # noise_draws
GAMMAS = (0.5, 1.0, 2.0, 4.0)    # the gammas MEA and the gamma-centroid are reported at
NOISE_SIGMA_KCAL = 0.3           # the authors' default sigma for noise_draws
PROPOSALS_PER_DRAW = 100         # envelope_draws gives up past this many proposals per draw
CARRY_ROWS = 3                   # generating-set rows checked per loaded table
CARRY_SCAN = 1000                # ... found among at most this many of the most-moved rows
P_SLACK = 1e-9                   # rounding slack on a probability, and on a base's total


# ----------------------------------------------------------------------------------
# small guards
# ----------------------------------------------------------------------------------
def _check_seq(seq: str) -> None:
    """Refuse anything but a non-empty uppercase ACGU string.

    ViennaRNA maps an unknown letter to an unpairable N rather than erroring, and the count model
    raises a bare KeyError on it, so the refusal belongs here, where the message can say why.
    """
    if not isinstance(seq, str) or not seq or set(seq) - set("ACGU"):
        raise GuardFailure(f"expected a non-empty uppercase ACGU sequence, got {seq!r:.60}")


def _pairs(db: str) -> list[tuple[int, int]]:
    """The (i, j) pairs of a dot-bracket, 1-based with i < j, sorted.

    Its own parser rather than turner_nullspace.pair_table, because that one ends in a bare
    assert, and a structure ViennaRNA hands back unbalanced has to fail under python -O too.
    """
    stack, out = [], []
    for k, c in enumerate(db, start=1):
        if c == "(":
            stack.append(k)
        elif c == ")":
            if not stack:
                raise GuardFailure(f"unbalanced ')' at {k} in {db[:60]}")
            out.append((stack.pop(), k))
        elif c != ".":
            raise GuardFailure(f"unexpected {c!r} at {k} in {db[:60]}")
    if stack:
        raise GuardFailure(f"{len(stack)} unclosed '(' in {db[:60]}")
    return sorted(out)


def _nested(pairs) -> bool:
    """True if `pairs` is a secondary structure: no base twice, no two pairs crossing."""
    partner: dict[int, int] = {}
    for i, j in pairs:
        if not i < j or i in partner or j in partner:
            return False
        partner[i], partner[j] = j, i
    closing: list[int] = []
    for x in sorted(partner):
        if partner[x] > x:
            closing.append(partner[x])
        elif not closing or closing.pop() != x:
            return False
    return True


def _check_bpp(P: np.ndarray, n: int | None = None) -> None:
    """A base-pair probability matrix as pair_probabilities returns it, or GuardFailure.

    (n+1) x (n+1), 1-based, nonzero only strictly above the diagonal and off row and column 0,
    every entry in [0, 1], and no base paired with total probability above 1. A matrix that
    fails is not a probability matrix, and every estimator below would read it as one.
    """
    if P.ndim != 2 or P.shape[0] != P.shape[1] or P.shape[0] < 2:
        raise GuardFailure(f"a base-pair probability matrix is square (n+1)x(n+1); got {P.shape}")
    if n is not None and P.shape[0] != n + 1:
        raise GuardFailure(f"expected a {n + 1}x{n + 1} matrix for n = {n}; got {P.shape}")
    if not np.all(np.isfinite(P)):
        raise GuardFailure("the base-pair probability matrix has a non-finite entry")
    if P.min() < 0.0 or P.max() > 1.0 + P_SLACK:
        raise GuardFailure(f"a probability outside [0, 1]: [{P.min()}, {P.max()}]")
    if np.any(np.tril(P) != 0.0) or np.any(P[0] != 0.0) or np.any(P[:, 0] != 0.0):
        raise GuardFailure("the matrix carries weight on or below the diagonal, or at index 0; "
                           "pair_probabilities returns the upper triangle, 1-based")
    paired = P.sum(axis=0) + P.sum(axis=1)
    if paired.max() > 1.0 + P_SLACK:
        raise GuardFailure(f"base {int(paired.argmax())} is paired with total probability "
                           f"{paired.max():.12f} > 1")


# ----------------------------------------------------------------------------------
# 1-3: the ensemble, on the plain pinned compound
# ----------------------------------------------------------------------------------
def _pf_compound(seq: str):
    """The plain Turner 2004 compound for `seq`, with its partition function computed.

    The discipline certfold.certify_sequence uses, in its order: the energy table fingerprinted
    before anything is computed, the Boltzmann factors rescaled to the MFE (without that a long
    sequence overflows), and the Boltzmann table checked after pf(), because ViennaRNA caches it
    separately and the energy fingerprint cannot see it.
    """
    import RNA

    from thermo.vienna_ref import (
        assert_turner2004_boltzmann,
        assert_turner2004_params,
        reference_md,
    )

    _check_seq(seq)
    fc = RNA.fold_compound(seq, reference_md())
    assert_turner2004_params(fc)
    _star, e_mfe = fc.mfe()
    fc.exp_params_rescale(e_mfe)
    fc.pf()
    assert_turner2004_boltzmann(fc)
    return fc


def pair_probabilities(seq: str) -> np.ndarray:
    """The Turner 2004 base-pair probabilities of `seq`, a float64 (n+1) x (n+1) matrix, 1-based.

    P[i, j] for i < j is the probability that i pairs with j; the lower triangle, the diagonal
    and index 0 are zero. It is the score every per-pair baseline starts from, and the same
    matrix certfold.certify_sequence records beside each label, computed the same way.
    """
    fc = _pf_compound(seq)
    P = np.array(fc.bpp(), dtype=np.float64)
    _check_bpp(P, len(seq))
    return P


def mea_structures(seq: str, gammas: Sequence[float] = GAMMAS) -> dict[float, str]:
    """{gamma: dot-bracket}: ViennaRNA's maximum-expected-accuracy structure at each gamma.

    vrna_MEA maximises sum_{(i,j) in S} 2 gamma p_ij + sum_{i unpaired} p^u_i, so larger gamma
    buys sensitivity with specificity. That gamma is ViennaRNA's, not Hamada's: the two estimators
    share a symbol and a default grid, not a definition. All gammas are read off one partition
    function on the plain pinned compound (MEA leaves the probability matrix as it found it).
    """
    fc = _pf_compound(seq)
    out = {}
    for g in gammas:
        if not (isinstance(g, numbers.Real) and math.isfinite(g) and g > 0):
            raise GuardFailure(f"gamma must be a positive finite number, got {g!r}")
        s, _mea = fc.MEA(float(g))
        if len(s) != len(seq):
            raise GuardFailure(f"MEA at gamma = {g} returned {len(s)} characters for n = "
                               f"{len(seq)}")
        _pairs(s)
        out[g] = s
    return out


def _max_gain_nested(P: np.ndarray, thr: float) -> set[tuple[int, int]]:
    """The nested structure maximising sum over its pairs of (p_ij - thr): Hamada's gain.

    (gamma + 1) p_ij - 1 is (gamma + 1) times p_ij - thr, so the argmax is the same. A pair with
    p_ij <= thr can only lower the gain, so only the threshold set is a candidate, and every base
    carries fewer than 1/thr of them; the DP is Nussinov's, sparse in the candidates:
    F(i, j) = max(F(i+1, j), max over candidates (i, k), k <= j, of w_ik + F(i+1, k-1) + F(k+1, j)).
    The traceback recomputes each sum by the same expression, so float equality is exact.
    """
    n = P.shape[0] - 1
    cand: list[list[tuple[int, float]]] = [[] for _ in range(n + 2)]
    ii, jj = np.nonzero(np.triu(P, 1) > thr)
    for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
        cand[i].append((j, float(P[i, j]) - thr))
    for c in cand:
        c.sort()
    F = [[0.0] * (n + 2) for _ in range(n + 2)]
    for i in range(n, 0, -1):
        Fi, Fn, ci = F[i], F[i + 1], cand[i]
        if not ci:
            Fi[i:n + 1] = Fn[i:n + 1]
            continue
        for j in range(i, n + 1):
            best = Fn[j]
            for k, w in ci:
                if k > j:
                    break
                v = w + Fn[k - 1] + F[k + 1][j]
                if v > best:
                    best = v
            Fi[j] = best
    out: set[tuple[int, int]] = set()
    todo = [(1, n)]
    while todo:
        i, j = todo.pop()
        if i >= j:
            continue
        if F[i][j] == F[i + 1][j]:
            todo.append((i + 1, j))
            continue
        for k, w in cand[i]:
            if k > j:
                break
            if F[i][j] == w + F[i + 1][k - 1] + F[k + 1][j]:
                out.add((i, k))
                todo.extend([(i + 1, k - 1), (k + 1, j)])
                break
        else:
            raise GuardFailure(f"gamma-centroid traceback found no move at ({i}, {j})")
    return out


def gamma_centroid_pairs(bpp: np.ndarray,
                         gammas: Sequence[float] = GAMMAS) -> dict[float, set[tuple[int, int]]]:
    """{gamma: set of (i, j)}: Hamada et al. 2009's gamma-centroid estimator from `bpp`.

    The estimator maximises the expected gain sum_{(i,j) in S} ((gamma + 1) p_ij - 1) over
    secondary structures S. Every pair with p_ij > 1/(gamma + 1) has positive gain, so whenever
    that threshold set is itself a secondary structure it IS the estimator, and that is the
    answer returned. For gamma <= 1 the threshold is at least 0.5, and two pairs above 0.5 can
    neither share a base nor cross (their probabilities would sum past 1), so there the estimator
    is always the threshold set, and a threshold set that is not a structure means `bpp` is not a
    probability matrix: GuardFailure. gamma = 1 is the ordinary centroid, the pairs with
    p_ij > 0.5, which is ViennaRNA's fold_compound.centroid(); a test holds the two equal.

    For gamma > 1 the threshold set can put one base in two pairs or cross two pairs, and is
    then not a prediction at all; there the estimator is the maximum-gain structure, which a
    Nussinov-style DP over the threshold set finds (_max_gain_nested), as Hamada et al. do. The
    result is always a subset of the threshold set, and always nested.
    """
    P = np.asarray(bpp, dtype=np.float64)
    _check_bpp(P)
    out = {}
    for g in gammas:
        if not (isinstance(g, numbers.Real) and math.isfinite(g) and g > 0):
            raise GuardFailure(f"gamma must be a positive finite number, got {g!r}")
        thr = 1.0 / (g + 1.0)
        ii, jj = np.nonzero(np.triu(P, 1) > thr)
        above = {(int(i), int(j)) for i, j in zip(ii, jj, strict=True)}
        if _nested(above):
            out[g] = above
            continue
        if g <= 1:
            raise GuardFailure(f"at gamma = {g} (threshold {thr:.3f} >= 0.5) the pairs above "
                               f"the threshold conflict; bpp is not a probability matrix")
        got = _max_gain_nested(P, thr)
        if not _nested(got) or not got <= above:
            raise GuardFailure(f"the gamma = {g} DP returned a set that is not a nested subset "
                               f"of the threshold set")
        out[g] = got
    return out


# ----------------------------------------------------------------------------------
# 4-5: the draws
# ----------------------------------------------------------------------------------
def _rng(seed_tag: int) -> np.random.Generator:
    """default_rng(SeedSequence([SEED, seed_tag])), refusing a tag that is not a plain int >= 0.

    As scripts/certfold_data.py: SeedSequence takes True as 1, so without this a bool tag would
    silently replay tag 1's stream (certfold's gate draw) instead of failing.
    """
    if isinstance(seed_tag, bool) or not isinstance(seed_tag, int) or seed_tag < 0:
        raise GuardFailure(f"seed_tag {seed_tag!r} is not a non-negative int")
    return np.random.default_rng(np.random.SeedSequence([SEED, seed_tag]))


def _check_n_draws(n_draws: int) -> None:
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 1:
        raise GuardFailure(f"n_draws must be an int >= 1, got {n_draws!r}")


def _summary(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"min": float(x.min()), "median": float(np.median(x)), "max": float(x.max())}


def envelope_draws(B: dict, n_draws: int = 200, seed_tag: int = SEED_TAG_ENVELOPE,
                   lam: float = 1.0, *, max_proposals: int | None = None
                   ) -> tuple[list[np.ndarray], dict]:
    """`n_draws` tables drawn uniformly from the published envelope scaled by `lam`.

    The envelope is certfold's primary set, E = {theta0 + B4 t : ||t|| <= lam r_max and
    |(B4 t)_e| <= lam delta for every entry e}, with B4, r_max and delta from load_tr0_basis.
    A proposal is uniform in the k-ball of radius r_max (k = B4.shape[1]): a Gaussian direction,
    radius r_max U^(1/k). It is ACCEPTED only if |(B4 t)_e| <= delta on every entry, which makes
    the accepted draws uniform on the envelope (the ball conditioned on the box). The accepted
    t is then scaled by lam. Both constraints scale with lam, so acceptance does not depend on
    it, and the same seed gives the SAME directions at every lam: the arms differ by scale alone
    (common random numbers). In k = 448 dimensions U^(1/k) is close to 1, so the draws sit near
    the sphere; that is what uniform in a high-dimensional ball means, not a defect.

    Returns (shifts, record): each shift is a float64 vector in kcal/mol over B["cols"], B4 t on
    B["used"] and exactly zero everywhere else. The record carries the seed, the number of
    proposals and the acceptance rate, and the realised extremes of both constraints.
    """
    if not (isinstance(lam, numbers.Real) and math.isfinite(lam) and lam > 0):
        raise GuardFailure(f"lam must be a positive finite number, got {lam!r}")
    _check_n_draws(n_draws)
    rng = _rng(seed_tag)
    from scripts.certfold import ROW_NOISE

    B4 = np.asarray(B["B4"], dtype=np.float64)
    used = np.asarray(B["used"])
    cols = B["cols"]
    delta = float(B["envelope"]["delta_kcal"])
    r_max = float(B["envelope"]["r_max_kcal"])
    k = B4.shape[1]
    if B4.shape[0] != len(used) or len(B["u_env"]) != len(cols):
        raise GuardFailure(f"B4 is {B4.shape} for {len(used)} used of {len(cols)} columns, and "
                           f"u_env has {len(B['u_env'])} entries")
    # the record's delta and r_max must be the ones the per-entry bound u_env was built from,
    # or the set drawn from here is not the set the certificate labels against
    a = np.asarray(B["a_four"], dtype=np.float64)
    u_want = np.where(a > ROW_NOISE, np.minimum(delta, r_max * a), 0.0)
    if float(np.abs(u_want - np.asarray(B["u_env"])).max()) > 1e-12:
        raise GuardFailure("B['u_env'] is not min(delta, r_max ||a_e||) on the reached entries: "
                           "the envelope record and its per-entry bound disagree")
    cap = max_proposals if max_proposals is not None else PROPOSALS_PER_DRAW * n_draws
    shifts, norms, worst_box = [], [], []
    proposals = 0
    while len(shifts) < n_draws:
        if proposals >= cap:
            raise GuardFailure(f"{len(shifts)} of {n_draws} envelope draws accepted after "
                               f"{proposals} proposals; rejection would not finish")
        proposals += 1
        z = rng.standard_normal(k)
        u = rng.random()
        t1 = (r_max * u ** (1.0 / k) / np.linalg.norm(z)) * z
        y1 = B4 @ t1
        if np.abs(y1).max() > delta:
            continue
        shift = np.zeros(len(cols), dtype=np.float64)
        shift[used] = lam * y1
        shifts.append(shift)
        norms.append(float(np.linalg.norm(lam * t1)) / (lam * r_max))
        worst_box.append(float(np.abs(lam * y1).max()) / (lam * delta))
    u_env = np.asarray(B["u_env"], dtype=np.float64)
    reach = u_env > 0
    over_u = max(float((np.abs(s[reach]) / (lam * u_env[reach])).max()) for s in shifts)
    off_reach = max(float(np.abs(s[~reach]).max()) for s in shifts)
    if off_reach != 0.0:
        raise GuardFailure(f"an envelope draw moves an entry the envelope does not reach "
                           f"(|shift| = {off_reach:.3e}); B4 has weight off its reached rows")
    return shifts, {
        "kind": "envelope",
        "definition": "uniform on {t : ||t|| <= lam r_max, |(B4 t)_e| <= lam delta for every e} "
                      "by rejection from the uniform k-ball (Gaussian direction, radius "
                      "r_max U^(1/k)); the box is tested at lam = 1 and the accepted t scaled by "
                      "lam; shift = B4 t on B['used'], 0 elsewhere, kcal/mol",
        "seed": [SEED, int(seed_tag)], "lam": float(lam), "k": int(k),
        "r_max_kcal": r_max, "delta_kcal": delta,
        "n_draws": int(n_draws), "n_proposals": int(proposals),
        "acceptance_rate": n_draws / proposals,
        "norm_t_over_lam_r_max": _summary(norms),
        "max_abs_entry_over_lam_delta": _summary(worst_box),
        "max_over_draws_abs_shift_over_lam_u_env": over_u,
        "n_entries_reached": int(reach.sum()),
    }


def noise_draws(cols: Sequence[tuple], n_draws: int = 200, seed_tag: int = SEED_TAG_NOISE,
                sigma_kcal: float = NOISE_SIGMA_KCAL, *, theta0: dict | None = None
                ) -> tuple[list[np.ndarray], dict]:
    """`n_draws` shift vectors with independent Normal(0, sigma_kcal) on every finite entry.

    The setting of Layton and Bundschuh 2005: every parameter perturbed independently. sigma is
    the authors' choice (0.3 kcal/mol by default), not a value read from that paper. "Every
    parameter" means every PHYSICAL parameter: two file entries that describe one nearest-
    neighbour from opposite ends (stack[a][b] and its 180-degree partner, the symmetric int11 and
    int22 cells) are one parameter, so they receive ONE draw, shared -- turner_nullspace.tie_map's
    classes. Drawing them independently would break the 180-degree flip invariance the model has
    (CLAUDE.md lists it as an invariant) and perturb a model that no longer exists.

    An entry that is INF in `theta0` (Turner 2004 by default; typeb_common.INF_DCAL) is a
    forbidden motif and gets a shift of exactly zero: noise must not make it allowed. Turner
    2004 has no INF entry among the parsed columns, so there the rule is vacuous, and the record
    says so by counting them. Returns (shifts, record), shifts in kcal/mol over `cols`.
    """
    if not (isinstance(sigma_kcal, numbers.Real) and math.isfinite(sigma_kcal)
            and sigma_kcal > 0):
        raise GuardFailure(f"sigma_kcal must be a positive finite number, got {sigma_kcal!r}")
    _check_n_draws(n_draws)
    rng = _rng(seed_tag)
    if theta0 is None:
        from scripts.turner_par import template

        theta0, base = template()["theta"], "Turner 2004 (turner_par.template)"
    else:
        base = "the caller's theta0"
    cols = list(cols)
    if len(set(cols)) != len(cols):
        raise GuardFailure("cols has a repeated key")
    missing = [c for c in cols if c not in theta0]
    if missing:
        raise GuardFailure(f"{len(missing)} columns are not in theta0, e.g. {missing[:3]}")
    from scripts.turner_nullspace import tie_map

    t0 = np.array([theta0[c] for c in cols], dtype=np.int64)
    finite = t0 < INF_DCAL
    m = int(finite.sum())
    tied, T = tie_map(cols)                   # T[e, c] = 1 iff entry e belongs to class c
    cls = T.argmax(axis=1)
    shifts = []
    for _ in range(n_draws):
        z = rng.normal(0.0, float(sigma_kcal), len(tied))
        s = np.where(finite, z[cls], 0.0)
        shifts.append(s)
    allv = np.concatenate([s[finite] for s in shifts]) if m else np.zeros(1)
    return shifts, {
        "kind": "noise",
        "definition": "independent Normal(0, sigma) in kcal/mol per physical parameter "
                      "(turner_nullspace.tie_map class), shared by the class's file entries; "
                      "entries INF in theta0 shifted by exactly 0",
        "n_tie_classes": len(tied),
        "seed": [SEED, int(seed_tag)], "sigma_kcal": float(sigma_kcal), "theta0": base,
        "n_draws": int(n_draws), "n_entries": len(cols), "n_entries_perturbed": m,
        "n_entries_inf_left_alone": int((~finite).sum()),
        "realised_sd_kcal": float(allv.std()), "max_abs_shift_kcal": float(np.abs(allv).max()),
    }


# ----------------------------------------------------------------------------------
# 6: MFE pair frequencies over drawn tables
# ----------------------------------------------------------------------------------
def shifted_theta(theta0: dict, cols: Sequence[tuple], shift: np.ndarray
                  ) -> tuple[dict, float, int]:
    """theta0 + shift as a writable table: (theta, max rounding in kcal/mol, entries changed).

    The parameter file holds integers in dcal/mol, so theta0[c] + 100 shift_c is rounded to the
    nearest integer (ties to even, as turner_par.round_to_format), and the largest residual is
    returned rather than dropped. An INF entry is left exactly as it is, and a NONZERO shift on
    one is refused rather than ignored: the caller asked to move a forbidden motif. Keys theta0
    lacks keep Turner 2004, which is what write_par writes for them, so the returned table is
    the written one entry for entry and params_match and counts . theta can read it.
    """
    from scripts.turner_par import template

    cols = list(cols)
    shift = np.asarray(shift, dtype=np.float64)
    if shift.shape != (len(cols),):
        raise GuardFailure(f"a shift of shape {shift.shape} for {len(cols)} columns")
    if not np.all(np.isfinite(shift)):
        raise GuardFailure("a shift has a non-finite entry")
    missing = [c for c in cols if c not in theta0]
    if missing:
        raise GuardFailure(f"{len(missing)} columns are not in theta0, e.g. {missing[:3]}")
    bad = [c for c in cols if not isinstance(theta0[c], numbers.Integral)]
    if bad:
        raise GuardFailure(f"theta0 is integer dcal/mol; {bad[:3]} are not integers")
    t0 = np.array([int(theta0[c]) for c in cols], dtype=np.int64)
    inf = t0 >= INF_DCAL
    if np.any(shift[inf] != 0.0):
        raise GuardFailure(f"{int((shift[inf] != 0).sum())} nonzero shifts on INF entries of "
                           f"theta0; a forbidden motif is never moved")
    v = t0 + 100.0 * shift
    r = np.where(inf, t0, np.rint(v).astype(np.int64))
    if np.any(r[~inf] >= INF_DCAL):
        raise GuardFailure("a shifted entry reaches INF")
    resid = float(np.abs(r - v)[~inf].max()) / 100.0 if (~inf).any() else 0.0
    theta = dict(template()["theta"]) | dict(theta0)
    theta.update({c: int(x) for c, x in zip(cols, r.tolist(), strict=True)})
    return theta, resid, int((r != t0).sum())


@functools.lru_cache(maxsize=1)
def _generating_counts():
    """The generating set's count matrix over Turner 2004's keys, built once per process.

    The rows turner_par's W3/W4 use to prove ViennaRNA's reader exact, d0 and special_hp 0. A
    loaded table is checked on the rows whose energy it moves most, because params_match reads
    twelve 1-D and 2-D entries and cannot see the int11/int21/int22 tables at all, which are
    exactly the tables an envelope draw moves.
    """
    from scripts.turner_nullspace import Extractor, build_matrix, generating_set
    from scripts.turner_par import template

    T = template()
    keys = list(T["theta"])
    gen = generating_set(T["special"])
    ex = Extractor(T["special"], 0, 0)
    G = build_matrix([ex.counts(s, d) for _, s, d in gen], keys)
    t04 = np.array([T["theta"][k] for k in keys], dtype=np.int64)
    return gen, keys, G, t04


def _ninio_correction(db: str, m: int, cap: int) -> int:
    """ViennaRNA's Ninio term minus the count model's linearisation, exactly, in dcal/mol.

    typeb_common.ninio_correction, with one case it gets wrong. ViennaRNA 2.7.2's E_IntLoop adds
    MIN2(ninio_max, asym * ninio_m) to a generic or 1xn interior loop, but a 2x3 loop gets
    internal_loop[5] + ninio_m + mismatch23I, ninio_m UNCLIPPED. The count model counts ninio_m
    once for a 2x3 loop, so the true correction there is 0 for every table; the shared function
    returns MIN2(cap, m) - m, which is nonzero once a table has m > cap. Measured (a written
    table with m = 400, cap = 300): a 2x3 loop evaluates to counts . theta exactly, and the
    shared correction is 100 dcal off, so the MFE-energy guard would raise on a correct fold.
    Turner 2004 (m = 60, cap = 300) is far from that; a Gaussian draw at sigma = 1 kcal/mol
    reaches it in a few percent of tables. Bulges and the 1x1, 1x2, 2x2 table loops carry no
    Ninio term.
    """
    pairs = _pairs(db)
    pt = {}
    for i, j in pairs:
        pt[i], pt[j] = j, i
    corr = 0
    for i, j in pairs:
        inner, k = [], i + 1
        while k < j:
            if pt.get(k, 0) > k:
                inner.append((k, pt[k]))
                k = pt[k] + 1
            else:
                k += 1
        if len(inner) != 1:
            continue
        p, q = inner[0]
        ns, nl = sorted((p - i - 1, j - q - 1))
        if ns == 0 or (ns, nl) in ((1, 1), (1, 2), (2, 2)):
            continue                            # a stack, a bulge, or a table loop
        asym = nl - ns
        vienna = m if (ns, nl) == (2, 3) else min(cap, asym * m)
        corr += vienna - (asym * m if asym <= 5 else cap)
    return corr


def _energy(counts: dict, theta: dict) -> int:
    """counts . theta in dcal/mol, with the count model's fixed long-loop offset."""
    off = counts.get(("_offset",), 0)
    return int(sum(v * theta[k] for k, v in counts.items() if k != ("_offset",))) + int(off)


def _carry(theta: dict, md) -> list[dict]:
    """eval_structure == counts . theta on the CARRY_ROWS generating rows the table moves most.

    Moved relative to Turner 2004, because the failure this guards is a stale table, and the
    stale table is Turner 2004 (loaded restores it, and _rebuild runs after every load). If the
    table IS Turner 2004 on every entry, no row moves and the list is empty: there is nothing a
    stale table could get wrong. Rows touching an INF entry are skipped (a forbidden motif has
    no finite energy to compare). Rows are ranked by the LINEAR change |c . (theta - theta04)|,
    but a row whose TRUE energy does not change is skipped: where the Ninio cap absorbs the
    linear change (ninio_m raised past ninio_max / asym), Turner 2004 gives the same energy, so a
    stale table would pass that row and the check would test nothing.
    """
    import RNA

    gen, keys, G, t04 = _generating_counts()
    thv = np.array([theta[k] for k in keys], dtype=np.int64)
    inf = thv >= INF_DCAL
    ch = np.abs(G @ (np.where(inf, 0, thv) - t04)).astype(np.float64)
    if inf.any():
        touches = np.asarray((G[:, np.flatnonzero(inf)] != 0).sum(axis=1)).ravel() > 0
        ch[touches] = -1.0
    m, cap = theta[("ninio_m",)], theta[("ninio_max",)]
    m0, cap0 = (int(t04[keys.index(k)]) for k in (("ninio_m",), ("ninio_max",)))
    out = []
    for r in np.argsort(-ch, kind="stable")[:CARRY_SCAN]:
        if ch[r] <= 0 or len(out) == CARRY_ROWS:
            break
        _tag, s, db = gen[int(r)]
        want = int(G[int(r)].dot(thv)[0]) + _ninio_correction(db, m, cap)
        if want == int(G[int(r)].dot(t04)[0]) + _ninio_correction(db, m0, cap0):
            continue                            # the cap absorbs it: a stale table passes too
        got = int(round(RNA.fold_compound(s, md).eval_structure(db) * 100))
        if got != want:
            raise GuardFailure(f"the loaded table is not the written one: generating row "
                               f"{int(r)} evaluates to {got} dcal, counts . theta gives {want}")
        out.append({"row": int(r), "dcal": got,
                    "change_vs_turner2004_dcal": got - int(G[int(r)].dot(t04)[0])})
    return out


def _sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pair_frequencies(seqs: Sequence[str], shifts: Sequence[np.ndarray], theta0: dict,
                     scratch_dir, *, cols: Sequence[tuple], on_fold=None
                     ) -> tuple[list[dict[tuple[int, int], float]], dict]:
    """For each sequence, the fraction of tables theta0 + shift whose MFE contains each pair.

    freq[k][(i, j)] is that fraction for seqs[k], over all len(shifts) tables; a pair absent
    from freq[k] has frequency 0. `cols` names the entries the shift vectors index (B["cols"]
    for envelope_draws, the `cols` given to noise_draws), and it is required because a vector
    without its keys can be added to the wrong table silently.

    Per table, in order: theta0 + shift rounded to integer dcal (shifted_theta, INF untouched),
    written with turner_par.write_par to one temporary file in `scratch_dir` (which must exist;
    the file is removed at the end and its sha256 kept per draw), loaded ONCE with
    turner_par.loaded, and EVERY sequence folded under it, since loading is global. Three
    guards per table, each raising GuardFailure: params_match on the first compound; the
    generating rows the table moves most (_carry); and on every fold, ViennaRNA's MFE energy
    equal to the count model's counts . theta (with _ninio_correction for the one
    nonlinearity a perturbed ninio pair exposes). Afterwards Turner 2004 is asserted restored
    on a fresh compound, energy table and Boltzmann factors both.

    `on_fold(d, k, structure)`, if given, is called after every guarded fold: G5 uses it to
    check each MFE under an attacking table against the certificate, through the same guards.

    The record carries per draw the max rounding error in kcal/mol, the params_match result,
    the carry rows, the number of entries the rounded table changes, and how many sequences'
    MFE differs from their Turner 2004 MFE (a structure-to-structure count, not an accuracy).
    """
    import RNA

    from scripts.turner_nullspace import Extractor
    from scripts.turner_par import loaded, params_match, template, write_par
    from thermo.vienna_ref import (
        assert_turner2004_boltzmann,
        assert_turner2004_params,
        reference_md,
    )

    t_start = time.time()
    seqs, shifts, cols = list(seqs), list(shifts), list(cols)
    if not seqs or not shifts:
        raise GuardFailure(f"{len(seqs)} sequences and {len(shifts)} shifts: nothing to count")
    for s in seqs:
        _check_seq(s)
    if len(set(cols)) != len(cols):
        raise GuardFailure("cols has a repeated key")
    scratch = pathlib.Path(scratch_dir)
    if not scratch.is_dir():
        raise GuardFailure(f"scratch_dir {scratch} is not an existing directory")
    T = template()
    t2004 = T["theta"]
    ex = Extractor(T["special"], 0, 0, lxc=T["lxc"])
    _generating_counts()                      # built before any load, so its time is not a draw's

    md = reference_md()
    star0 = []
    for k, seq in enumerate(seqs):
        fc = RNA.fold_compound(seq, md)
        if k == 0:
            assert_turner2004_params(fc)
        star0.append(fc.mfe()[0])

    tally = [collections.Counter() for _ in seqs]
    draws, n_folds = [], 0
    fd, tmp = tempfile.mkstemp(prefix="baseline_draw_", suffix=".par", dir=scratch)
    os.close(fd)
    path = pathlib.Path(tmp)
    try:
        for d, shift in enumerate(shifts):
            theta, resid, n_changed = shifted_theta(theta0, cols, shift)
            write_par(theta, path)
            sha = _sha256(path)
            differs = 0
            with loaded(path):
                md = reference_md()
                first = RNA.fold_compound(seqs[0], md)
                bad = params_match(first, theta, 0, 0)
                if bad:
                    raise GuardFailure(f"draw {d}: the loaded table is not the written one: {bad}")
                carry = _carry(theta, md)
                m, cap = theta[("ninio_m",)], theta[("ninio_max",)]
                for k, seq in enumerate(seqs):
                    fc = first if k == 0 else RNA.fold_compound(seq, md)
                    s, e = fc.mfe()
                    want = _energy(ex.counts(seq, s), theta) + _ninio_correction(s, m, cap)
                    if want != int(round(e * 100)):
                        raise GuardFailure(f"draw {d}, sequence {k}: ViennaRNA's MFE energy "
                                           f"{int(round(e * 100))} dcal, counts . theta {want}")
                    tally[k].update(_pairs(s))
                    if on_fold is not None:
                        on_fold(d, k, s)
                    differs += s != star0[k]
                    n_folds += 1
            draws.append({"draw": d, "max_rounding_kcal": resid,
                          "n_entries_changed_dcal": n_changed, "par_sha256": sha,
                          "params_match": bad, "params_match_pass": not bad,
                          "carry_rows": carry,
                          "n_sequences_mfe_differs_from_turner2004": int(differs)})
    finally:
        path.unlink(missing_ok=True)

    fc = RNA.fold_compound("GGGGAAACCCC", reference_md())
    assert_turner2004_params(fc)
    _s, e = fc.mfe()
    fc.exp_params_rescale(e)
    fc.pf()
    assert_turner2004_boltzmann(fc)

    nd = len(shifts)
    freq = [{p: c / nd for p, c in sorted(t.items())} for t in tally]
    th0_keys = set(theta0) & set(t2004)
    return freq, {
        "n_draws": nd, "n_sequences": len(seqs), "n_columns": len(cols),
        "theta0_equals_turner2004": set(theta0) <= set(t2004)
        and all(theta0[k] == t2004[k] for k in th0_keys),
        "max_rounding_kcal": max(x["max_rounding_kcal"] for x in draws),
        "all_params_match_pass": all(x["params_match_pass"] for x in draws),
        "mfe_energy_identity": {"n_folds": n_folds, "n_mismatch": 0},
        "turner2004_restored": True,
        "draws": draws,
        "seconds": round(time.time() - t_start, 2),
    }
