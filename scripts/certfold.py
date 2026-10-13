#!/usr/bin/env python3
"""CERTIFIED FOLDING. Per-pair, for-all guarantees over the data-restricted null space.

THE QUESTION. For each sequence, with s* the Turner 2004 MFE structure, which base pairs of s*
are present in EVERY optimal structure for EVERY theta = theta0 + B t with ||t||_2 <= r (plus any
gauge shift), which pairs outside s* are absent from every one of them, and how large can r grow
before a pair stops being certified?

THE MODEL is pinned exactly as everywhere else in this repository: Turner 2004 in ViennaRNA 2.7.2
at 37 C, dangles 0, special hairpins off, through thermo.vienna_ref.reference_md(). Energies are
integer dcal/mol. A structure's energy is E(s) = c(s).theta + o(s) with c(s) a sum of per-loop
count vectors; the certificate needs those PER LOOP, which is what LoopCounts adds to the
validated Extractor count model.

PHASES, in order; each refuses unless its predecessor's record is committed.
  build-basis   B_TR0 from TR0 references only, its must-holds, the radii, r_cap, and what the
                ball actually permits per loop
  gate          G1 to G4 and G6; folds no test-set sequence at all
  preregister   PRED, committed before any test-set fold
  gate-attacks  G5, after PRED; violation counts only
  run           TS0 and the held-out one-per-family set
  analyse       R1 to R3, the descriptive fractions, the ablations

Run:
  .venv/bin/python scripts/certfold.py --phase build-basis      # then commit

THE SET, stated exactly (a decision this script makes, recorded in the artifact).
The authors' block fixes Theta_r = {theta0 + B t : ||t||_2 <= r} with per-entry weights
w_L(r) = r * sum_e |c_{L,e}| * ||a_e||, a_e the row of B for entry e. The callback can only be
cheap if it returns 0 for every decomposition code other than PAIR_IL, and that is exact only if
B has no weight outside the four tables PAIR_IL carries (int11, int21, int22, bulge). Measured,
that weight is at the basis's numerical noise floor, not identically zero, and there is no
affordable exact certificate that it vanishes: singleton elimination determines an eliminated
column's null entry from the others rather than forcing it to zero, and a mod-p rank gives a
lower bound on the rational rank, not the equality this would need.

So the set is DEFINED to be the ball in B_four: B_TR0 restricted to the coordinates that are
BOTH in the four tables and carry a row norm above the basis's own arithmetic noise, then
re-orthonormalised. Then returning 0 on every other entry is exact rather than approximate, and
the only thing owed to the reader is how far B_four sits from B_TR0, which this phase measures and
records (about 1e-13 in Frobenius norm, against the 0.005 kcal/mol rounding the parameter file
already imposes). The second restriction is not a threshold choice: the rows that carry a
direction have norms from 0.36 to 0.90 and every other row sits at 4e-14 or below. Keeping those
rows would be sound but would cost a full dcal on every loop that touches one, since ceil of any
positive weight is at least 1 dcal, and it would make rho(p) a ratio with noise in the
denominator -- measured, it sent rho to 1e15. Every radius, every published component and every
label is stated with respect to B_four. The alternative -- keeping all of B and charging every
loop -- is sound but costs one dcal per differing loop after the ceiling, because ceil of any
positive weight is at least 1 dcal, and it would require implementing all nine decomposition
codes for weights at the 1e-15 level. That trade is the authors' to revisit; it is recorded here
rather than taken silently.

GATES. build-basis refuses unless the tree is clean outside runs/, and records its provenance and
code closure; every later phase refuses unless the committed artifact carries the previous
phase's record, PRED still equals this file's PRED, and every recorded digest still matches.
Guards raise GuardFailure and are never repaired.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_nullspace import (  # noqa: E402
    BCODE,
    PTYPE,
    Extractor,
    pair_table,
    ptype,
)
from scripts.turner_nullspace_tr0 import distinct_rows, tr0_ts0_rows  # noqa: E402
from scripts.turner_par import template  # noqa: E402
from scripts.turner_typeb import GuardFailure, compute_basis, load_basis  # noqa: E402

RUN_ID = "2026-10-13_certified_folding"
OUT = R / "runs" / RUN_ID
SHARDS = OUT / "shards"          # gitignored (runs/**/shards/); sha256 recorded
VIENNARNA = "2.7.2"
SEED = 20261013

# the four tables VRNA_DECOMP_PAIR_IL carries and the null space lives in
FOUR = ("int11", "int21", "int22", "bulge")
# the bar for "B has no weight here": the TR0+TS0 basis is orthonormal to 4e-17 and annihilated
# by its reference matrix to 1e-13, so 1e-12 is far above the noise and a real direction would
# exceed it by orders of magnitude
WEIGHT_BAR = 1e-12
# a row norm at or below this is the basis's own noise, not a direction: the four-table
# entries TR0's references pin carry about 1e-15, against an annihilation residual of 1e-13
ROW_NOISE = 1e-9
# pair type by base CODE, so the callback need not build strings: PT[a][b] is what
# turner_nullspace.ptype returns for those two bases, 0 for a non-canonical pair
PT = [[0] * 5 for _ in range(5)]
for _p, _k in PTYPE.items():
    PT[BCODE[_p[0]]][BCODE[_p[1]]] = _k
R_CAP_MULTIPLE = 4               # r_cap = 4 x the largest published radius on B_four
G1_SEQUENCES = 20                # the authors' G1 size: 20 sequences, n 50 to 500
G1_SAMPLES = 1000                # about 1,000 sampled structures each
G2_SEQUENCES = 200               # the authors' G2 size: 200 sequences at n <= 20
G2_MAX_N = 20
G2_RADII = (1.0, 5.0, 20.0)      # the authors' G2 radii
G2_POWER_SEQUENCES = 50          # the halved-weights power check, on the same draw
G3_SEQUENCES = 10                # G3's realistic-length half, n 120 to 150
G3_PAIRS = 60                    # pairs sampled per sequence per radius, each side
G6_SEQUENCES = 20                # monotonicity in r
G6_GAUGE_SEQUENCES = 10          # the gauge half, which writes parameter files
G6_GAUGE_NORM = 100.0            # kcal/mol, the authors' size for the gauge shift
G4_SEQUENCES = 100               # the authors' G4 size, on TR0 (amendment 1)
G4_MAX_N = 150
G4_DELTA_CAP = 600               # dcal: a band wider than this is skipped, not cut
PILOT_SEQUENCES = 300            # the authors' pilot size, TR0 only
PILOT_WORKERS = 8                # of 16 cores, at nice 10


def _sha(p: pathlib.Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def read_out():
    p = OUT / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"run_id": RUN_ID}


def _git(*a):
    return subprocess.run(["git", *a], cwd=R, capture_output=True, text=True)


# ----------------------------------------------------------------------------------
# per-loop counts: the certificate's feature map, from the validated count model
# ----------------------------------------------------------------------------------
class LoopCounts(Extractor):
    """Extractor.counts, but one count vector per LOOP INSTANCE instead of one per structure.

    The parent's loop walk already branches on exactly the rule classes a Zuker-style DP uses,
    and its _hairpin / _interior / _multi take the accumulator as an argument, so each loop gets
    a fresh one. The exterior walk is inline in the parent and is mirrored here; `loops` is
    checked against `counts` exactly, which is what catches any divergence.

    A record is {"kind", "i", "j", "k", "l", "counts"}. (i, j) is the closing pair, 1-based;
    (k, l) is the enclosed pair for an interior loop and (0, 0) otherwise, matching the
    (i, j, k, l) a VRNA_DECOMP_PAIR_IL callback receives. "kind" is one of ext_stem, hairpin,
    interior, multi; interior records also carry "sub" (stack, bulge, int11, int21, int22, int1n,
    int23, generic), which is what the callback has to recover from (i, j, k, l) alone, since
    every one of them arrives under the same decomposition code.
    """

    def loops(self, seq: str, db: str) -> list[dict]:
        n = len(seq)
        pt = pair_table(db)
        S = [0] + [BCODE[c] for c in seq] + [0]
        out: list[dict] = []

        def tau(t):
            return 1 if t > 2 else 0

        # exterior loop: one record per top-level stem (mirrors Extractor.counts)
        i = 1
        while i <= n:
            if pt[i] > i:
                j = pt[i]
                c: dict[tuple, int] = {}

                def add(k, v=1, c=c):
                    c[k] = c.get(k, 0) + v

                t = ptype(seq[i - 1], seq[j - 1])
                if self.d == 2:
                    s5 = S[i - 1] if i > 1 else -1
                    s3 = S[j + 1] if j < n else -1
                    if s5 > 0 and s3 > 0:
                        add(("mismatchExt", t, s5, s3))
                    elif s5 > 0:
                        add(("dangle5", t, s5))
                    elif s3 > 0:
                        add(("dangle3", t, s3))
                if tau(t):
                    add(("TerminalAU",))
                out.append({"kind": "ext_stem", "i": i, "j": j, "k": 0, "l": 0, "counts": c})
                i = j + 1
            else:
                i += 1

        for i in range(1, n + 1):
            j = pt[i]
            if j <= i:
                continue
            inner = []
            p = i + 1
            while p < j:
                if pt[p] > p:
                    inner.append((p, pt[p]))
                    p = pt[p] + 1
                else:
                    p += 1
            t = ptype(seq[i - 1], seq[j - 1])
            c = {}

            def add(k, v=1, c=c):
                c[k] = c.get(k, 0) + v

            if not inner:
                self._hairpin(seq, S, i, j, t, add, tau)
                out.append({"kind": "hairpin", "i": i, "j": j, "k": 0, "l": 0, "counts": c})
            elif len(inner) == 1:
                p_, q_ = inner[0]
                self._interior(seq, S, i, j, inner[0], t, add, tau)
                out.append({"kind": "interior", "sub": interior_sub(i, j, p_, q_),
                            "i": i, "j": j, "k": p_, "l": q_, "counts": c})
            else:
                self._multi(seq, S, i, j, inner, add, tau)
                out.append({"kind": "multi", "i": i, "j": j, "k": 0, "l": 0,
                            "n_branches": len(inner), "counts": c})
        return out

    def loops_checked(self, seq: str, db: str) -> list[dict]:
        """`loops`, with the exact identity against the parent's own count vector."""
        per = self.loops(seq, db)
        total: dict[tuple, int] = {}
        for rec in per:
            for k, v in rec["counts"].items():
                total[k] = total.get(k, 0) + v
        want = self.counts(seq, db)
        if {k: v for k, v in total.items() if v} != {k: v for k, v in want.items() if v}:
            only_loops = {k: v for k, v in total.items() if want.get(k, 0) != v}
            only_counts = {k: v for k, v in want.items() if total.get(k, 0) != v}
            raise GuardFailure(
                f"per-loop decomposition: sum over loops differs from Extractor.counts "
                f"(loops-only {list(only_loops.items())[:4]}, counts-only "
                f"{list(only_counts.items())[:4]})")
        return per


def il_keys(S: list[int], i: int, j: int, k: int, l: int  # noqa: E741  (i, j, k, l) is
            ) -> dict[tuple, int]:  # ViennaRNA's own callback signature; renaming hides it
    """The parameter-file entries vrna_E_internal reads for the interior loop closed by (i, j)
    and enclosing (k, l), from the positions and the encoded sequence alone.

    This is the CALLBACK's side of the feature map: a VRNA_DECOMP_PAIR_IL callback is handed
    exactly (i, j, k, l) and must recover the entries itself, because stack, bulge, int11,
    int21, int22, 1xn, 2x3 and generic interior loops all arrive under that one decomposition
    code. `LoopCounts` walks a structure instead and is the checker's side.

    The two share a convention -- which entry a case reads, and in which orientation -- so
    agreement between them would not catch a convention applied wrongly in both. That is why
    `check_il_keys_against_viennarna` dots these keys with theta0 and compares against
    RNA.E_IntLoop, ViennaRNA's own interior-loop energy: an oracle that shares nothing with
    either implementation.

    S is 1-based with sentinels, as Extractor uses it: S = [0] + codes + [0].
    """
    n1, n2 = k - i - 1, j - l - 1
    t = PT[S[i]][S[j]]
    t2 = PT[S[l]][S[k]]                       # the enclosed pair, reversed
    si1, sj1, sp1, sq1 = S[i + 1], S[j - 1], S[k - 1], S[l + 1]
    nl, ns = max(n1, n2), min(n1, n2)
    c: dict[tuple, int] = {}

    def add(key, v=1):
        c[key] = c.get(key, 0) + v

    def tau(pt):
        return pt > 2                            # AU, UA, GU, UG close with a terminal penalty

    if nl == 0:
        add(("stack", t, t2))
        return c
    if ns == 0:
        add(("bulge", nl) if nl <= 30 else ("bulge", 30))
        if nl == 1:
            add(("stack", t, t2))
        else:
            if tau(t):
                add(("TerminalAU",))
            if tau(t2):
                add(("TerminalAU",))
        return c
    if ns == 1:
        if nl == 1:
            add(("int11", t, t2, si1, sj1))
            return c
        if nl == 2:
            if n1 == 1:
                add(("int21", t, t2, si1, sq1, sj1))
            else:
                add(("int21", t2, t, sq1, si1, sp1))
            return c
        add(("internal_loop", min(nl + 1, 30)))
        _ninio_keys(nl - ns, add)
        add(("mismatch1nI", t, si1, sj1))
        add(("mismatch1nI", t2, sq1, sp1))
        return c
    if ns == 2:
        if nl == 2:
            add(("int22", t, t2, si1, sp1, sq1, sj1))
            return c
        if nl == 3:
            add(("internal_loop", 5))
            add(("ninio_m",))
            add(("mismatch23I", t, si1, sj1))
            add(("mismatch23I", t2, sq1, sp1))
            return c
    add(("internal_loop", min(nl + ns, 30)))
    _ninio_keys(nl - ns, add)
    add(("mismatchI", t, si1, sj1))
    add(("mismatchI", t2, sq1, sp1))
    return c


def _ninio_keys(asym: int, add):
    """MIN2(MAX_NINIO, asym * m), linearised at Turner 2004's (m, cap) = (60, 300)."""
    if asym == 0:
        return
    if asym * 60 <= 300:
        add(("ninio_m",), asym)
    else:
        add(("ninio_max",))


def interior_sub(i: int, j: int, p: int, q: int) -> str:
    """Which vrna_E_internal case an interior loop (i, j, p, q) takes, from positions alone.

    This is the classification the callback must perform on (i, j, k, l), and it is written here
    from the case order of Extractor._interior so that the two agree by construction on the
    naming while remaining independent in the energy they compute.
    """
    n1, n2 = p - i - 1, j - q - 1
    nl, ns = max(n1, n2), min(n1, n2)
    if nl == 0:
        return "stack"
    if ns == 0:
        return "bulge"
    if ns == 1:
        return {1: "int11", 2: "int21"}.get(nl, "int1n")
    if ns == 2 and nl == 2:
        return "int22"
    if ns == 2 and nl == 3:
        return "int23"
    return "generic"


# ----------------------------------------------------------------------------------
# the certificate: per-loop weights, the callback, and E'
# ----------------------------------------------------------------------------------
def reached_shapes(cols: list[tuple], a: np.ndarray) -> frozenset:
    """The (u1, u2) shapes an interior loop must have for the ball to move it at all.

    Each of the four tables fixes its loop's shape: int11 is 1x1, int21 is 1x2 and 2x1, int22 is
    2x2, and a bulge of size n is nx0 and 0xn. Every other shape -- a stack, a generic interior
    loop, a 1xn, a 2x3, a bulge of an unreached size -- reads only entries outside those tables,
    where B_four is exactly zero, so its weight is exactly zero too.

    This is derived from the basis rather than assumed, and it is what lets the callback answer
    the common case with one integer test instead of rebuilding a key dictionary: stacks alone
    are 81% of the interior loops in the generating set. The equivalence is not taken on trust --
    G1's constructed cases walk every vrna_E_internal case with both signs, and G2 enumerates
    whole structure spaces, so a shape wrongly left out of this set would show up as an
    under-charged loop and a false certificate.
    """
    shapes: set[tuple[int, int]] = set()
    for k, nrm in zip(cols, a, strict=True):
        if nrm <= ROW_NOISE:
            continue
        if k[0] == "int11":
            shapes.add((1, 1))
        elif k[0] == "int21":
            shapes.update({(1, 2), (2, 1)})
        elif k[0] == "int22":
            shapes.add((2, 2))
        elif k[0] == "bulge":
            shapes.update({(int(k[1]), 0), (0, int(k[1]))})
    return frozenset(shapes)


class Weights:
    """Per-loop weights w_L(r) = r * sum_e |c_{L,e}| ||a_e||, in ViennaRNA's integer dcal/mol.

    Unit discipline, in one place because a silent factor of 100 would make every certificate
    either vacuous or false: r and ||a_e|| are kcal/mol in the parameter-file gauge, ViennaRNA's
    energies and a soft-constraint callback's return value are integer dcal/mol, so the weight is
    formed in kcal, multiplied by 100 and rounded UP. Rounding up on both sides of the
    comparison is what keeps E'(s) - E'(s*) a lower bound on the true gap: the certificate then
    subtracts at least the worst-case penalty rather than at most.
    """

    def __init__(self, a_full: np.ndarray, cols: list[tuple], r_kcal: float):
        self.a = a_full
        self.idx = {k: n for n, k in enumerate(cols)}
        self.r = float(r_kcal)
        self.shapes = reached_shapes(cols, a_full)

    def kcal(self, keys: dict[tuple, int]) -> float:
        # float() deliberately: self.a is a numpy array, so without it every weight, every
        # D(s), every rho and every flag derived from them is a numpy scalar, and numpy scalars
        # serialise through json's default=str as QUOTED STRINGS. The artifact is read by jq and
        # by the paper's pins, so a quoted number there is a broken pin later.
        return float(self.r * sum(abs(v) * self.a[self.idx[k]] for k, v in keys.items()
                                  if k in self.idx))

    def dcal(self, keys: dict[tuple, int]) -> int:
        w = self.kcal(keys)
        return 0 if w <= 0 else int(math.ceil(w * 100.0 - 1e-12))

    def loop_dcal(self, S: list[int], i: int, j: int, k: int, l: int) -> int:  # noqa: E741
        return self.dcal(il_keys(S, i, j, k, l))


def star_interior_loops(lc: LoopCounts, seq: str, db: str) -> set:
    """The (i, j, k, l) of s*'s interior loops: constant-time membership for the callback.

    A loop of some other structure counts as a loop of s* exactly when s* contains it, closing
    pair, enclosed pair and an unpaired interior alike, which is what this set encodes: it holds
    only loops the walk over s* actually found.
    """
    return {(r["i"], r["j"], r["k"], r["l"]) for r in lc.loops(seq, db)
            if r["kind"] == "interior"}


def make_callback(S: list[int], star: set, W: Weights, counter: list | None = None):
    """The soft-constraint callback: +ceil(w_L) on a loop of s*, -ceil(w_L) on any other loop.

    Returns 0 for every decomposition code other than VRNA_DECOMP_PAIR_IL. That is exact rather
    than approximate because the set certified over is the ball in B_four, whose rows outside
    int11, int21, int22 and bulge are exactly zero -- build-basis asserts it. If the basis ever
    carried weight on a hairpin, multiloop or exterior-stem entry, this callback would silently
    under-charge those loops, so the assertion is a precondition and not a nicety.

    `counter`, when given, is a one-element list the caller can read afterwards: the number of
    invocations, which is how the gate reports cost without the callback holding mutable state
    that a parallel region could race on.
    """
    import RNA

    il = RNA.DECOMP_PAIR_IL
    shapes = W.shapes
    loop_dcal = W.loop_dcal

    def cb(i, j, k, l, d, _data=None):  # noqa: E741  ViennaRNA's own argument names
        if counter is not None:
            counter[0] += 1
        if d != il:
            return 0
        # the shape test first, because it decides the overwhelming majority of calls with two
        # subtractions and a frozenset lookup. One mfe() at n = 134 invokes this callback
        # 834,253 times; rebuilding a key dictionary on each of them cost 0.68 s a fold, which
        # is what made G3 take 14 minutes and would have made one n = 467 sequence of the pilot
        # take hours. A loop whose shape no reached table carries has weight exactly zero, so
        # the early return changes no value -- see reached_shapes.
        if (k - i - 1, j - l - 1) not in shapes:
            return 0
        w = loop_dcal(S, i, j, k, l)
        if w == 0:
            return 0
        return w if (i, j, k, l) in star else -w

    return cb


def eprime_offset(lc: LoopCounts, seq: str, s: str, star: set, W: Weights) -> int:
    """E'(s) - E0(s) in dcal, summed over the loops of s from the checker's side.

    Only interior loops can contribute, because B_four reaches no other table, and this is the
    quantity the gate compares against what ViennaRNA's own decomposition plus the callback
    produce. E'(s*) is taken from here rather than from eval_structure on the constrained
    compound, which honours only the hairpin and interior-loop codes and silently drops the
    multiloop and exterior ones.
    """
    S = [0] + [BCODE[c] for c in seq] + [0]
    total = 0
    for rec in lc.loops(seq, s):
        if rec["kind"] != "interior":
            continue
        key = (rec["i"], rec["j"], rec["k"], rec["l"])
        w = W.loop_dcal(S, *key)
        if w:
            total += w if key in star else -w
    return total


# ----------------------------------------------------------------------------------
# the certificate: X', Z', the labels and rho
# ----------------------------------------------------------------------------------
def admissible_pairs(seq: str, min_hp: int = 3) -> list[tuple[int, int]]:
    """Every (i, j) some structure of `seq` could contain: canonical, min_hp unpaired between.

    This is the denominator of the certified-absent fraction. It is a property of the sequence
    alone, not of any fold, so the fraction does not move when the structure does.
    """
    S = [0] + [BCODE[c] for c in seq] + [0]
    n = len(seq)
    return [(i, j) for i in range(1, n + 1) for j in range(i + min_hp + 1, n + 1)
            if PT[S[i]][S[j]]]


def loop_coeff_map(lc: LoopCounts, seq: str, db: str, W1: Weights) -> dict:
    """{(i, j, k, l): D_L} over the interior loops of `db`, in kcal/mol per unit radius.

    D_L = sum_e |c_{L,e}| ||a_e|| is the weight at r = 1, so w_L(r) = r * D_L exactly. The keys
    come from il_keys, the callback's own reconstruction, so this and the callback cannot drift
    apart; G1 is what established they agree with ViennaRNA's own interior-loop energy.
    """
    S = [0] + [BCODE[c] for c in seq] + [0]
    return {(d["i"], d["j"], d["k"], d["l"]):
            W1.kcal(il_keys(S, d["i"], d["j"], d["k"], d["l"]))
            for d in lc.loops(seq, db) if d["kind"] == "interior"}


def d_coeff(star_map: dict, s_map: dict) -> float:
    """D(s): the sum of D_L over the loops in the symmetric difference of s and s*.

    Only the symmetric difference is charged, and that is not an approximation. For theta in the
    ball the gap is E(s) - E(s*) = [E0(s) - E0(s*)] + sum over loops of s of c_L.Bt - the same
    over s*, and a loop the two structures share cancels exactly, term for term, because it is
    the same loop with the same counts. So the worst case over the ball charges the differing
    loops only, which is also why the callback signs by membership in s* rather than charging
    every loop: a shared loop gets +w in E'(s) and +w in E'(s*) and drops out of the difference.
    """
    tot = 0.0
    for k, v in s_map.items():
        if k not in star_map:
            tot += v
    for k, v in star_map.items():
        if k not in s_map:
            tot += v
    return tot


class Folder:
    """One constrained compound, reused for every fold of one sequence.

    Measured on this box: hc_init() clears the hard constraints while leaving the soft callback
    in force, so a per-pair sweep costs one compound and one callback registration rather than
    one of each per pair. The callback object is held on the instance because ViennaRNA keeps
    only a borrowed reference and would call into freed memory otherwise.

    Every fold is cross-checked: the DP's own E' for the structure it returns must equal the
    checker's E0(s) + signed ceiled weights, exactly, in integer dcal. That is the G1 identity
    re-run on every fold in anger, and it is what would catch the DP applying the callback on a
    decomposition the checker does not model -- a failure mode no amount of sampled structures
    rules out, because it would show up only on the structures the DP actually visits.
    """

    def __init__(self, lc: LoopCounts, seq: str, star_db: str, W: Weights,
                 pinned: bool = True):
        import RNA

        from thermo.vienna_ref import assert_turner2004_params, reference_md

        self.RNA = RNA
        self.lc, self.seq, self.star_db, self.W = lc, seq, star_db, W
        self.S = [0] + [BCODE[c] for c in seq] + [0]
        self.star = star_interior_loops(lc, seq, star_db)
        self.plain = RNA.fold_compound(seq, reference_md())
        # pinned=False is for the gauge half of G6 and nowhere else: inside a loaded() block the
        # global table is a written parameter file, so the Turner 2004 fingerprint is the wrong
        # pin and params_match against the written theta is the right one. The caller does that.
        if pinned:
            assert_turner2004_params(self.plain)
        self.fc = RNA.fold_compound(seq, reference_md())
        self.calls = [0]
        self._cb = make_callback(self.S, self.star, W, self.calls)
        rc = self.fc.sc_add_f(self._cb)
        if not rc:
            raise GuardFailure(f"sc_add_f refused the callback on {seq[:20]}... (rc = {rc})")
        self.e0_star = int(round(self.plain.eval_structure(star_db) * 100))
        self.folds = 0
        self.set_radius(W.r)

    def set_radius(self, r: float) -> None:
        """Change the radius in place. The callback reads W.r when the DP calls it.

        ViennaRNA stores an f-callback, not a table of its values, so the next fold sees the new
        weights; nothing is precomputed at registration time. G2 checks that against the
        alternative of a fresh compound per radius, because "nothing is cached" is a claim about
        someone else's code.
        """
        self.W.r = float(r)
        self.eprime_star = self.e0_star + eprime_offset(
            self.lc, self.seq, self.star_db, self.star, self.W)

    def eprime(self, s: str) -> tuple[int, int]:
        """(E'(s), E0(s)) in dcal from the checker: E0 plus the signed ceiled weights."""
        e0 = int(round(self.plain.eval_structure(s) * 100))
        return e0 + eprime_offset(self.lc, self.seq, s, self.star, self.W), e0

    def _mfe(self, what: str) -> tuple[str, int, int]:
        self.folds += 1
        s, e = self.fc.mfe()
        got = int(round(e * 100))
        want, e0 = self.eprime(s)
        if got != want:
            raise GuardFailure(
                f"the DP and the checker disagree on E' ({what}, r = {self.W.r}): "
                f"DP {got} != checker {want} (E0 {e0}) on {s[:60]}")
        return s, got, e0

    def free(self) -> tuple[str, int, int]:
        self.fc.hc_init()
        return self._mfe("free")

    def without(self, i: int, j: int) -> tuple[str, int, int]:
        """X'(i, j): the minimum of E' over structures that do NOT contain the pair.

        NO_REMOVE is load-bearing and was found by G2, not by reading the header. Without it
        vrna_hc_add_bp also removes the pairs that CROSS (i, j) -- sound when a pair is being
        enforced, wrong when it is being forbidden -- so the minimisation runs over too small a
        space, X' comes back too HIGH, and pairs certify that are not robust. That is a false
        certificate, the one failure this whole gate exists to catch, and on the first sequence
        where it mattered it cost 0.75 kcal/mol: the DP returned -1013 dcal where the enumerated
        minimum over (1,18)-free structures was -1088. With the flag the two agree exactly.
        """
        self.fc.hc_init()
        self.fc.hc_add_bp(i, j, self.RNA.CONSTRAINT_CONTEXT_NONE
                          | self.RNA.CONSTRAINT_CONTEXT_NO_REMOVE)
        s, e, e0 = self._mfe(f"without {(i, j)}")
        if pair_table(s)[i] == j:
            raise GuardFailure(f"forbidding {(i, j)} did not: {s[:60]}")
        return s, e, e0

    def with_pair(self, i: int, j: int) -> tuple[str, int, int] | None:
        """Z'(i, j) by an enforced fold, or None if no structure can contain the pair."""
        self.fc.hc_init()
        self.fc.hc_add_bp(i, j, self.RNA.CONSTRAINT_CONTEXT_ALL_LOOPS
                          | self.RNA.CONSTRAINT_CONTEXT_ENFORCE)
        self.folds += 1
        s, e = self.fc.mfe()
        if pair_table(s)[i] != j or e > 1e4:
            return None
        got = int(round(e * 100))
        want, e0 = self.eprime(s)
        if got != want:
            raise GuardFailure(f"the DP and the checker disagree on E' (with {(i, j)}): "
                               f"DP {got} != checker {want}")
        return s, got, e0

    def zuker(self) -> list[str]:
        """The Zuker suboptimal set under E': one optimal structure per possible pair.

        Its licence is G3. Zuker's algorithm computes, for each pair, the optimal structure
        containing it, so the minimum of E' over the returned structures that contain q is
        Z'(q), and a pair in none of them has no legal structure containing it at all. Both
        halves of that are tested against enumeration and against enforced folds before any
        label depends on them.
        """
        self.fc.hc_init()
        self.folds += 1
        return [z.structure for z in self.fc.subopt_zuker()]


def zprime_map(F: Folder, structures: list[str]) -> dict:
    """{(i, j): min E'} over the given structures, with E' from the checker, not from subopt.

    subopt_zuker reports its own energies; taking them on trust would make the labels depend on
    how that routine accounts for the soft constraint. Recomputing E' from E0 plus the signed
    weights costs 60 microseconds a structure and removes the question.
    """
    best: dict[tuple[int, int], int] = {}
    for s in structures:
        e, _e0 = F.eprime(s)
        pt = pair_table(s)
        for i in range(1, len(s) + 1):
            if pt[i] > i:
                k = (i, pt[i])
                if e < best.get(k, 1 << 30):
                    best[k] = e
    return best


def certify(F: Folder, star_pairs: list, adm: list, zuker: bool = True) -> dict:
    """Labels at F's current radius: certified present, certified absent, or uncertain.

    A pair p of s* is certified present when EVERY structure without it has E' above E'(s*),
    which is one forbidden-pair fold; a pair q outside s* is certified absent when every
    structure containing it does, which is one Zuker fold for all of them at once. Both
    comparisons are strict and in integer dcal: a tie is uncertain, never certified, because at
    a tie some theta in the ball makes the competitor an equal-energy ground state.
    """
    present, absent = {}, {}
    for (i, j) in star_pairs:
        _s, x, _e0 = F.without(i, j)
        present[(i, j)] = x - F.eprime_star
    zmap = zprime_map(F, F.zuker()) if zuker else {}
    n_unreachable = 0
    for q in adm:
        if q in present:
            continue
        z = zmap.get(q)
        if z is None:
            n_unreachable += 1
            absent[q] = None                      # no structure contains it: trivially absent
        else:
            absent[q] = z - F.eprime_star
    n_p = sum(1 for v in present.values() if v > 0)
    n_a = sum(1 for v in absent.values() if v is None or v > 0)
    return {
        "radius_kcal": F.W.r,
        "n_pairs_star": len(present),
        "n_certified_present": n_p,
        "n_pairs_admissible_outside_star": len(absent),
        "n_certified_absent": n_a,
        "n_absent_because_unreachable": n_unreachable,
        "whole_structure_certified": n_p == len(present) and n_a == len(absent),
        "margins_present_dcal": present,
        "margins_absent_dcal": absent,
        "folds": F.folds,
    }


def rho_of_pair(F: Folder, star_map: dict, W1: Weights, i: int, j: int, r_cap: float,
                max_iter: int = 20) -> dict:
    """rho(i, j): the largest radius at which the pair is still certified, by Dinkelbach.

    rho = min over structures s without the pair of N(s)/D(s), with N(s) = E0(s) - E0(s*) >= 0
    and D(s) the symmetric-difference coefficient, both exact in kcal from the counts. Dinkelbach
    iterates r <- N(s_r)/D(s_r) for s_r the minimiser of N - r D, which is one constrained fold.

    The inner minimisation uses the CEILED integer weights, because that is what a callback can
    return, so it searches with a slightly over-charged D. The bias is toward structures with
    larger D and hence smaller ratios, which is the conservative direction, and G2 measures what
    it costs against the exact minimum over an enumerated structure space.

    Edge cases are the registered ones. A nominal tie (N = 0 at r = 0) gives rho = 0 and is
    reported separately rather than averaged in. A minimiser with D = 0 is insensitive to the
    ball along every direction, so the iteration restarts at r_cap; if the margin there is still
    positive the pair is recorded as rho >= r_cap, capped, and never given a number.
    """
    if W1 is F.W or W1.r != 1.0:
        # set_radius mutates the Weights object in place, so the unit-radius coefficients must
        # be a separate instance: sharing one would zero D(s) the moment the iteration visits
        # r = 0 and hand back rho = r_cap for every pair.
        raise GuardFailure("rho needs its own r = 1 Weights, distinct from the folder's")
    F.set_radius(0.0)
    s0, x0, _e0_0 = F.without(i, j)
    nominal_gap = (x0 - F.e0_star) / 100.0
    if x0 <= F.e0_star:
        return {"rho": 0.0, "tied_at_zero": True, "capped": False, "converged": True,
                "iterations": 0, "nominal_gap_kcal": nominal_gap}
    d0 = d_coeff(star_map, loop_coeff_map(F.lc, F.seq, s0, W1))
    r = r_cap if d0 <= 0 else min(nominal_gap / d0, r_cap)
    jumped = d0 <= 0
    for it in range(1, max_iter + 1):
        F.set_radius(r)
        s, _x, e0 = F.without(i, j)
        n_s = (e0 - F.e0_star) / 100.0
        d_s = d_coeff(star_map, loop_coeff_map(F.lc, F.seq, s, W1))
        if d_s <= 0:
            if jumped or r >= r_cap:
                return {"rho": r_cap, "tied_at_zero": False, "capped": True, "converged": True,
                        "iterations": it, "nominal_gap_kcal": nominal_gap}
            r, jumped = r_cap, True
            continue
        margin = n_s - r * d_s
        r_new = n_s / d_s
        if abs(margin) <= 1e-9 * max(1.0, abs(n_s)) or abs(r_new - r) <= 1e-9 * max(1.0, r):
            return {"rho": min(r, r_new), "tied_at_zero": False,
                    "capped": min(r, r_new) >= r_cap, "converged": True, "iterations": it,
                    "nominal_gap_kcal": nominal_gap}
        r = min(r_new, r_cap)
    return {"rho": r, "tied_at_zero": False, "capped": r >= r_cap, "converged": False,
            "iterations": max_iter, "nominal_gap_kcal": nominal_gap}


# ----------------------------------------------------------------------------------
# the exact test G2 checks the certificate against
# ----------------------------------------------------------------------------------
def enumerate_structures(seq: str, min_hp: int = 3) -> list[str]:
    """Every secondary structure of `seq`, canonical pairs and no lone-pair rule.

    Written here rather than taken from ViennaRNA's subopt so that G2's reference structure
    space shares no code with the thing it tests. The recursion is the textbook one: position i
    is unpaired, or pairs with some k, and the two sides are independent.
    """
    n = len(seq)
    S = [0] + [BCODE[c] for c in seq] + [0]
    memo: dict[tuple[int, int], tuple] = {}

    def f(i: int, j: int) -> tuple:
        if j - i + 1 < min_hp + 2:
            return ((),)
        key = (i, j)
        if key in memo:
            return memo[key]
        out = list(f(i + 1, j))
        for k in range(i + min_hp + 1, j + 1):
            if PT[S[i]][S[k]]:
                for a in f(i + 1, k - 1):
                    for b in f(k + 1, j):
                        out.append(((i, k),) + a + b)
        memo[key] = tuple(out)
        return memo[key]

    res = []
    for pairs in f(1, n):
        db = ["."] * n
        for a, b in pairs:
            db[a - 1], db[b - 1] = "(", ")"
        res.append("".join(db))
    return res


def exact_worst_case(B4: np.ndarray, pos: dict, dc: dict, n_kcal: float, r: float) -> float:
    """N(s) - r ||B_four^T (c(s) - c(s*))||_2: the exact worst case over the ball.

    Exact, not a bound: the minimum of t.(B^T dc) over ||t|| <= r is -r times that norm, attained
    at t along -B^T dc. The certificate's per-entry weights replace this norm by the sum of
    |dc_e| ||a_e||, which the triangle inequality makes no smaller, so certified implies exactly
    robust as mathematics. G2 tests the implementation of that claim, not the claim.
    """
    v = np.zeros(B4.shape[1])
    for k, c in dc.items():
        p = pos.get(k)
        if p is not None and c:
            v += c * B4[p]
    return n_kcal - r * float(np.linalg.norm(v))

def load_tr0_basis() -> dict:
    """B_TR0 and B_four from the build-basis shard, with the digest the artifact recorded.

    The npz is gitignored, as every basis in this repository is; the contract is that the
    artifact carries its sha256 and the matrix is recomputable by rerunning the phase.
    """
    import ast

    out = read_out()
    rec = out["build_basis"]["files_sha256"]
    for f, h in rec.items():
        got = _sha(SHARDS / f)
        if got != h:
            raise GuardFailure(f"{f} changed since build-basis ({got[:12]} != {h[:12]})")
    z = np.load(SHARDS / "basis_tr0.npz", allow_pickle=False)
    cols = [ast.literal_eval(str(s)) for s in z["cols"]]
    return {"cols": cols, "used": z["used"], "NT": z["NT"], "B4": z["B4"],
            "a_tr0": z["a_tr0"], "a_four": z["a_four"],
            "radii": out["build_basis"]["published_radii"]}


def gate_sequences(rows, n_seq: int, lo: int, hi: int, seed: int):
    """`n_seq` TR0 sequences spread over lengths lo..hi, drawn by a recorded seed.

    Stratified rather than uniform, because the identity has to be exercised at the lengths the
    evaluation will see: a uniform draw from TR0 is almost all short sequences.
    """
    pool = [r for r in rows if lo <= len(r[2]) <= hi]
    pool.sort(key=lambda r: len(r[2]))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    edges = np.linspace(0, len(pool), n_seq + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if b > a:
            out.append(pool[int(rng.integers(a, b))])
    return out


# ----------------------------------------------------------------------------------
# G1: the callback identity
# ----------------------------------------------------------------------------------
def il_keys_against_viennarna(lc: LoopCounts, theta0: dict, rows) -> dict:
    """il_keys . theta0 == RNA.E_IntLoop, over every interior loop of `rows`.

    The oracle that makes G1 more than two implementations agreeing with each other.
    RNA.E_IntLoop is ViennaRNA's own interior-loop energy and shares nothing with either the
    callback's key reconstruction or the Extractor's loop walk, so a wrong orientation for the
    enclosed pair, a wrong mismatch base or a wrong int21 orientation fails here. Reported with
    the case coverage, because a green identity proves nothing if the rows never build the hard
    loops.
    """
    import RNA

    from thermo.vienna_ref import assert_turner2004_params, reference_md

    fc = RNA.fold_compound("GGGGAAAACCCC", reference_md())
    assert_turner2004_params(fc)
    par = fc.params
    bad, n = [], 0
    cases: dict[str, int] = {}
    bulges, asym = set(), set()
    for tag, seq, db in rows:
        S = [0] + [BCODE[c] for c in seq] + [0]
        for rec in lc.loops(seq, db):
            if rec["kind"] != "interior":
                continue
            i, j, k, ll = rec["i"], rec["j"], rec["k"], rec["l"]
            keys = il_keys(S, i, j, k, ll)
            mine = sum(theta0[key] * v for key, v in keys.items())
            n1, n2 = k - i - 1, j - ll - 1
            want = RNA.E_IntLoop(n1, n2, PT[S[i]][S[j]], PT[S[ll]][S[k]],
                                 S[i + 1], S[j - 1], S[k - 1], S[ll + 1], par)
            n += 1
            cases[rec["sub"]] = cases.get(rec["sub"], 0) + 1
            if rec["sub"] == "bulge":
                bulges.add(max(n1, n2))
            if n1 != n2:
                asym.add(n1 > n2)
            if mine != want:
                bad.append({"row": tag, "sub": rec["sub"], "ijkl": [i, j, k, ll],
                            "n1_n2": [n1, n2], "keys_dot_theta0": mine, "E_IntLoop": want})
            own = {a: b for a, b in rec["counts"].items() if b}
            if own != {a: b for a, b in keys.items() if b}:
                bad.append({"row": tag, "sub": rec["sub"] + " (checker disagrees)",
                            "callback": dict(keys), "checker": own})
    return {
        "n_interior_loops": n,
        "cases": dict(sorted(cases.items(), key=lambda z: -z[1])),
        "bulge_sizes": {"n": len(bulges), "min": min(bulges) if bulges else None,
                        "max": max(bulges) if bulges else None},
        "both_asymmetry_directions": bool(asym == {True, False}),
        "mismatches": bad[:10],
        "n_mismatches": len(bad),
        "pass": not bad,
        "oracle": "RNA.E_IntLoop on the pinned compound's own parameters",
    }


def sample_structures(seq: str, k: int, seed: int) -> list[str]:
    """`k` structures of `seq` by stochastic backtracking.

    uniq_ML has to be 1 for pbacktrack, and it is a field the pinned REFERENCE_MODEL fixes at 0,
    so the sampling runs on its OWN compound and never on the one any energy comes from. The
    sampling distribution is irrelevant here: these structures are inputs to an identity, not a
    measurement.
    """
    import RNA

    from thermo.vienna_ref import reference_md

    md = reference_md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(seq, md)
    fc.pf()
    RNA.init_rand(seed)
    out = fc.pbacktrack(k)
    return [s for s in (out if isinstance(out, (list, tuple)) else [out]) if s]


def g1_identity(lc: LoopCounts, W: Weights, seq: str, structures, star_db: str) -> dict:
    """eval_structure on the constrained compound == E0(s) + the checker's signed ceil weights.

    The left side is ViennaRNA's own decomposition driving the callback; the right side is the
    Extractor's loop walk. eval_structure honours only the hairpin and interior-loop codes,
    which is exactly the set the callback is nonzero on, so the identity is well posed; it would
    not be if the basis reached a multiloop or exterior entry.
    """
    import RNA

    from thermo.vienna_ref import assert_turner2004_params, reference_md

    S = [0] + [BCODE[c] for c in seq] + [0]
    star = star_interior_loops(lc, seq, star_db)
    plain = RNA.fold_compound(seq, reference_md())
    assert_turner2004_params(plain)
    con = RNA.fold_compound(seq, reference_md())
    calls = [0]
    cb = make_callback(S, star, W, calls)
    rc = con.sc_add_f(cb)
    bad, n, nonzero = [], 0, 0
    for s in structures:
        e0 = int(round(plain.eval_structure(s) * 100))
        got = int(round(con.eval_structure(s) * 100))
        want = e0 + eprime_offset(lc, seq, s, star, W)
        n += 1
        if want != e0:
            nonzero += 1
        if got != want:
            bad.append({"structure": s[:60], "eval_constrained": got, "expected": want,
                        "E0": e0})
    return {"n": n, "n_with_a_nonzero_offset": nonzero, "sc_add_f_rc": int(rc),
            "callback_invocations": calls[0], "n_mismatches": len(bad),
            "mismatches": bad[:5], "pass": not bad}


# ----------------------------------------------------------------------------------
# build-basis
# ----------------------------------------------------------------------------------
def tr0_rows():
    """The TR0 half of the registered loader, distinct and pair-bearing.

    tr0_ts0_rows tags every row "TR0" or "TS0" (turner_nullspace_tr0.py:146), so the only
    difference from the basis every other experiment used is the filter on that tag. No TS0
    sequence takes part in the basis, which is what removes the leakage the design doc names.
    """
    rows, stats = tr0_ts0_rows()
    tr0 = [r for r in distinct_rows(rows) if r[0] == "TR0" and "(" in r[3]]
    assert tr0, "no TR0 rows"
    return tr0, stats


def four_table_projection(B: dict):
    """B_four: B_TR0's columns projected onto the four PAIR_IL tables' coordinates.

    Returns (B_four in used coordinates, diagnostics). The projection is a coordinate mask
    followed by re-orthonormalisation, so B_four is an orthonormal basis of the image of the
    ball under that mask, and a callback that returns 0 outside PAIR_IL is exact for it.
    """
    cols, used, NT = B["cols"], B["used"], B["NT"]
    tab = np.array([cols[e][0] for e in used])
    mask = np.isin(tab, FOUR)
    off = NT[~mask]                                   # the rows the table mask drops
    M = NT * mask[:, None]
    # the second mask. A row norm at ROW_NOISE or below is the basis's own arithmetic, not a
    # direction: among the rows that do carry one the SMALLEST norm is 0.36, thirteen orders of
    # magnitude above the 4e-14 the rest sit at, so the cut is unambiguous rather than a
    # threshold choice. Left in, such a row costs a full dcal on every loop that touches it,
    # because ceil of any positive weight is at least 1, and it makes rho divide by noise.
    rn = np.linalg.norm(M, axis=1)
    phys = rn > ROW_NOISE
    noise = M[mask & ~phys]
    Q, s_, _ = np.linalg.svd(M * phys[:, None], full_matrices=False)
    keep = s_ > 1e-8
    B4 = Q[:, keep]
    # the whole point of the projection is that a callback returning 0 outside these rows is
    # EXACT for this basis, so they are set to exactly zero rather than left at the SVD's
    # 1e-15: LAPACK has no reason to deliver them as zeros and did not
    B4[~(mask & phys)] = 0.0
    diag = {
        "n_used": int(len(used)),
        "n_used_in_four_tables": int(mask.sum()),
        "n_used_outside_four_tables": int((~mask).sum()),
        "n_rows_zeroed_as_noise": int((mask & ~phys).sum()),
        "row_noise_floor": ROW_NOISE,
        "dim_B_TR0": int(NT.shape[1]),
        "dim_B_four": int(B4.shape[1]),
        "dropped_frobenius": float(np.linalg.norm(off)),
        "dropped_max_row_norm": float(np.linalg.norm(off, axis=1).max()) if off.size else 0.0,
        "noise_dropped_frobenius": float(np.linalg.norm(noise)),
        "noise_dropped_max_row_norm": (float(np.linalg.norm(noise, axis=1).max())
                                       if noise.size else 0.0),
        "smallest_row_norm_kept": float(rn[phys].min()) if phys.any() else 0.0,
        "singular_values_dropped": [float(x) for x in s_[~keep][:8]],
        "weight_outside_four_tables": float((off ** 2).sum() / (NT ** 2).sum()),
        "weight_in_the_noise_rows": float((noise ** 2).sum() / (NT ** 2).sum()),
    }
    return B4, diag


def table_norms(cols, used, V):
    """Per-table squared-norm share and the largest per-entry row norm, unrounded."""
    rn = np.linalg.norm(V, axis=1)
    tot = float((V ** 2).sum())
    by: dict[str, dict] = {}
    for k, e in enumerate(used):
        t = cols[e][0]
        d = by.setdefault(t, {"share": 0.0, "max_row_norm": 0.0, "n_used": 0})
        d["share"] += float(rn[k] ** 2)
        d["max_row_norm"] = max(d["max_row_norm"], float(rn[k]))
        d["n_used"] += 1
    for d in by.values():
        d["share"] = d["share"] / tot if tot else 0.0
    return dict(sorted(by.items(), key=lambda z: -z[1]["share"]))


def compare_spans(V_tr0: np.ndarray, I_tr0: np.ndarray, V_ref: np.ndarray) -> dict:
    """How the TR0-only span relates to the TR0+TS0 one, in TR0's used coordinates.

    `principal_overlap` answers this only for spans of equal dimension and returns NaN
    otherwise, and these differ: dropping TS0's references can only enlarge the null space, so
    B_TR0 has more directions than the TR0+TS0 basis.

    The fact that does hold, and the one worth checking, is this. Let v satisfy A_ref v = 0.
    Every column of A_TR0 for an entry no TR0 reference uses is identically zero, so
    A_TR0 v = A_TR0 v_used = 0, and v's restriction to TR0's used entries lies in TR0's FULL
    within-used null space -- nontrivial AND gauge. It need not lie in the nontrivial part
    alone: v is orthogonal to the gauge in the 4,561 coordinates TR0+TS0 uses, and restricting
    it to TR0's 4,337 can give it a gauge component. Checking against the nontrivial part alone
    reports a residual near 1 and means nothing, which is what a first run of this phase did.

    So the containment is tested against [nontrivial | gauge], and the nontrivial-only cosine
    spectrum is reported separately, as the measure of how much the basis moves.
    """
    Qa, _ = np.linalg.qr(V_tr0)
    Qfull, _ = np.linalg.qr(np.hstack([V_tr0, I_tr0]))
    # The statement is about the UN-NORMALISED restriction v_used, so the residual has to be
    # absolute. Orthonormalising the restricted reference basis first is what a first run did,
    # and it reports a residual near 1 for a reason that says nothing about the mathematics: a
    # TR0+TS0 direction living mostly on the 224 entries only TS0 uses restricts to a nearly
    # zero vector, and QR renormalises that noise into an arbitrary unit vector.
    # The restricted reference basis loses rank -- 426 directions of the TR0+TS0 space do not
    # stay independent once confined to TR0's entries -- so its orthonormal basis comes from an
    # SVD with an explicit tolerance. np.linalg.qr pads a rank-deficient matrix with arbitrary
    # directions, and a residual taken against those is what produced a meaningless 0.999 and a
    # count of 448 on the first two runs of this phase.
    U_ref, s_ref, _ = np.linalg.svd(V_ref, full_matrices=False)
    keep = s_ref > 1e-8 * s_ref.max()
    Qb = U_ref[:, keep]
    back = np.linalg.norm(Qa - Qb @ (Qb.T @ Qa), axis=0)
    norms = np.linalg.norm(V_ref, axis=0)
    resid_full = np.linalg.norm(V_ref - Qfull @ (Qfull.T @ V_ref), axis=0)
    resid_nt = np.linalg.norm(V_ref - Qa @ (Qa.T @ V_ref), axis=0)
    return {
        "restricted_reference_columns": {
            "n": int(V_ref.shape[1]),
            "norm_min": float(norms.min()), "norm_max": float(norms.max()),
            "n_nearly_vanishing_below_1e_6": int((norms < 1e-6).sum()),
            "rank_at_1e_8": int(keep.sum()),
            "rank_lost_by_restriction": int(V_ref.shape[1] - keep.sum()),
            "singular_values_smallest_five": [float(x) for x in np.sort(s_ref)[:5]],
        },
        "abs_residual_in_B_TR0_plus_gauge_max": float(resid_full.max()),
        "abs_residual_in_B_TR0_plus_gauge_rel_max": float(
            (resid_full / np.maximum(norms, 1e-300)).max()),
        "containment_holds": bool(resid_full.max() < 1e-8),
        "abs_residual_in_B_TR0_nontrivial_only_max": float(resid_nt.max()),
        # How much larger the TR0-only space is, stated so that it does not depend on which
        # basis was drawn: the dimensions, not a count over basis vectors. Every vector of a
        # 448-dimensional space has a component outside a 399-dimensional subspace of it, so a
        # per-vector count is always the full 448 and says nothing.
        "dimension_arithmetic": {
            "tr0ts0_nontrivial": int(V_ref.shape[1]),
            "surviving_restriction_to_tr0_entries": int(keep.sum()),
            "vanishing_on_tr0_entries": int(V_ref.shape[1] - keep.sum()),
            "B_TR0_nontrivial": int(V_tr0.shape[1]),
            "new_relative_to_the_restricted_reference_span": int(V_tr0.shape[1] - keep.sum()),
            "note": "directions that vanish on TR0's entries live entirely on entries only TS0 "
                    "uses, and are coverage rather than Type B once TS0 is held out; the new "
                    "ones are what TS0's references had been pinning",
        },
        "residual_of_B_TR0_basis_against_restricted_reference_span": {
            "max": float(back.max()), "min": float(back.min()),
        },
    }


def published_radii(B: dict, B4: np.ndarray) -> dict:
    """The three published components measured on B_four, which set the evaluation radii.

    delta_X = theta_X - theta_0 in dcal; the radius is the norm in kcal/mol of its component
    along B_four, i.e. the smallest r whose ball reaches that component. r_cap, which the
    registered rho edge case needs, is R_CAP_MULTIPLE times the largest of them.
    """
    from scripts.typeb_common import full_tables

    cols, used = B["cols"], B["used"]
    t0 = template()["theta"]
    out = {}
    for tb in full_tables():
        th = tb["theta"]
        d = np.array([th[k] - t0[k] for k in cols], dtype=np.float64) / 100.0
        inf = np.array([th[k] >= 100000 for k in cols])          # Langdon 2018 forbids entries
        d = np.where(inf, 0.0, d)
        du = d[used]
        comp = B4 @ (B4.T @ du)
        out[tb["id"].split(":")[-1]] = {
            "norm_delta_kcal": float(np.linalg.norm(d)),
            "radius_on_B_four_kcal": float(np.linalg.norm(comp)),
            "n_entries_set_to_INF_and_skipped": int(inf.sum()),
        }
    out["r_cap_kcal"] = R_CAP_MULTIPLE * max(
        v["radius_on_B_four_kcal"] for v in out.values() if isinstance(v, dict))
    out["r_cap_multiple"] = R_CAP_MULTIPLE
    return out


def per_loop_shift(B: dict, B4: np.ndarray, a_four: np.ndarray, radii: dict, rows,
                   n_seq: int | None = None) -> dict:
    """What the ball actually permits per loop, beside what a published table actually does.

    The authors' amendment of 2026-10-13. A ball of radius 21 kcal/mol may put its whole norm on
    a single int22 entry, while the Andronescu component spreads the same norm over hundreds of
    directions, so the two per-entry scales are nowhere near each other and the certified
    fractions have to be read against that rather than as a performance number. Measured over
    the loops of s* on TR0 MFE structures: the worst-case per-loop shift the ball allows,
    w_L(r)/r = sum_e |c_{L,e}| ||a_e||, at each published radius; and, on the same entries, the
    shift the published component itself produces.
    """
    import RNA

    from scripts.typeb_common import full_tables
    from thermo.vienna_ref import assert_turner2004_params, reference_md

    P = template()
    cols = B["cols"]
    idx = {k: n for n, k in enumerate(cols)}
    t0 = P["theta"]
    deltas = {}
    for tb in full_tables():
        th = tb["theta"]
        deltas[tb["id"].split(":")[-1]] = np.array(
            [0.0 if th[k] >= 100000 else (th[k] - t0[k]) / 100.0 for k in cols])

    lc = LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    md = reference_md()
    rows = rows if n_seq is None else rows[:n_seq]
    unit, entry_ball, entry_pub = [], [], {x: [] for x in deltas}
    for n, (_tag, _name, seq, _db) in enumerate(rows):
        fc = RNA.fold_compound(seq, md)
        if n == 0:
            assert_turner2004_params(fc)
        s_star, _e = fc.mfe()
        for rec in lc.loops(seq, s_star):
            w = sum(abs(v) * a_four[idx[k]] for k, v in rec["counts"].items() if k in idx)
            unit.append(w)
            for k in rec["counts"]:
                if k not in idx or a_four[idx[k]] <= ROW_NOISE:
                    continue
                entry_ball.append(a_four[idx[k]])
                for x, d in deltas.items():
                    entry_pub[x].append(abs(d[idx[k]]))
    unit = np.array(unit)
    ball = np.array(entry_ball)
    # A four-table entry TR0's references do pin still carries a row norm at the basis's noise
    # floor, around 1e-15, so "any weight at all" counts almost every interior loop and its
    # median is 1e-14. The bar that matters is physical: ViennaRNA's energies are integer
    # dcal/mol, so an allowance below half a dcal cannot move an energy at all.
    real = unit > ROW_NOISE

    def q(v, scale=1.0):
        if not len(v):
            return None
        return {"median": float(np.median(v) * scale), "p95": float(np.quantile(v, 0.95) * scale),
                "p99": float(np.quantile(v, 0.99) * scale), "max": float(v.max() * scale)}

    per_radius = {}
    for x, v in radii.items():
        if not isinstance(v, dict):
            continue
        r = v["radius_on_B_four_kcal"]
        at_r = unit * r
        per_radius[x] = {
            "radius_kcal": r,
            "per_loop_shift_the_ball_allows_kcal_over_loops_above_the_noise_floor": q(
                unit[real], r),
            "loops_the_ball_can_move": {
                "above_half_a_dcal": int((at_r >= 0.005).sum()),
                "above_0.1_kcal": int((at_r >= 0.1).sum()),
                "above_1_kcal": int((at_r >= 1.0).sum()),
                "fraction_above_half_a_dcal": float((at_r >= 0.005).mean()),
            },
            "per_entry_shift_the_ball_allows_kcal": q(ball, r),
            "per_entry_shift_this_component_makes_kcal": q(np.array(entry_pub[x])),
        }
    return {
        "n_sequences_folded": len(rows),
        "n_loops": int(len(unit)),
        "n_loops_above_the_row_noise_floor": int(real.sum()),
        "row_noise_floor": ROW_NOISE,
        "fraction_of_loops_above_the_noise_floor": float(real.mean()) if len(unit) else 0.0,
        "by_radius": per_radius,
        "expectation_user": "at the published radii, few pairs that depend on an internal loop "
                            "will be certified, because the set allows the entire norm on one "
                            "entry; the descriptive label fractions are read against this, not "
                            "as a performance number",
        "what_this_does_not_imply_user": "This counts the loops of s* only, and it does NOT "
            "imply that nearly every pair certifies because most of s*'s loops are untouched "
            "(a wrong gloss, corrected by the authors on 2026-10-13 before preregistration). "
            "The certificate credits every competitor structure -w_L for each touched loop IT "
            "contains, up to about 19 kcal/mol at the Andronescu radius, so a pair is uncertain "
            "as soon as some structure lacking it, within that allowance of the MFE, forms a "
            "touched loop. That depends on the competitors' loops, not on s*'s. The expectation "
            "PRED carries is written from the TR0 pilot's measured label fractions instead.",
    }


def basis_quality(B: dict, V: np.ndarray, label: str) -> dict:
    """How well V behaves as a null basis: orthonormality and annihilation residuals.

    These set the scale against which a per-entry row norm counts as zero. An entry that some
    direction genuinely moved would carry a row norm of order 1e-1; one at 1e-15, against an
    annihilation residual of order 1e-13, is identifiable and the basis simply cannot say
    otherwise. This replaces the mod-p rank corroboration, which would need four Gaussian
    eliminations over 4,561 columns of a 10,682-row matrix: hours, for a necessary condition
    that the row norms already answer. Soundness does not rest on either, because the set this
    run certifies over is the ball in B_four, which lies exactly inside the four tables.
    """
    A = B["A"]
    gram = V.T @ V
    annih = A @ V if V.size else np.zeros((A.shape[0], 0))
    return {
        "label": label,
        "dim": int(V.shape[1]),
        "orthonormality_max_abs": float(np.abs(gram - np.eye(V.shape[1])).max())
        if V.size else 0.0,
        "annihilation_max_abs_dcal": float(np.abs(annih).max()) if V.size else 0.0,
        "row_norm_max": float(np.linalg.norm(V, axis=1).max()) if V.size else 0.0,
    }


def build_basis() -> int:
    out = read_out()
    if "predictions" in out:
        raise SystemExit("preregistered: the basis is fixed")
    if git_provenance(R)["git_dirty_excluding_runs"]:
        raise SystemExit("the tree is not clean outside runs/: commit the code first")
    t0 = time.time()
    SHARDS.mkdir(parents=True, exist_ok=True)

    P = template()
    rows, loader = tr0_rows()
    B = compute_basis(rows, P)                      # exact per-row reconstruction gate inside
    cols, used, NT = B["cols"], B["used"], B["NT"]

    # ---- what the four-table projection costs, and the set this run certifies over
    B4, proj = four_table_projection(B)

    # ---- per-entry row norms, lifted to all 12,607 entries, for both bases
    def lifted_norms(V):
        full = np.zeros(len(cols))
        full[used] = np.linalg.norm(V, axis=1)
        return full

    a_tr0, a_four = lifted_norms(NT), lifted_norms(B4)
    tabs_tr0 = table_norms(cols, used, NT)
    tabs_four = table_norms(cols, used, B4)

    # Which entries B_four actually reaches. This is the population a future melting study would
    # have to measure, and it bounds how many MFE pairs can be uncertain at all: a pair whose
    # loops read none of these entries is certified at every radius, trivially.
    reach = np.nonzero(a_four > ROW_NOISE)[0]
    by_tab: dict[str, int] = {}
    for e in reach:
        by_tab[cols[e][0]] = by_tab.get(cols[e][0], 0) + 1
    entries_reached = {
        "n": int(len(reach)),
        "of_used": int(len(used)),
        "by_table": dict(sorted(by_tab.items(), key=lambda z: -z[1])),
        "row_norm": {"min": float(a_four[reach].min()) if len(reach) else None,
                     "median": float(np.median(a_four[reach])) if len(reach) else None,
                     "max": float(a_four[reach].max()) if len(reach) else None},
        "noise_floor": ROW_NOISE,
    }

    # ---- must-holds
    idx = {k: n for n, k in enumerate(cols)}
    ninio = {str(k): float(a_tr0[idx[k]]) for k in (("ninio_m",), ("ninio_max",)) if k in idx}
    outside = {t: d for t, d in tabs_tr0.items() if t not in FOUR}
    worst_outside = max((d["share"] for d in outside.values()), default=0.0)
    must = {
        "ninio_rows_on_B_TR0": ninio,
        "ninio_bar": WEIGHT_BAR,
        "ninio_ok": all(v <= WEIGHT_BAR for v in ninio.values()),
        "largest_share_outside_four_tables": worst_outside,
        "share_bar": WEIGHT_BAR,
        "outside_ok": worst_outside <= WEIGHT_BAR,
        "B_four_is_exactly_inside_four_tables": float(
            np.linalg.norm(B4[~np.isin(np.array([cols[e][0] for e in used]), FOUR)])),
    }
    if not must["ninio_ok"]:
        raise GuardFailure(f"build-basis: B_TR0 has weight on the ninio entries ({ninio}); the "
                           f"count model is not linear there and they must be pinned out")
    if not must["outside_ok"]:
        raise GuardFailure(
            f"build-basis: B_TR0 carries share {worst_outside:.3g} outside {FOUR}, above "
            f"{WEIGHT_BAR}: the callback needs the other decomposition codes "
            f"({ {t: d['share'] for t, d in outside.items()} })")

    # the scale against which those row norms count as zero
    quality = [basis_quality(B, NT, "B_TR0"), basis_quality(B, B4, "B_four")]

    # ---- how far B_TR0 sits from the basis every other experiment used
    try:
        ref = load_basis("tr0ts0")
    except (AssertionError, FileNotFoundError, OSError) as exc:
        ref = None
        ref_why = f"{type(exc).__name__}: {exc}"
    shared = np.intersect1d(used, ref["used"]) if ref else np.array([], dtype=int)
    def on_shared(V, u):
        pos = {int(e): k for k, e in enumerate(u)}
        return V[[pos[int(e)] for e in shared]]

    if ref is None:
        angles = {"unavailable": ref_why,
                  "note": "the TR0+TS0 basis is recomputable by turner_typeb's bases phase; "
                          "its npz is gitignored and was not on disk"}
    else:
        angles = compare_spans(on_shared(NT, used), on_shared(B["I"], used),
                               on_shared(ref["NT"], ref["used"]))
        angles.update({
            "n_shared_used_columns": int(len(shared)),
            "tr0ts0_used_columns": int(len(ref["used"])),
            "tr0_used_columns": int(len(used)),
            "dim_tr0ts0": int(ref["NT"].shape[1]),
            "dim_B_TR0": int(NT.shape[1]),
            "extra_directions_in_B_TR0": int(NT.shape[1] - ref["NT"].shape[1]),
        })

    # ---- the radii, r_cap, and what the ball permits per loop (the authors' amendment 2)
    radii = published_radii(B, B4)
    shifts = per_loop_shift(B, B4, a_four, radii, rows)

    files = {}
    np.savez_compressed(SHARDS / "basis_tr0.npz", cols=np.array([repr(k) for k in cols]),
                        used=used, NT=NT, B4=B4, a_tr0=a_tr0, a_four=a_four)
    files["basis_tr0.npz"] = _sha(SHARDS / "basis_tr0.npz")

    # the gate is a statement about a basis: rebuilding one invalidates every shard that ran
    # against the old one, so they are dropped here rather than left to look current. The count
    # of what was dropped goes in the record, because a reader of the artifact is entitled to
    # know the gate was re-run rather than inherited.
    stale = sorted(out.pop("gate", {}))
    out.update({
        "status": "basis built",
        "gate_shards_dropped_on_rebuild": stale,
        "viennarna": VIENNARNA,
        "build_basis": {
            "rows": {"n": len(rows), "loader": loader, "tag": "TR0 only"},
            "dims": {"cols": len(cols), "used": int(len(used)),
                     "null_within_used": B["null_within_used"], "dim_B_TR0": int(NT.shape[1]),
                     "dim_B_four": int(B4.shape[1])},
            "qr": B["qr"],
            "four_table_projection": proj,
            "weight_by_table_B_TR0": tabs_tr0,
            "weight_by_table_B_four": tabs_four,
            "entries_reached_by_B_four": entries_reached,
            "must_hold": must,
            "basis_quality": quality,
            "published_radii": radii,
            "per_loop_shift": shifts,
            "principal_angles": angles,
            "files_sha256": files,
            "seconds": round(time.time() - t0, 1),
            "provenance": git_provenance(R),
        },
    })
    write(out)
    print(json.dumps({"dims": out["build_basis"]["dims"],
                      "four_table_projection": proj,
                      "must_hold": must,
                      "published_radii": radii,
                      "per_loop_shift": shifts,
                      "principal_angles": angles}, indent=1, default=str))
    return 0


def gate_g1() -> int:
    """G1: the callback identity, with the constructed-case extension.

    Two halves, and the second is the one the authors added on 2026-10-13. The sampled half
    takes about 1,000 stochastic structures on each of 20 TR0 sequences spread over n = 50 to
    500. The constructed half runs every row of the generating set, which builds every
    vrna_E_internal case deliberately -- the full int11, int21 and int22 tables, 1xn and 2x3,
    18 generic shapes, bulges of 1 to 30 on either side -- and checks the keys against
    RNA.E_IntLoop as well as the identity. That closes the hole G4 cannot: G4's band is computed
    from the callback, so a callback that charges a loop too LITTLE shrinks the band and hides
    the very structures that would break the certificate, and G4 passes vacuously.
    """
    out = read_out()
    assert out.get("status") == "basis built", out.get("status")
    if "predictions" in out:
        raise SystemExit("preregistered: the gate's shards are fixed")
    t0 = time.time()
    B = load_tr0_basis()
    P = template()
    lc = LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    rows, _ = tr0_rows()

    from scripts.turner_nullspace import generating_set

    gen = generating_set(P["special"])
    oracle = il_keys_against_viennarna(lc, P["theta"], gen)
    if not oracle["pass"]:
        raise GuardFailure(f"G1: il_keys disagrees with RNA.E_IntLoop on "
                           f"{oracle['n_mismatches']} interior loops: {oracle['mismatches'][:2]}")

    radii = {"r1": 1.0, "Andronescu2007": B["radii"]["Andronescu2007"]["radius_on_B_four_kcal"],
             "r_cap": B["radii"]["r_cap_kcal"]}
    seqs = gate_sequences(rows, G1_SEQUENCES, 50, 500, SEED)
    sampled, constructed = {}, {}
    for name, r in radii.items():
        W = Weights(B["a_four"], B["cols"], r)
        per_seq = []
        for n, (_t, nm, seq, _db) in enumerate(seqs):
            import RNA

            from thermo.vienna_ref import reference_md
            fc = RNA.fold_compound(seq, reference_md())
            s_star, _e = fc.mfe()
            structs = sample_structures(seq, G1_SAMPLES, SEED + n) + [s_star]
            res = g1_identity(lc, W, seq, structs, s_star)
            res.update({"name": nm, "n_nt": len(seq)})
            per_seq.append(res)
            if not res["pass"]:
                raise GuardFailure(
                    f"G1: the callback identity failed at r = {r} on {nm} "
                    f"({res['n_mismatches']} of {res['n']}): {res['mismatches'][:2]}")
        sampled[name] = {
            "radius_kcal": r, "n_sequences": len(per_seq),
            "n_structures": sum(x["n"] for x in per_seq),
            "n_with_a_nonzero_offset": sum(x["n_with_a_nonzero_offset"] for x in per_seq),
            "callback_invocations": sum(x["callback_invocations"] for x in per_seq),
            "lengths": [x["n_nt"] for x in per_seq], "pass": True,
        }
        # The constructed half: every row of the generating set, under two reference structures.
        # With s* = the row's own structure every loop is a loop of s* and only the + branch
        # fires; with s* all-unpaired no loop is, and only the - branch does. Every
        # vrna_E_internal case therefore gets checked with both signs, which a sampled structure
        # set covers only where it happens to differ from s* in a weighted loop.
        bad, n_rows, nonzero = [], 0, {"plus": 0, "minus": 0}
        for _tag, seq, db in gen:
            for sign, star_db in (("plus", db), ("minus", "." * len(seq))):
                res = g1_identity(lc, W, seq, [db], star_db)
                nonzero[sign] += res["n_with_a_nonzero_offset"]
                if not res["pass"]:
                    bad.append({"seq": seq, "db": db, "sign": sign, **res})
            n_rows += 1
        if bad:
            raise GuardFailure(f"G1 constructed cases: {len(bad)} rows failed at r = {r}: "
                               f"{bad[:2]}")
        constructed[name] = {"radius_kcal": r, "n_rows": n_rows,
                             "n_with_a_nonzero_offset": nonzero, "pass": True}

    out.setdefault("gate", {})["G1"] = {
        "sampled": sampled,
        "constructed": constructed,
        "il_keys_vs_viennarna": oracle,
        "samples_per_sequence": G1_SAMPLES,
        "sequences": G1_SEQUENCES,
        "seed": SEED,
        "radii_kcal": radii,
        "pass": True,
        "seconds": round(time.time() - t0, 1),
        "provenance": git_provenance(R),
    }
    out["status"] = "G1 passed"
    write(out)
    print(json.dumps({"G1": {k: v for k, v in out["gate"]["G1"].items()
                             if k not in ("provenance", "il_keys_vs_viennarna")},
                      "oracle": {k: v for k, v in oracle.items() if k != "mismatches"}},
                     indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------------
# G2: the labels against full enumeration at n <= 20
# ----------------------------------------------------------------------------------
def offset_from_map(star_map: dict, s_map: dict, r: float) -> int:
    """E'(s) - E0(s) in dcal from a precomputed loop map: the callback's arithmetic, once.

    eprime_offset re-walks the structure for every radius, and G2 evaluates every enumerated
    structure at three; the walk is done once and only the ceiling applied here. The two must
    agree exactly, which G2 checks on a sample rather than assuming.
    """
    tot = 0
    for k, d in s_map.items():
        w = r * d
        if w <= 0:
            continue
        c = int(math.ceil(w * 100.0 - 1e-12))
        tot += c if k in star_map else -c
    return tot


def basis_image(B4: np.ndarray, pos: dict, counts: dict) -> np.ndarray:
    """B_four^T c, the count vector's image in the ball's coordinates."""
    v = np.zeros(B4.shape[1])
    for k, c in counts.items():
        p = pos.get(k)
        if p is not None and c:
            v += c * B4[p]
    return v


def g2_sequences(rows, n_seq: int, lo: int, hi: int, seed: int, lc: LoopCounts,
                 W1: Weights) -> tuple[list, dict]:
    """Sequences for G2: half random, half TR0 fragments, each screened for a reachable loop.

    The screen is the authors' condition that the structure spaces contain int11/int21/int22
    or bulge loops the basis actually reaches. Measured first and recorded: every 20-mer tried,
    random or real, has such a structure, so the screen rejects nothing and the sample is not
    quietly selected -- which is worth recording precisely because it could have been.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    pool = [r[2] for r in rows if len(r[2]) >= hi]
    out, screened, rejected = [], 0, 0
    while len(out) < n_seq:
        screened += 1
        n = int(rng.integers(lo, hi + 1))
        if len(out) % 2 == 0:
            seq = "".join(rng.choice(list("ACGU"), n))
            src = "random"
        else:
            s = pool[int(rng.integers(len(pool)))]
            a = int(rng.integers(0, len(s) - n + 1))
            seq, src = s[a:a + n], "TR0 fragment"
        structs = enumerate_structures(seq)
        if not any(any(v > 0 for v in loop_coeff_map(lc, seq, s, W1).values()) for s in structs):
            rejected += 1
            continue
        out.append((src, seq, len(structs)))
    return out, {"n_screened": screened, "n_rejected_no_reachable_loop": rejected,
                 "n_kept": len(out), "lengths": [lo, hi]}


def g2_one(lc: LoopCounts, B: dict, pos: dict, W1: Weights, seq: str, radii,
           scale: float = 1.0, with_rho: bool = True) -> dict:
    """One sequence of G2: the pipeline's labels against the exact worst case over the ball.

    The exact test is min over structures without p of [N(s) - r ||B^T (c(s) - c(s*))||], which
    for a ball is the worst case and not a bound. The certificate replaces that norm by the
    per-entry sum, which the triangle inequality makes no smaller, so a certified pair must be
    exactly robust; a pair that is exactly robust need not be certified, and the gap between the
    two is the slack this reports rather than hides.

    `scale` multiplies the PIPELINE's weights only, leaving the exact test alone: that is the
    authors' power check, and at scale 0.5 it must produce false certificates.
    """
    import RNA

    from thermo.vienna_ref import assert_turner2004_params, reference_md

    plain = RNA.fold_compound(seq, reference_md())
    assert_turner2004_params(plain)
    star, e_mfe = plain.mfe()
    allS = enumerate_structures(seq)
    if star not in allS:
        raise GuardFailure(f"the MFE structure is not in the enumeration: {seq} {star}")
    e0 = np.array([int(round(plain.eval_structure(s) * 100)) for s in allS])
    k_star = allS.index(star)
    if e0[k_star] != e0.min() or abs(e0[k_star] / 100.0 - e_mfe) > 1e-6:
        raise GuardFailure(f"ViennaRNA's MFE is not the enumeration's minimum on {seq}: "
                           f"{e0[k_star]} vs {e0.min()}")
    star_map = loop_coeff_map(lc, seq, star, W1)
    maps = [loop_coeff_map(lc, seq, s, W1) for s in allS]
    u_star = basis_image(B["B4"], pos, lc.counts(seq, star))
    norms = np.array([float(np.linalg.norm(basis_image(B["B4"], pos, lc.counts(seq, s)) - u_star))
                      for s in allS])
    D = np.array([d_coeff(star_map, m) for m in maps])
    N = (e0 - e0[k_star]) / 100.0
    if (N < 0).any():
        raise GuardFailure(f"a structure beats the MFE on {seq}")
    if (norms > D + 1e-9).any():
        raise GuardFailure(f"the per-entry sum is below the exact norm on {seq}: the triangle "
                           f"inequality does not hold, so the weights are wrong")

    pt_star = pair_table(star)
    star_pairs = [(i, pt_star[i]) for i in range(1, len(seq) + 1) if pt_star[i] > i]
    adm = admissible_pairs(seq)
    holds: dict[tuple, list] = {}
    for k, s in enumerate(allS):
        pt = pair_table(s)
        for i in range(1, len(seq) + 1):
            if pt[i] > i:
                holds.setdefault((i, pt[i]), []).append(k)

    res = {"seq": seq, "n_nt": len(seq), "n_structures": len(allS),
           "n_pairs_star": len(star_pairs), "n_admissible": len(adm),
           "n_structures_the_ball_can_move": int((D > 0).sum()), "by_radius": {}}
    W_pipe = Weights(B["a_four"] * scale, B["cols"], radii[0])
    F = Folder(lc, seq, star, W_pipe)
    for r in radii:
        F.set_radius(r)
        c = certify(F, star_pairs, adm)
        eprime = e0 + np.array([offset_from_map(star_map, m, r * scale) for m in maps])
        if eprime[k_star] != F.eprime_star:
            raise GuardFailure(f"E'(s*) differs between the map and the walk on {seq}: "
                               f"{eprime[k_star]} != {F.eprime_star}")
        exact = N - r * norms
        bad_x, bad_z, false_p, false_a, robust_p, robust_a = [], [], [], [], 0, 0
        for p in star_pairs:
            keep = np.ones(len(allS), bool)
            keep[holds.get(p, [])] = False
            if not keep.any():
                continue
            if int(eprime[keep].min()) != c["margins_present_dcal"][p] + F.eprime_star:
                bad_x.append({"pair": p, "dp": c["margins_present_dcal"][p] + F.eprime_star,
                              "enumerated": int(eprime[keep].min())})
            ok = float(exact[keep].min()) > 0
            robust_p += ok
            if c["margins_present_dcal"][p] > 0 and not ok:
                false_p.append({"pair": p, "exact_margin_kcal": float(exact[keep].min())})
        for q, m in c["margins_absent_dcal"].items():
            idx = holds.get(q)
            if idx is None:
                robust_a += 1
                continue
            if m is not None and int(eprime[idx].min()) != m + F.eprime_star:
                bad_z.append({"pair": q, "zuker": m + F.eprime_star,
                              "enumerated": int(eprime[idx].min())})
            ok = float(exact[idx].min()) > 0
            robust_a += ok
            if (m is None or m > 0) and not ok:
                false_a.append({"pair": q, "exact_margin_kcal": float(exact[idx].min())})
        if bad_x or bad_z:
            raise GuardFailure(f"the pair tables disagree with enumeration on {seq} at r = {r}: "
                               f"X' {bad_x[:2]} Z' {bad_z[:2]}")
        rec = {
            "radius_kcal": r, "eprime_star_dcal": int(F.eprime_star),
            "n_certified_present": c["n_certified_present"],
            "n_certified_absent": c["n_certified_absent"],
            "n_exactly_robust_present": int(robust_p),
            "n_exactly_robust_absent": int(robust_a),
            "n_false_certificates": len(false_p) + len(false_a),
            "false_certificates": (false_p + false_a)[:3],
            "whole_structure_certified": c["whole_structure_certified"],
        }
        res["by_radius"][f"{r:g}"] = rec
    if with_rho:
        # rho is a radius, not a quantity at a radius, so it is computed once per pair and
        # compared against the exact minimum of N(s)/D(s) over the enumerated structures that
        # lack the pair. A positive difference would mean rho overstates robustness, which is
        # the one direction that matters: it is what the ceiled inner minimisation could cost.
        r_cap = float(B["radii"]["r_cap_kcal"])
        rhos = [rho_of_pair(F, star_map, W1, i, j, r_cap) for i, j in star_pairs]
        worst, n_cmp = 0.0, 0
        for (i, j), got in zip(star_pairs, rhos, strict=True):
            keep = np.ones(len(allS), bool)
            keep[holds.get((i, j), [])] = False
            m = (D > 0) & keep
            ex = float((N[m] / D[m]).min()) if m.any() else float("inf")
            if not got["capped"] and ex < float("inf"):
                worst = max(worst, got["rho"] - ex)
                n_cmp += 1
        res["rho"] = {
            "n_pairs": len(rhos), "n_compared_against_enumeration": n_cmp,
            "overstatement_vs_enumeration_kcal": worst,
            "capped": sum(x["capped"] for x in rhos),
            "tied_at_zero": sum(x["tied_at_zero"] for x in rhos),
            "unconverged": sum(not x["converged"] for x in rhos),
            "r_cap_kcal": r_cap,
        }
    res["folds"] = F.folds
    return res


def gate_g2() -> int:
    """G2: zero false certificates against full enumeration, and the power check must fail.

    Exhaustive rather than lazy: at n <= 20 the whole structure space is enumerated by code that
    shares nothing with ViennaRNA, so there is no band to grow and no structure to miss. That is
    what makes this the gate's backstop, and why G4 can afford to be lazy at realistic lengths.
    """
    out = read_out()
    assert out.get("status") in ("G1 passed", "G2 passed"), out.get("status")
    if "predictions" in out:
        raise SystemExit("preregistered: the gate's shards are fixed")
    t0 = time.time()
    B = load_tr0_basis()
    P = template()
    lc = LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    rows, _ = tr0_rows()
    W1 = Weights(B["a_four"], B["cols"], 1.0)
    pos = {B["cols"][e]: k for k, e in enumerate(B["used"])}
    seqs, screen = g2_sequences(rows, G2_SEQUENCES, 16, G2_MAX_N, SEED, lc, W1)

    per = []
    for src, seq, _n in seqs:
        rec = g2_one(lc, B, pos, W1, seq, G2_RADII)
        rec["source"] = src
        per.append(rec)
        bad = [v for v in rec["by_radius"].values() if v["n_false_certificates"]]
        if bad:
            raise GuardFailure(f"G2: {sum(v['n_false_certificates'] for v in bad)} false "
                               f"certificates on {seq}: {bad[0]['false_certificates'][:2]}")

    def agg(key):
        return {f"{r:g}": sum(p["by_radius"][f"{r:g}"][key] for p in per) for r in G2_RADII}

    # the power check: halving the weights must produce at least one false certificate. It is
    # the only test here that can fail by passing, so it runs on the same sequences and is
    # reported as a count, not a boolean -- a single false certificate is weak evidence that the
    # exact test has teeth, and the number says how much.
    planted, n_planted = 0, min(G2_POWER_SEQUENCES, len(seqs))
    for _src, seq, _n in seqs[:n_planted]:
        rec = g2_one(lc, B, pos, W1, seq, G2_RADII, scale=0.5, with_rho=False)
        planted += sum(v["n_false_certificates"] for v in rec["by_radius"].values())
    if planted == 0:
        raise GuardFailure(f"G2 power check: halving the weights produced no false certificate "
                           f"on {n_planted} sequences, so G2 is not sensitive enough to catch "
                           f"an under-charging callback. Enlarge it.")

    shard = {
        "sequences": screen,
        "n_sequences": len(per),
        "radii_kcal": list(G2_RADII),
        "n_structures_total": sum(p["n_structures"] for p in per),
        "n_pairs_star_total": sum(p["n_pairs_star"] for p in per),
        "n_admissible_total": sum(p["n_admissible"] for p in per),
        "certified_present": agg("n_certified_present"),
        "certified_absent": agg("n_certified_absent"),
        "exactly_robust_present": agg("n_exactly_robust_present"),
        "exactly_robust_absent": agg("n_exactly_robust_absent"),
        "false_certificates": agg("n_false_certificates"),
        "whole_structure_certified": {f"{r:g}": sum(p["by_radius"][f"{r:g}"]
                                                    ["whole_structure_certified"] for p in per)
                                      for r in G2_RADII},
        "slack_present": {f"{r:g}": (agg("n_certified_present")[f"{r:g}"]
                                     / max(1, agg("n_exactly_robust_present")[f"{r:g}"]))
                          for r in G2_RADII},
        "slack_absent": {f"{r:g}": (agg("n_certified_absent")[f"{r:g}"]
                                    / max(1, agg("n_exactly_robust_absent")[f"{r:g}"]))
                         for r in G2_RADII},
        "rho": {
            "n_pairs": sum(p["rho"]["n_pairs"] for p in per),
            "n_compared_against_enumeration":
                sum(p["rho"]["n_compared_against_enumeration"] for p in per),
            "max_overstatement_vs_enumeration_kcal":
                max(p["rho"]["overstatement_vs_enumeration_kcal"] for p in per),
            "capped": sum(p["rho"]["capped"] for p in per),
            "tied_at_zero": sum(p["rho"]["tied_at_zero"] for p in per),
            "unconverged": sum(p["rho"]["unconverged"] for p in per),
            "r_cap_kcal": per[0]["rho"]["r_cap_kcal"],
        },
        "power_check": {"weights_scaled_by": 0.5, "n_sequences": n_planted,
                        "false_certificates_produced": planted, "pass": planted > 0},
        "pair_tables_vs_enumeration": {
            "x_prime_mismatches": 0, "z_prime_mismatches": 0,
            "note": "X' from the forbidden-pair fold and Z' from the Zuker set are compared "
                    "against the enumerated minimum of E' on every sequence and every radius "
                    "here; a mismatch raises. This is G3's n <= 20 half, done where the "
                    "enumeration already exists.",
        },
        "folds": sum(p["folds"] for p in per),
        "pass": True,
        "seconds": round(time.time() - t0, 1),
        "provenance": git_provenance(R),
    }
    out.setdefault("gate", {})["G2"] = shard
    out["status"] = "G2 passed"
    write(out)
    print(json.dumps({k: v for k, v in shard.items() if k != "provenance"}, indent=1,
                     default=str))
    return 0

# ----------------------------------------------------------------------------------
# G3: the pair tables at realistic lengths
# ----------------------------------------------------------------------------------
def gate_g3() -> int:
    """G3: Z' from the Zuker set against an enforced fold, pair by pair, at n about 134.

    G2 already checks both pair tables against the enumerated minimum of E' on 200 sequences at
    three radii, which is this gate's n <= 20 half and is recorded in G2's shard. What that
    cannot reach is the length where the evaluation runs, and the claim being licensed is
    specific: Zuker's suboptimal fold returns, for every possible pair, an optimal structure
    containing it, so the minimum of E' over the returned structures that contain q is exactly
    Z'(q), and a pair in none of them has no legal structure containing it. Here the oracle is an
    enforced-pair fold, one per sampled pair, which answers the same question by a different
    route. Both halves of the claim are tested: the pairs the Zuker set covers must agree
    exactly, and the pairs it omits must be infeasible when enforced.
    """
    out = read_out()
    assert out.get("status") in ("G2 passed", "G3 passed"), out.get("status")
    if "predictions" in out:
        raise SystemExit("preregistered: the gate's shards are fixed")
    t0 = time.time()
    B = load_tr0_basis()
    P = template()
    lc = LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    rows, _ = tr0_rows()
    seqs = gate_sequences(rows, G3_SEQUENCES, 120, 150, SEED + 3)
    radii = {"r1": 1.0,
             "Andronescu2007": float(B["radii"]["Andronescu2007"]["radius_on_B_four_kcal"])}
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 3]))
    per, mismatch, infeasible_bad = [], [], []
    for _t, nm, seq, _db in seqs:
        import RNA

        from thermo.vienna_ref import assert_turner2004_params, reference_md

        plain = RNA.fold_compound(seq, reference_md())
        assert_turner2004_params(plain)
        star, _e = plain.mfe()
        adm = admissible_pairs(seq)
        F = Folder(lc, seq, star, Weights(B["a_four"], B["cols"], 1.0))
        rec = {"name": nm, "n_nt": len(seq), "n_admissible": len(adm), "by_radius": {}}
        for label, r in radii.items():
            F.set_radius(r)
            zmap = zprime_map(F, F.zuker())
            covered = [q for q in adm if q in zmap]
            omitted = [q for q in adm if q not in zmap]
            take_c = [covered[k] for k in rng.permutation(len(covered))[:G3_PAIRS]]
            take_o = [omitted[k] for k in rng.permutation(len(omitted))[:G3_PAIRS]]
            agree = 0
            for q in take_c:
                got = F.with_pair(*q)
                if got is None or got[1] != zmap[q]:
                    mismatch.append({"name": nm, "radius": label, "pair": q,
                                     "zuker": int(zmap[q]),
                                     "enforced": None if got is None else int(got[1])})
                else:
                    agree += 1
            for q in take_o:
                got = F.with_pair(*q)
                if got is not None:
                    infeasible_bad.append({"name": nm, "radius": label, "pair": q,
                                           "enforced": int(got[1])})
            rec["by_radius"][label] = {
                "radius_kcal": r, "n_covered_by_the_zuker_set": len(covered),
                "n_omitted": len(omitted), "n_sampled_covered": len(take_c),
                "n_sampled_omitted": len(take_o), "n_agree": agree,
            }
        rec["folds"] = F.folds
        per.append(rec)
    if mismatch or infeasible_bad:
        raise GuardFailure(f"G3: {len(mismatch)} Z' disagreements with an enforced fold and "
                           f"{len(infeasible_bad)} pairs the Zuker set omitted that are in fact "
                           f"feasible: {(mismatch + infeasible_bad)[:3]}")
    shard = {
        "n_sequences": len(per), "lengths": [p["n_nt"] for p in per],
        "radii_kcal": radii, "pairs_sampled_per_radius": G3_PAIRS,
        "n_sampled_covered": sum(v["n_sampled_covered"] for p in per
                                 for v in p["by_radius"].values()),
        "n_sampled_omitted": sum(v["n_sampled_omitted"] for p in per
                                 for v in p["by_radius"].values()),
        "n_zprime_disagreements": 0, "n_omitted_but_feasible": 0,
        "n_covered_by_the_zuker_set": {k: sum(p["by_radius"][k]["n_covered_by_the_zuker_set"]
                                              for p in per) for k in radii},
        "n_omitted": {k: sum(p["by_radius"][k]["n_omitted"] for p in per) for k in radii},
        "n_at_most_20_half": "in G2's shard: X' and Z' against the enumerated minimum of E'",
        "folds": sum(p["folds"] for p in per),
        "pass": True, "seconds": round(time.time() - t0, 1),
        "provenance": git_provenance(R),
    }
    out.setdefault("gate", {})["G3"] = shard
    out["status"] = "G3 passed"
    write(out)
    print(json.dumps({k: v for k, v in shard.items() if k != "provenance"}, indent=1,
                     default=str))
    return 0


# ----------------------------------------------------------------------------------
# G6: the gauge, and monotonicity in r
# ----------------------------------------------------------------------------------
def gauge_tables(cols: list[tuple], theta0: dict, norm_kcal: float) -> dict:
    """theta0 shifted along each analytic gauge direction, at about `norm_kcal` in norm.

    The shift is an exact multiple of 0.01 kcal/mol on every entry, because the directions are
    integer patterns (+1 on the mismatch tables, -1 or -2 on the loop initiations) and the step
    is an integer number of dcal. So the written table is the intended table exactly, with no
    rounding to argue about, and any change in a structure's energy is the gauge claim failing
    rather than the parameter file quantising.
    """
    from scripts.turner_nullspace import analytic_directions

    out = {}
    for name, g in analytic_directions(cols, 0, 0).items():
        scale = float(np.linalg.norm(g))
        step = max(1, int(round(norm_kcal * 100.0 / scale)))      # dcal per unit coefficient
        shifted = dict(theta0)
        touched = 0
        for k, c in zip(cols, g, strict=True):
            if c and k in shifted:
                shifted[k] = int(shifted[k] + step * int(round(c)))
                touched += 1
        out[name] = {
            "theta": shifted, "step_dcal": step, "n_entries_touched": touched,
            "norm_kcal": step * scale / 100.0,
            "max_entry_shift_dcal": int(step * max(abs(int(round(c))) for c in g if c)),
        }
    return out


def gate_g6() -> int:
    """G6: a gauge shift changes no label, and the certified sets shrink as r grows.

    The two halves fail differently. Monotonicity is a property of the certificate: w_L(r) is
    increasing in r, so the band can only widen and a certified pair can only stop being
    certified, never start. A violation would mean the weights are not what they are claimed to
    be. The gauge half is a property of the model: a direction that changes no structure's energy
    cannot change a label, and the only way the pipeline could disagree is if the count model
    were wrong about which entries a loop reads -- which is exactly what it would take for the
    paper's Type A claim to be wrong too. Both are run against written parameter files, so the
    shift goes through ViennaRNA rather than being asserted.
    """
    out = read_out()
    assert out.get("status") in ("G3 passed", "G6 passed"), out.get("status")
    if "predictions" in out:
        raise SystemExit("preregistered: the gate's shards are fixed")
    t0 = time.time()
    B = load_tr0_basis()
    P = template()
    lc = LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    rows, _ = tr0_rows()
    rad = B["radii"]
    ladder = sorted([1.0, float(rad["Langdon2018"]["radius_on_B_four_kcal"]),
                     float(rad["Andronescu2007"]["radius_on_B_four_kcal"]),
                     float(rad["Turner1999"]["radius_on_B_four_kcal"])])

    # --- monotonicity
    mono_seqs = gate_sequences(rows, G6_SEQUENCES, 50, 150, SEED + 6)
    viol, mono = [], []
    for _t, nm, seq, _db in mono_seqs:
        import RNA

        from thermo.vienna_ref import assert_turner2004_params, reference_md

        plain = RNA.fold_compound(seq, reference_md())
        assert_turner2004_params(plain)
        star, _e = plain.mfe()
        pt = pair_table(star)
        sp = [(i, pt[i]) for i in range(1, len(seq) + 1) if pt[i] > i]
        adm = admissible_pairs(seq)
        F = Folder(lc, seq, star, Weights(B["a_four"], B["cols"], ladder[0]))
        sets = []
        for r in ladder:
            F.set_radius(r)
            c = certify(F, sp, adm)
            sets.append(({p for p, m in c["margins_present_dcal"].items() if m > 0},
                         {q for q, m in c["margins_absent_dcal"].items()
                          if m is None or m > 0}))
        for a, b, r_a, r_b in zip(sets[:-1], sets[1:], ladder[:-1], ladder[1:], strict=True):
            for half, kind in ((0, "present"), (1, "absent")):
                gained = b[half] - a[half]
                if gained:
                    viol.append({"name": nm, "kind": kind, "from_kcal": r_a, "to_kcal": r_b,
                                 "n_gained": len(gained), "example": sorted(gained)[:3]})
        mono.append({"name": nm, "n_nt": len(seq),
                     "certified_present": [len(s[0]) for s in sets],
                     "certified_absent": [len(s[1]) for s in sets]})

    # --- the gauge
    from scripts.turner_par import loaded, params_match, write_par

    gauge = gauge_tables(B["cols"], P["theta"], G6_GAUGE_NORM)
    g_seqs = gate_sequences(rows, G6_GAUGE_SEQUENCES, 50, 150, SEED + 7)
    g_radii = [1.0, float(rad["Andronescu2007"]["radius_on_B_four_kcal"])]
    base = {}
    for _t, nm, seq, _db in g_seqs:
        import RNA

        from thermo.vienna_ref import assert_turner2004_params, reference_md

        plain = RNA.fold_compound(seq, reference_md())
        assert_turner2004_params(plain)
        star, _e = plain.mfe()
        pt = pair_table(star)
        sp = [(i, pt[i]) for i in range(1, len(seq) + 1) if pt[i] > i]
        adm = admissible_pairs(seq)
        F = Folder(lc, seq, star, Weights(B["a_four"], B["cols"], 1.0))
        labels = {}
        for r in g_radii:
            F.set_radius(r)
            c = certify(F, sp, adm)
            labels[f"{r:g}"] = ({p for p, m in c["margins_present_dcal"].items() if m > 0},
                                {q for q, m in c["margins_absent_dcal"].items()
                                 if m is None or m > 0})
        base[nm] = {"seq": seq, "star": star, "e0": F.e0_star, "pairs": sp, "adm": adm,
                    "labels": labels}

    g_out, g_viol = {}, []
    for name, rec in gauge.items():
        path = SHARDS / f"gauge_{name}.par"
        write_par(rec["theta"], path)
        changed_e0, changed_star, changed_labels = 0, 0, 0
        with loaded(path):
            import RNA

            from thermo.vienna_ref import reference_md

            for nm, b in base.items():
                fc = RNA.fold_compound(b["seq"], reference_md())
                bad = params_match(fc, rec["theta"], 0, 0)
                if bad:
                    raise GuardFailure(f"G6: the loaded table is not the written one: {bad}")
                e0 = int(round(fc.eval_structure(b["star"]) * 100))
                if e0 != b["e0"]:
                    changed_e0 += 1
                    g_viol.append({"direction": name, "name": nm, "kind": "E0 of s*",
                                   "before": b["e0"], "after": e0})
                star2, _e = fc.mfe()
                if star2 != b["star"]:
                    changed_star += 1
                    g_viol.append({"direction": name, "name": nm, "kind": "MFE structure"})
                    continue
                F = Folder(lc, b["seq"], b["star"], Weights(B["a_four"], B["cols"], 1.0),
                           pinned=False)
                for r in g_radii:
                    F.set_radius(r)
                    c = certify(F, b["pairs"], b["adm"])
                    now = ({p for p, m in c["margins_present_dcal"].items() if m > 0},
                           {q for q, m in c["margins_absent_dcal"].items()
                            if m is None or m > 0})
                    was = b["labels"][f"{r:g}"]
                    if now != was:
                        changed_labels += 1
                        g_viol.append({
                            "direction": name, "name": nm, "kind": "labels",
                            "radius_kcal": r,
                            "present_gained": len(now[0] - was[0]),
                            "present_lost": len(was[0] - now[0]),
                            "absent_gained": len(now[1] - was[1]),
                            "absent_lost": len(was[1] - now[1])})
        g_out[name] = {k: v for k, v in rec.items() if k != "theta"}
        g_out[name].update({"par_sha256": _sha(path), "n_sequences": len(base),
                            "n_with_a_changed_E0": changed_e0,
                            "n_with_a_changed_MFE_structure": changed_star,
                            "n_with_changed_labels": changed_labels})
    if viol or g_viol:
        raise GuardFailure(f"G6: {len(viol)} monotonicity violations and {len(g_viol)} gauge "
                           f"violations: {(viol + g_viol)[:3]}")
    shard = {
        "monotonicity": {"radii_kcal": ladder, "n_sequences": len(mono), "n_violations": 0,
                         "per_sequence": mono[:5],
                         "certified_present_totals": [sum(m["certified_present"][k]
                                                          for m in mono)
                                                      for k in range(len(ladder))],
                         "certified_absent_totals": [sum(m["certified_absent"][k] for m in mono)
                                                     for k in range(len(ladder))]},
        "gauge": {"target_norm_kcal": G6_GAUGE_NORM, "radii_kcal": g_radii,
                  "directions": g_out, "n_violations": 0},
        "pass": True, "seconds": round(time.time() - t0, 1),
        "provenance": git_provenance(R),
    }
    out.setdefault("gate", {})["G6"] = shard
    out["status"] = "G6 passed"
    write(out)
    print(json.dumps({k: v for k, v in shard.items() if k != "provenance"}, indent=1,
                     default=str))
    return 0

# ----------------------------------------------------------------------------------
# the TR0 pilot: the whole run path, on training data, before any registration
# ----------------------------------------------------------------------------------
def exposure(seq: str, W1: Weights, shapes) -> int:
    """Admissible interior loops, both closing pairs canonical, that read a reached entry.

    This is the COMPETITOR-side exposure, and it is the quantity that actually decides the
    labels. A pair of s* is uncertain whenever some structure LACKING it, within the
    certificate's allowance of the MFE, forms a loop the ball can move; whether such a structure
    exists depends on the competitors' loops, not on s*'s. Counting s*'s own weighted loops --
    the obvious thing to count -- answers a different question and would understate the exposure
    badly, because the certificate credits every competitor with -w_L for each moved loop it
    contains.
    """
    S = [0] + [BCODE[c] for c in seq] + [0]
    n = len(seq)
    cnt = 0
    for i in range(1, n + 1):
        for j in range(i + 4, n + 1):
            if not PT[S[i]][S[j]]:
                continue
            for u1, u2 in shapes:
                k, l = i + u1 + 1, j - u2 - 1      # noqa: E741
                if l - k < 4 or l > n or k < 1 or not PT[S[k]][S[l]]:
                    continue
                if W1.kcal(il_keys(S, i, j, k, l)) > 0:
                    cnt += 1
    return cnt


def pilot_one(lc: LoopCounts, B: dict, W1: Weights, shapes, name: str, seq: str,
              radii: dict, r_cap: float) -> dict:
    """One sequence of the pilot: labels at every radius, rho once, and the exposure."""
    import RNA

    from thermo.vienna_ref import assert_turner2004_params, reference_md

    t0 = time.time()
    plain = RNA.fold_compound(seq, reference_md())
    assert_turner2004_params(plain)
    star, e_mfe = plain.mfe()
    pt = pair_table(star)
    sp = [(i, pt[i]) for i in range(1, len(seq) + 1) if pt[i] > i]
    adm = admissible_pairs(seq)
    t_setup = time.time() - t0
    F = Folder(lc, seq, star, Weights(B["a_four"], B["cols"], 1.0))
    rec = {"name": name, "n_nt": len(seq), "e0_star_dcal": int(round(e_mfe * 100)),
           "n_pairs_star": len(sp), "n_admissible": len(adm), "by_radius": {}}
    for label, r in radii.items():
        t1 = time.time()
        F.set_radius(r)
        c = certify(F, sp, adm)
        n_out = c["n_pairs_admissible_outside_star"]
        rec["by_radius"][label] = {
            "radius_kcal": r,
            "n_certified_present": c["n_certified_present"],
            "n_uncertain_present": len(sp) - c["n_certified_present"],
            "n_certified_absent": c["n_certified_absent"],
            "n_uncertain_absent": n_out - c["n_certified_absent"],
            "n_absent_because_unreachable": c["n_absent_because_unreachable"],
            "whole_structure_certified": bool(c["whole_structure_certified"]),
            "seconds": round(time.time() - t1, 2),
        }
    t2 = time.time()
    rhos = [rho_of_pair(F, star_map := loop_coeff_map(lc, seq, star, W1), W1, i, j, r_cap)
            for i, j in sp] if sp else []
    fin = sorted(x["rho"] for x in rhos if not x["capped"] and not x["tied_at_zero"])
    rec["rho"] = {
        "n_pairs": len(rhos), "n_finite": len(fin),
        "capped": sum(1 for x in rhos if x["capped"]),
        "tied_at_zero": sum(1 for x in rhos if x["tied_at_zero"]),
        "unconverged": sum(1 for x in rhos if not x["converged"]),
        "quartiles_kcal": [float(np.percentile(fin, q)) for q in (0, 25, 50, 75, 100)]
        if fin else None,
        "values_kcal": [round(x, 4) for x in fin],
        "seconds": round(time.time() - t2, 2),
    }
    rec["n_weighted_loops_of_star"] = sum(1 for v in star_map.values() if v > 0)
    t3 = time.time()
    rec["exposure"] = {"n_reachable_admissible_interior_loops": exposure(seq, W1, shapes),
                       "shapes": sorted(list(s) for s in shapes),
                       "seconds": round(time.time() - t3, 2)}
    rec["folds"] = F.folds
    rec["callback_invocations"] = F.calls[0]
    rec["seconds"] = round(time.time() - t0, 2)
    rec["seconds_setup"] = round(t_setup, 2)
    return rec


def pilot_radii(B: dict) -> dict:
    rad = B["radii"]
    return {"r1": 1.0,
            "Langdon2018": float(rad["Langdon2018"]["radius_on_B_four_kcal"]),
            "Andronescu2007": float(rad["Andronescu2007"]["radius_on_B_four_kcal"]),
            "Turner1999": float(rad["Turner1999"]["radius_on_B_four_kcal"])}


def pilot_sequences(rows) -> list[tuple[str, str]]:
    """300 TR0 sequences, stratified over the sorted length pool by a recorded seed.

    Equal-count bins, not equal-width: the draw then has TR0's own length distribution, so the
    label fractions it reports are the fractions a TR0-like set gives rather than the fractions
    of an artificially long set. The cost is that the long tail is thinly covered, which is why
    the runtime is reported against length rather than averaged.
    """
    picked = gate_sequences(rows, PILOT_SEQUENCES, 1, 10 ** 6, SEED + 10)
    return [(nm, seq) for _t, nm, seq, _db in picked]


def pilot_worker() -> int:
    w = int(sys.argv[sys.argv.index("--shard") + 1])
    n_w = int(sys.argv[sys.argv.index("--of") + 1])
    B = load_tr0_basis()
    P = template()
    lc = LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    W1 = Weights(B["a_four"], B["cols"], 1.0)
    shapes = W1.shapes
    radii = pilot_radii(B)
    r_cap = float(B["radii"]["r_cap_kcal"])
    rows, _ = tr0_rows()
    seqs = pilot_sequences(rows)
    dst = SHARDS / f"pilot_{w}.jsonl"
    with open(dst, "w") as fh:
        for k, (name, seq) in enumerate(seqs):
            if k % n_w != w:
                continue
            rec = pilot_one(lc, B, W1, shapes, name, seq, radii, r_cap)
            rec["index"] = k
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
    return 0


def pilot() -> int:
    """The pilot, on TR0 only, before any registration. NOT a registered test.

    The authors' instruction of 2026-10-13: the expectation PRED states about the label
    fractions must be written from measured numbers rather than from an argument, because the
    argument I gave was wrong. Nothing here touches TS0 or the held-out families, and the
    numbers are labelled a pilot wherever they appear.

    It is also the smoke test of the run path: the same sharded driver, the same worker, the same
    JSON-lines-per-sequence discipline the TS0 run will use, so a fault in that machinery shows
    up here where no registration depends on it.
    """
    out = read_out()
    assert out.get("status") in ("G6 passed", "G4 passed", "pilot done"), out.get("status")
    if "predictions" in out:
        raise SystemExit("preregistered: the pilot belongs before the registration")
    # provenance at the START: the pilot runs for hours, and what it must record is the code
    # it ran, not whatever HEAD is when it finishes. A dirty tree is refused for the same reason.
    prov = git_provenance(R)
    if prov.get("git_dirty_excluding_runs"):
        raise SystemExit(f"commit first: {prov.get('git_dirty_paths_outside_runs')}")
    t0 = time.time()
    B = load_tr0_basis()
    rows, _ = tr0_rows()
    seqs = pilot_sequences(rows)
    SHARDS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    procs = []
    for w in range(PILOT_WORKERS):
        procs.append(subprocess.Popen(
            ["nice", "-n", "10", sys.executable, str(R / "scripts" / "certfold.py"),
             "--phase", "pilot-worker", "--shard", str(w), "--of", str(PILOT_WORKERS)],
            cwd=R, env=env, stdout=(SHARDS / f"pilot_{w}.log").open("w"),
            stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise GuardFailure(f"pilot workers exited {codes}; shards kept at {SHARDS}")

    per = []
    for w in range(PILOT_WORKERS):
        with open(SHARDS / f"pilot_{w}.jsonl") as fh:
            per += [json.loads(line) for line in fh if line.strip()]
    if len(per) != len(seqs):
        raise GuardFailure(f"the pilot ran {len(per)} of {len(seqs)} sequences")
    radii = pilot_radii(B)

    def frac(label, num, den):
        a = sum(p["by_radius"][label][num] for p in per)
        b = sum(p["by_radius"][label][den] for p in per)
        return {"n": a, "of": b, "fraction": (a / b) if b else None}

    by_radius = {}
    for label in radii:
        pres_den = sum(p["n_pairs_star"] for p in per)
        abs_den = sum(p["by_radius"][label]["n_certified_absent"]
                      + p["by_radius"][label]["n_uncertain_absent"] for p in per)
        by_radius[label] = {
            "radius_kcal": radii[label],
            "certified_present": {
                "n": sum(p["by_radius"][label]["n_certified_present"] for p in per),
                "of": pres_den,
                "fraction": sum(p["by_radius"][label]["n_certified_present"]
                                for p in per) / pres_den if pres_den else None},
            "certified_absent": {
                "n": sum(p["by_radius"][label]["n_certified_absent"] for p in per),
                "of": abs_den,
                "fraction": sum(p["by_radius"][label]["n_certified_absent"]
                                for p in per) / abs_den if abs_den else None},
            "uncertain_present_fraction": (
                sum(p["by_radius"][label]["n_uncertain_present"] for p in per) / pres_den
                if pres_den else None),
            "absent_because_unreachable": sum(
                p["by_radius"][label]["n_absent_because_unreachable"] for p in per),
            "whole_structure_certified": {
                "n": sum(p["by_radius"][label]["whole_structure_certified"] for p in per),
                "of": len(per),
                "fraction": sum(p["by_radius"][label]["whole_structure_certified"]
                                for p in per) / len(per)},
            "mean_per_sequence_certified_present_fraction": float(np.mean(
                [p["by_radius"][label]["n_certified_present"] / p["n_pairs_star"]
                 for p in per if p["n_pairs_star"]])),
        }

    allrho = [v for p in per for v in p["rho"]["values_kcal"]]
    expo = np.array([p["exposure"]["n_reachable_admissible_interior_loops"] for p in per])
    secs = np.array([p["seconds"] for p in per])
    lens = np.array([p["n_nt"] for p in per])
    # the exposure is the thing the labels should track, so it is reported against them rather
    # than on its own: the uncertain-present fraction by exposure quartile, at the Andronescu
    # radius, which is the one the paper quotes
    q = np.quantile(expo, [0.25, 0.5, 0.75])
    bins = np.digitize(expo, q)
    unc = np.array([(p["by_radius"]["Andronescu2007"]["n_uncertain_present"]
                     / p["n_pairs_star"]) if p["n_pairs_star"] else np.nan for p in per])
    by_quartile = [{"quartile": int(b),
                    "n_sequences": int((bins == b).sum()),
                    "exposure_median": float(np.median(expo[bins == b])),
                    "uncertain_present_fraction_mean":
                        float(np.nanmean(unc[bins == b])) if (bins == b).any() else None}
                   for b in range(4)]

    shard = {
        "label": "PILOT on TR0 training data, not a registered test",
        "n_sequences": len(per), "seed": SEED + 10, "workers": PILOT_WORKERS,
        "lengths": {"min": int(lens.min()), "p25": int(np.percentile(lens, 25)),
                    "median": int(np.median(lens)), "p75": int(np.percentile(lens, 75)),
                    "p95": int(np.percentile(lens, 95)), "max": int(lens.max())},
        "radii_kcal": radii,
        "by_radius": by_radius,
        "rho": {
            "n_pairs": sum(p["rho"]["n_pairs"] for p in per),
            "n_finite": len(allrho),
            "capped": sum(p["rho"]["capped"] for p in per),
            "tied_at_zero": sum(p["rho"]["tied_at_zero"] for p in per),
            "unconverged": sum(p["rho"]["unconverged"] for p in per),
            "r_cap_kcal": float(B["radii"]["r_cap_kcal"]),
            "quartiles_kcal": [float(np.percentile(allrho, x)) for x in (0, 25, 50, 75, 100)]
            if allrho else None,
        },
        "exposure": {
            "shapes": per[0]["exposure"]["shapes"],
            "n_reachable_admissible_interior_loops": {
                "min": int(expo.min()), "median": float(np.median(expo)),
                "p95": float(np.percentile(expo, 95)), "max": int(expo.max()),
                "n_sequences_with_none": int((expo == 0).sum())},
            "weighted_loops_of_star": {
                "median": float(np.median([p["n_weighted_loops_of_star"] for p in per])),
                "max": int(max(p["n_weighted_loops_of_star"] for p in per)),
                "n_sequences_with_none": int(sum(1 for p in per
                                                 if not p["n_weighted_loops_of_star"]))},
            "uncertain_present_by_exposure_quartile_at_Andronescu": by_quartile,
        },
        "runtime": {
            "seconds_per_sequence": {
                "median": float(np.median(secs)), "p95": float(np.percentile(secs, 95)),
                "max": float(secs.max()), "total_cpu_seconds": float(secs.sum())},
            "slowest": sorted(({"name": p["name"], "n_nt": p["n_nt"],
                                "seconds": p["seconds"], "folds": p["folds"]} for p in per),
                              key=lambda x: -x["seconds"])[:5],
            "folds_total": sum(p["folds"] for p in per),
            "callback_invocations_total": sum(p["callback_invocations"] for p in per),
            "wall_seconds": round(time.time() - t0, 1),
        },
        "files_sha256": {f"pilot_{w}.jsonl": _sha(SHARDS / f"pilot_{w}.jsonl")
                         for w in range(PILOT_WORKERS)},
        "provenance": prov,
    }
    out = read_out()          # re-read: hours have passed since the start
    out["pilot"] = shard
    out["status"] = "pilot done"
    write(out)
    print(json.dumps({k: v for k, v in shard.items()
                      if k not in ("provenance", "files_sha256")}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------------
# G4: the certificate's own band, tested exactly, at realistic lengths
# ----------------------------------------------------------------------------------
def gate_g4() -> int:
    """G4: every structure E' admits within reach of E'(s*), tested exactly, on TR0 at n <= 150.

    The band is exactly the set that can break a certificate: if no structure with
    E'(s) <= E'(s*) witnesses against a label, none does, because E' is a lower bound on the gap
    for every theta in the ball. So this is G2's test made lazy -- the structures are drawn from
    ViennaRNA's Wuchty enumeration under E' instead of from a complete enumeration -- which is
    what makes it affordable at the lengths the evaluation runs at.

    It runs on TR0, not TS0 (the authors' amendment of 2026-10-13): no test-set sequence is
    folded before PRED is committed, and TR0 is the basis's own reference set, so there is
    nothing to leak. Its role, soundness at realistic lengths, is unchanged.

    THE GAP IT CANNOT CLOSE, stated because it matters. The band is computed FROM the callback,
    so a callback that charges a loop too little shrinks the band and hides the very structures
    that would break the certificate: the check would pass vacuously. That gap is closed in G1,
    whose constructed cases walk every vrna_E_internal case with both signs, and in G2, whose
    structure space is built by code that shares nothing with ViennaRNA. G4 adds length, not
    independence.

    The band grows with r, so the radii are taken in order and a sequence whose band would
    exceed the recorded cap is SKIPPED rather than truncated, with the skips reported per radius.
    A truncated band is a silent gap; a skip is a number.
    """
    out = read_out()
    assert out.get("status") in ("G6 passed", "pilot done", "G4 passed"), out.get("status")
    if "predictions" in out:
        raise SystemExit("preregistered: the gate's shards are fixed")
    t0 = time.time()
    B = load_tr0_basis()
    P = template()
    lc = LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    rows, _ = tr0_rows()
    pos = {B["cols"][e]: k for k, e in enumerate(B["used"])}
    seqs = gate_sequences(rows, G4_SEQUENCES, 30, G4_MAX_N, SEED + 4)

    import RNA

    from thermo.vienna_ref import assert_turner2004_params, reference_md

    per, viol = [], []
    for _t, nm, seq, _db in seqs:
        plain = RNA.fold_compound(seq, reference_md())
        assert_turner2004_params(plain)
        star, _e = plain.mfe()
        pt = pair_table(star)
        sp = [(i, pt[i]) for i in range(1, len(seq) + 1) if pt[i] > i]
        adm = admissible_pairs(seq)
        u_star = basis_image(B["B4"], pos, lc.counts(seq, star))
        F = Folder(lc, seq, star, Weights(B["a_four"], B["cols"], G2_RADII[0]))
        rec = {"name": nm, "n_nt": len(seq), "n_pairs_star": len(sp), "by_radius": {}}
        for r in G2_RADII:
            F.set_radius(r)
            _s_min, e_min, _e0 = F.free()
            delta = F.eprime_star - e_min
            if delta < 0:
                raise GuardFailure(f"E'(s*) is below the constrained minimum on {nm}")
            if delta > G4_DELTA_CAP:
                rec["by_radius"][f"{r:g}"] = {"radius_kcal": r, "skipped": True,
                                              "band_delta_dcal": int(delta),
                                              "cap_dcal": G4_DELTA_CAP}
                continue
            band = [z.structure for z in F.fc.subopt(int(delta))]
            if not band:
                raise GuardFailure(f"an empty band on {nm} at r = {r} with delta = {delta}")
            # subopt must be honouring the callback: its best structure must reach E'_min
            best = min(F.eprime(s)[0] for s in band)
            if best != e_min:
                raise GuardFailure(f"subopt under E' does not reach the constrained minimum on "
                                   f"{nm}: {best} != {e_min}")
            c = certify(F, sp, adm)
            cert_p = {p for p, m in c["margins_present_dcal"].items() if m > 0}
            cert_a = {q for q, m in c["margins_absent_dcal"].items() if m is None or m > 0}
            n_witness = 0
            for s in band:
                ep, e0 = F.eprime(s)
                if ep > F.eprime_star:
                    continue
                n_s = (e0 - F.e0_star) / 100.0
                u = basis_image(B["B4"], pos, lc.counts(seq, s))
                margin = n_s - r * float(np.linalg.norm(u - u_star))
                if margin > 0:
                    continue
                n_witness += 1
                here = set()
                p_t = pair_table(s)
                for i in range(1, len(seq) + 1):
                    if p_t[i] > i:
                        here.add((i, p_t[i]))
                for p in cert_p - here:
                    viol.append({"name": nm, "radius_kcal": r, "kind": "present", "pair": p,
                                 "exact_margin_kcal": margin, "structure": s[:60]})
                for q in cert_a & here:
                    viol.append({"name": nm, "radius_kcal": r, "kind": "absent", "pair": q,
                                 "exact_margin_kcal": margin, "structure": s[:60]})
            rec["by_radius"][f"{r:g}"] = {
                "radius_kcal": r, "skipped": False, "band_delta_dcal": int(delta),
                "band_size": len(band), "n_exact_witnesses": n_witness,
                "n_certified_present": len(cert_p), "n_certified_absent": len(cert_a),
            }
        rec["folds"] = F.folds
        per.append(rec)
        if viol:
            raise GuardFailure(f"G4: {len(viol)} false certificates: {viol[:2]}")

    def at(r, key, pred=lambda v: True):
        return [v[key] for p in per for k, v in p["by_radius"].items()
                if k == f"{r:g}" and key in v and pred(v)]

    shard = {
        "n_sequences": len(per), "max_n": G4_MAX_N, "radii_kcal": list(G2_RADII),
        "band_cap_dcal": G4_DELTA_CAP,
        "lengths": {"min": min(p["n_nt"] for p in per), "max": max(p["n_nt"] for p in per)},
        "by_radius": {f"{r:g}": {
            "n_run": len(at(r, "band_size")),
            "n_skipped": sum(1 for p in per
                             if p["by_radius"].get(f"{r:g}", {}).get("skipped")),
            "band_size_total": sum(at(r, "band_size")),
            "band_size_max": max(at(r, "band_size"), default=0),
            "band_delta_dcal_max": max(at(r, "band_delta_dcal"), default=0),
            "n_exact_witnesses": sum(at(r, "n_exact_witnesses")),
            "n_certified_present": sum(at(r, "n_certified_present")),
            "n_certified_absent": sum(at(r, "n_certified_absent")),
        } for r in G2_RADII},
        "n_false_certificates": 0,
        "folds": sum(p["folds"] for p in per),
        "the_gap_this_cannot_close": "the band comes from the callback, so an under-charging "
                                     "callback would shrink it and pass vacuously; G1's "
                                     "constructed cases and G2's independent enumeration are "
                                     "what close that",
        "pass": True, "seconds": round(time.time() - t0, 1),
        "provenance": git_provenance(R),
    }
    out.setdefault("gate", {})["G4"] = shard
    out["status"] = "G4 passed"
    write(out)
    print(json.dumps({k: v for k, v in shard.items() if k != "provenance"}, indent=1,
                     default=str))
    return 0

if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"build-basis": build_basis, "gate-g1": gate_g1,
              "gate-g2": gate_g2, "gate-g3": gate_g3,
              "gate-g6": gate_g6, "gate-g4": gate_g4,
              "pilot": pilot, "pilot-worker": pilot_worker}[ph]())
