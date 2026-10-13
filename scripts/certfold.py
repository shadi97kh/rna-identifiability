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

So the set is DEFINED to be the ball in B_four, the projection of B_TR0 onto the four tables'
coordinates, re-orthonormalised. Then returning 0 outside PAIR_IL is exact rather than
approximate, and the only thing owed to the reader is how far B_four sits from B_TR0, which this
phase measures and records (expected about 1e-10 in Frobenius norm, against the 0.005 kcal/mol
rounding the parameter file already imposes). Every radius, every published component and every
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
R_CAP_MULTIPLE = 4               # r_cap = 4 x the largest published radius on B_four


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
    off = NT[~mask]                                   # the rows the mask drops
    Q, s_, _ = np.linalg.svd(NT * mask[:, None], full_matrices=False)
    keep = s_ > 1e-8
    B4 = Q[:, keep]
    # the whole point of the projection is that a callback returning 0 outside PAIR_IL is EXACT
    # for this basis, so the rows outside the four tables are set to exactly zero rather than
    # left at the SVD's 1e-15: LAPACK has no reason to deliver them as zeros and did not
    B4[~mask] = 0.0
    diag = {
        "n_used": int(len(used)),
        "n_used_in_four_tables": int(mask.sum()),
        "n_used_outside_four_tables": int((~mask).sum()),
        "dim_B_TR0": int(NT.shape[1]),
        "dim_B_four": int(B4.shape[1]),
        "dropped_frobenius": float(np.linalg.norm(off)),
        "dropped_max_row_norm": float(np.linalg.norm(off, axis=1).max()) if off.size else 0.0,
        "singular_values_dropped": [float(x) for x in s_[~keep][:8]],
        "weight_outside_four_tables": float((off ** 2).sum() / (NT ** 2).sum()),
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
        "reverse_residual_B_TR0_in_tr0ts0": {
            "max": float(back.max()), "min": float(back.min()),
            "n_outside_at_1e_8": int((back > 1e-8).sum()),
            "note": "a TR0 direction outside the TR0+TS0 null space is one TS0's references "
                    "pinned; the count is the price of removing the leakage",
        },
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

    files = {}
    np.savez_compressed(SHARDS / "basis_tr0.npz", cols=np.array([repr(k) for k in cols]),
                        used=used, NT=NT, B4=B4, a_tr0=a_tr0, a_four=a_four)
    files["basis_tr0.npz"] = _sha(SHARDS / "basis_tr0.npz")

    out.update({
        "status": "basis built",
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
            "must_hold": must,
            "basis_quality": quality,
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
                      "principal_angles": angles}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"build-basis": build_basis}[ph]())
