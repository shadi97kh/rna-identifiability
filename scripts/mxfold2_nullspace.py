#!/usr/bin/env python3
"""Does MXfold2's output decomposition have a null space? Turner-gauge plan, step 2.

Read-only analysis: decomposition algebra on MXfold2's published source plus a
reading of its objective. Nothing is trained and MXfold2 is not executed.

SOURCE READ

MXfold2 (Sato, Akiyama, Sakakibara, Nat Commun 2021), github.com/mxfold/mxfold2 at
commit 51b2136 (2023-04-13, MIT licence): `mxfold2/fold/zuker.py`, `fold/mix.py`,
`fold/rnafold.py`, `loss.py`, `train.py`, `src/param/positional.cpp`,
`src/fold/zuker.cpp`, `models/TrainSetAB.conf`. The shipped pretrained model is
`--model MixC`: a TRAINABLE Turner 2004 table (every table an nn.Parameter,
initialised from Turner 2004) PLUS a ZukerFold of model_type 'C', whose network
emits position-indexed score matrices. Per loop, MixC's score is the sum of the two.

THE COUNTING RULE (positional part; i<j is a pair, (k,l) its inner pair)

The network emits N x N matrices. Upper triangle (i,j), i<j, is a pair seen from
INSIDE (the loop it closes); lower triangle (j,i) is the same pair seen from OUTSIDE.
  S   helix stacking       score_helix_stacking
  M   mismatch             score_mismatch_{hairpin,internal,multi,external}; ONE
                           shared channel in types 'C' and 'M' (separate in 'L')
  U   unpaired             'C': one value per interval U(a,b) (channel 2 of the
                           paired output); 'M': per nucleotide u(k), summed over
                           the interval by `unpair_interval`
  plus position-independent length tables: hairpin_length[31], bulge_length[31],
  internal_length[31], internal_explicit[5][5], internal_symmetry[16],
  internal_asymmetry[29], helix_length[31]. score_basepair is identically zero in
  'C' and 'M'.

From positional.cpp and zuker.cpp:
  helix of m stacked pairs from (i,j)   sum_{k=1}^{m-1} [S(i+k-1, j-k+1) + S(j-k, i+k)]
                                        + helix_length[min(m,30)]
     (the DP's C(i,j) = N(i+m-1, j-m+1) + score_helix(i,j,m) and N excludes stacks,
      so m is the full stack run: the parse is unique)
  hairpin closed by (i,j)               hairpin_length[l] + U(i+1,j-1) + M(i,j)
  bulge (i,j | k,l)                     bulge_length[ll] + U(sides) + M(i,j) + M(l,k)
  internal loop (i,j | k,l)             internal_length[ls+ll] + U(i+1,k-1) + U(l+1,j-1)
                                        + internal_explicit[min(ls,4)][min(ll,4)]
                                        + internal_symmetry[ll] (if ls==ll)
                                        + internal_asymmetry[ll-ls]
                                        + M(i,j) + M(l,k)
  multiloop closed by (i,j)             M(i,j) + per branch (u,v): M(v,u)
                                        + unpaired: the run before the first branch as
                                        ONE interval U(i+1,u-1), every later run one
                                        nucleotide at a time U(k,k) (M_UNPAIRED /
                                        M1_UNPAIRED recursions)
  exterior loop                         per stem (i,j): M(j,i); per unpaired k: U(k,k)
                                        (F_UNPAIRED is single-nucleotide)
So every pair (i,j) takes exactly ONE inside term -- S(i,j) if the next pair inward
stacks on it, else M(i,j) -- and exactly ONE outside term -- S(j,i) if the next pair
outward stacks on it, else M(j,i). The exterior stem has a per-pair term, M(j,i).

PRE-REGISTERED PREDICTIONS (written into metrics.json before computing)

Q1. Pair gauge: for every pair (i,j) that occurs in some structure, the direction
    +c on {S(i,j), M(i,j)} and -c on {S(j,i), M(j,i)} is null. This is exactly the
    direction step 1 found NOT null in Turner (A4), and it survives here because the
    exterior stem has a per-pair term. Dimension grows like the number of candidate
    pairs, i.e. O(N^2) per sequence, against 2-3 for the whole Turner model.
Q2. The duplex-paper gauge transfers: +c on every S entry and -2c(m-1) on
    helix_length[m] is null, because MXfold2 has a per-helix term Turner lacks
    (contrast step 1 P2, where the uniform stack shift is not null).
Q3. The plan's candidate "helix opening vs helix stacking" trade, taken literally as
    +c on S(i,j) and -c on M(i,j) for one pair, is NOT null: a structure takes one or
    the other, never both. The null trade is inside-vs-outside (Q1), not
    opening-vs-stacking.
Q4. The thermodynamic regulariser is gauge-INVARIANT. `StructuredLossWithTurner`
    (loss.py; train.py `--loss-func hinge_mix`) adds
        sl_weight * (ref - ref2)^2 / L,
    with ref = the model's total score of the REFERENCE structure and ref2 = Turner's
    total for the same structure. It depends on the parameters only through one
    structure's total, and every null direction leaves every structure's total
    unchanged, so two gauge-equivalent score sets give IDENTICAL regulariser values:
    the difference is exactly zero by construction, on any example. The plan's
    premise ("pulls scores toward Turner values in a gauge-dependent norm") is
    refuted for MXfold2 as published. The only other penalty, l1_weight * sum|w|, is
    on network WEIGHTS, not on scores, and does not reference Turner.
Q5. What the regulariser DOES constrain is a different object: directions that move
    every structure of a sequence by the same constant (e.g. +c on every U(k,k) and
    every unpaired interval scaled by its length, with the matching per-pair terms)
    are invisible to the structured hinge loss (which compares structures of one
    sequence) but not to ref - ref2. That is the step-3 phenomenon (a direction
    flat in L_struct and pinned by L_thermo), not a gauge choice.

CORRECTION TO Q4, 2026-10-07 (from the user; Q4's pre-registered text is left as it
was written). "Gauge-invariant" in Q4 holds only for the EXACT null space G_x: the
directions that leave every structure's total unchanged. The structured hinge loss is
blind to more than that. For each sequence it is blind to G_x plus ONE further
direction, the constant shift c_x, which moves every structure's total by the same
amount N (here: +(b-a+1) on every unpaired interval U(a,b), or +1 on every u(k) in
type M, and +1 on every S and M entry, inside and outside, so +1 per unpaired
nucleotide and +2 per pair). Any direction that moves all structures equally is c_x
times a constant plus an element of G_x. So, relative to the hinge loss's full
invariance, the thermodynamic regulariser is NOT gauge-invariant in general: it sees
c_x, because c_x changes the reference structure's total, and it is blind to every
direction of G_x (the per-pair trades and the stack/helix-length trade). It pins
exactly one direction per sequence and leaves the rest free. Check Q6 below confirms
numerically that c_x moves every enumerated structure by exactly N. Prediction W2 of
the seed study (scripts/mxfold2_seed_study.py) tests the consequence directly.

NEGATIVE CONTROLS
C1. A random direction with null component removed changes some structure's total.
C2. Every numerical null basis vector leaves every structure unchanged to 1e-9.
C3. Within-row label shuffle changes the null dimension.
C4. Planted defect: an extractor that charges the exterior stem nothing (Turner-like)
    must make the Q1 pair gauge stop being null for pairs that can be exterior stems.

SEQUENCES

Per-sequence null spaces (the columns are position-indexed, so they do not align
across sequences). Every secondary structure of each sequence is enumerated
exhaustively (ViennaRNA subopt with an unbounded energy window, canonical pairs,
hairpins of at least 3, lonely pairs allowed), so each per-sequence answer is exact
for the structure space, not a sample. The sequences are short constructions taken
from step 1's generating set (scripts/turner_nullspace.py), not data; ArchiveII is
BLOCKED (docs/BLOCKED.md item 27). Exhaustive enumeration caps the length.

Run: .venv/bin/python scripts/mxfold2_nullspace.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import RNA  # noqa: E402

from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_nullspace import (  # noqa: E402
    build_matrix,
    dump_params,
    generating_set,
    null_space,
    pair_table,
    rank_mod_p,
    theta_from_par,
)
from thermo.vienna_ref import reference_md  # noqa: E402

RUN_ID = "2026-10-07_mxfold2_nullspace"
OUT = R / "runs" / RUN_ID
TOL = 1e-9
SEED = 20261007
MXFOLD2_COMMIT = "51b213676708bebd664f0c40873a46e09353e1ee"

# the first step-1 generating-set construction of each tag that is short enough to
# enumerate every structure exhaustively
SEQUENCE_TAGS = ("hairpin", "int11", "ext_two", "il_size", "ml_closing", "ml_branch")
MAX_LEN = 31


def pick_sequences() -> dict[str, str]:
    rows = generating_set(theta_from_par(dump_params())["special"])
    out = {}
    for tag in SEQUENCE_TAGS:
        seq = next(s for t, s, _ in rows if t == tag and len(s) <= MAX_LEN)
        out[tag] = seq
    return out

PRED = {
    "Q1": "pair gauge (+inside, -outside, per occurring pair) is null; dim grows ~ #pairs",
    "Q2": "uniform S shift absorbed by helix_length[m] -2c(m-1) is null (duplex gauge transfers)",
    "Q3": "literal opening-vs-stacking trade (+S(i,j), -M(i,j)) is NOT null",
    "Q4": "thermodynamic regulariser sl*(ref-ref2)^2/L is exactly gauge-invariant",
    "Q5": "regulariser pins per-sequence constant shifts the hinge loss cannot see (step-3 "
          "object), it does not choose a gauge",
}


def enumerate_structures(seq: str) -> list[str]:
    md = reference_md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(seq, md)
    subs = fc.subopt(100_000)  # 1000 kcal/mol window: every structure
    out = sorted({s.structure for s in subs})
    assert "." * len(seq) in out
    return out


class MX:
    """Count vector of MXfold2's positional score (model_type 'C' or 'M')."""

    def __init__(self, model_type: str, exterior_stem_term: bool = True):
        assert model_type in ("C", "M")
        self.t = model_type
        self.ext = exterior_stem_term

    def counts(self, seq: str, db: str) -> dict:
        n = len(seq)
        pt = pair_table(db)
        c: dict[tuple, int] = {}

        def add(k, v=1):
            c[k] = c.get(k, 0) + v

        def unpaired(a, b):  # interval a..b inclusive, 1-based
            if b < a:
                return
            if self.t == "C":
                add(("U", a, b))
            else:
                for k in range(a, b + 1):
                    add(("u", k))

        # exterior
        k = 1
        while k <= n:
            if pt[k] > k:
                if self.ext:
                    add(("M", pt[k], k))
                k = pt[k] + 1
            else:
                unpaired(k, k)
                k += 1

        # helices: maximal stack runs, each from its outermost pair
        for i in range(1, n + 1):
            j = pt[i]
            if j <= i:
                continue
            if i > 1 and j < n and pt[i - 1] == j + 1:
                continue  # not the outermost pair of its helix
            m = 1
            while pt[i + m] == j - m and i + m < j - m:
                m += 1
            for kk in range(1, m):
                add(("S", i + kk - 1, j - kk + 1))
                add(("S", j - kk, i + kk))
            add(("helix_length", min(m, 30)))
            # the loop closed by the innermost pair
            a, b = i + m - 1, j - m + 1
            self._loop(a, b, pt, add, unpaired)
        return c

    def _loop(self, i, j, pt, add, unpaired):
        inner = []
        p = i + 1
        while p < j:
            if pt[p] > p:
                inner.append((p, pt[p]))
                p = pt[p] + 1
            else:
                p += 1
        add(("M", i, j))
        if not inner:
            add(("hairpin_length", min(j - i - 1, 30)))
            unpaired(i + 1, j - 1)
            return
        if len(inner) == 1:
            p, q = inner[0]
            l1, l2 = p - i - 1, j - q - 1
            ls, ll = min(l1, l2), max(l1, l2)
            assert ll > 0, "a stack inside a loop position means the helix walk is wrong"
            add(("M", q, p))
            unpaired(i + 1, p - 1)
            unpaired(q + 1, j - 1)
            if ls == 0:
                add(("bulge_length", min(ll, 30)))
            else:
                add(("internal_length", min(ls + ll, 30)))
                add(("internal_explicit", min(ls, 4), min(ll, 4)))
                if ls == ll:
                    add(("internal_symmetry", min(ll, 15)))
                add(("internal_asymmetry", min(ll - ls, 28)))
            return
        # multiloop
        first = inner[0][0]
        unpaired(i + 1, first - 1)  # leading run as one interval
        for idx, (u, v) in enumerate(inner):
            add(("M", v, u))
            nxt = inner[idx + 1][0] if idx + 1 < len(inner) else j
            for kk in range(v + 1, nxt):
                unpaired(kk, kk)  # later runs one nucleotide at a time


def analytic(cols, pairs_seen, model_type):
    idx = {k: n for n, k in enumerate(cols)}
    out = {}
    for (i, j) in sorted(pairs_seen):
        g = np.zeros(len(cols))
        for k, x in ((("S", i, j), 1), (("M", i, j), 1), (("S", j, i), -1), (("M", j, i), -1)):
            if k in idx:
                g[idx[k]] = x
        out[f"pair_gauge_{i}_{j}"] = g
    g = np.zeros(len(cols))
    for k, n in idx.items():
        if k[0] == "S":
            g[n] = 1.0
        if k[0] == "helix_length":
            g[n] = -2.0 * (k[1] - 1)
    out["stack_vs_helix_length"] = g
    return out


def analyse(name, seq, model_type, rng):
    structs = enumerate_structures(seq)
    ex = MX(model_type)
    rows = [ex.counts(seq, s) for s in structs]

    # C0: MXfold2 is not executed here, so the counting rule is checked against its
    # own invariants on every structure: one inside and one outside term per pair,
    # every unpaired nucleotide covered exactly once, one helix_length per maximal helix
    c0_fail = 0
    for s, r in zip(structs, rows, strict=True):
        pt = pair_table(s)
        npairs = sum(1 for i in range(1, len(s) + 1) if pt[i] > i)
        inside = sum(v for k, v in r.items() if k[0] in ("S", "M") and k[1] < k[2])
        outside = sum(v for k, v in r.items() if k[0] in ("S", "M") and k[1] > k[2])
        if model_type == "C":
            cover = sum(v * (k[2] - k[1] + 1) for k, v in r.items() if k[0] == "U")
        else:
            cover = sum(v for k, v in r.items() if k[0] == "u")
        nhel = sum(1 for i in range(1, len(s) + 1) if pt[i] > i
                   and not (i > 1 and pt[i] < len(s) and pt[i - 1] == pt[i] + 1))
        hl = sum(v for k, v in r.items() if k[0] == "helix_length")
        if (inside, outside, cover, hl) != (npairs, npairs, s.count("."), nhel):
            c0_fail += 1
    cols = sorted({k for r in rows for k in r}, key=repr)
    M = build_matrix(rows, cols)
    N, info, red = null_space(M)
    dense = red.toarray()
    rk = {str(p): rank_mod_p(dense, p) for p in (2_147_483_647, 1_000_000_007)}
    exact_null = dense.shape[1] - max(rk.values())
    pairs_seen = {(i, pair_table(s)[i]) for s in structs for i in range(1, len(seq) + 1)
                  if pair_table(s)[i] > i}

    an = analytic(cols, pairs_seen, model_type)
    an_null = {k: float(np.abs(M @ g).max()) for k, g in an.items()}
    A = np.stack(list(an.values()), axis=1)
    rank_A = int(np.linalg.matrix_rank(A))
    # how much of the numerical null space the analytic directions explain
    QA, _ = np.linalg.qr(A)
    resid = N - QA @ (QA.T @ N) if N.size else N
    sv = np.linalg.svd(resid, compute_uv=False) if N.size else np.array([])
    extra_dim = int((sv > 1e-8).sum())
    extra_loadings = {}
    if extra_dim:
        U_, s_, _ = np.linalg.svd(resid, full_matrices=False)
        for d in range(min(extra_dim, 6)):
            v = U_[:, d]
            w: dict[str, float] = {}
            for k, x in zip(cols, v, strict=True):
                w[k[0]] = w.get(k[0], 0.0) + float(x * x)
            extra_loadings[f"extra_{d}"] = {k: round(x, 4) for k, x in
                                            sorted(w.items(), key=lambda z: -z[1]) if x > 1e-6}

    # Q3: literal opening-vs-stacking trade, for every pair that can both stack inward
    # and close a loop
    idx = {k: n for n, k in enumerate(cols)}
    q3 = []
    for (i, j) in sorted(pairs_seen):
        if ("S", i, j) in idx and ("M", i, j) in idx:
            g = np.zeros(len(cols))
            g[idx[("S", i, j)]], g[idx[("M", i, j)]] = 1.0, -1.0
            q3.append(float(np.abs(M @ g).max()))
    # controls
    g = rng.standard_normal(len(cols))
    if N.size:
        g -= N @ (N.T @ g)
    c1 = float(np.abs(M @ g).max())
    c2 = float(np.abs(M @ N).max()) if N.size else 0.0
    Mc = M.tocoo()
    Ms = sp.csr_matrix((Mc.data, (Mc.row, rng.integers(0, len(cols), Mc.nnz))), shape=M.shape)
    Ns, _, _ = null_space(Ms)
    exd = MX(model_type, exterior_stem_term=False)
    Md = build_matrix([exd.counts(seq, s) for s in structs], cols)
    ext_pairs = [k for k in an if k.startswith("pair_gauge") and
                 any(pair_table(s)[int(k.split("_")[2])] == int(k.split("_")[3])
                     and _is_exterior(pair_table(s), int(k.split("_")[2])) for s in structs)]
    c4 = max(float(np.abs(Md @ an[k]).max()) for k in ext_pairs) if ext_pairs else 0.0

    # Q6: the constant-shift direction c_x moves every structure's total by exactly N
    cx = np.zeros(len(cols))
    for n_, k in enumerate(cols):
        if k[0] in ("S", "M", "u"):
            cx[n_] = 1.0
        elif k[0] == "U":
            cx[n_] = k[2] - k[1] + 1
    moved = M @ cx
    q6 = {"length_N": len(seq), "min_move": float(moved.min()), "max_move": float(moved.max()),
          "every_structure_moves_by_N": bool(np.abs(moved - len(seq)).max() <= TOL),
          "component_outside_exact_null_space": float(np.linalg.norm(cx - N @ (N.T @ cx))
                                                      / np.linalg.norm(cx)) if N.size else 1.0}

    return {
        "sequence": seq, "length": len(seq), "model_type": model_type,
        "C0_counting_invariants": {"n_structures_failing": c0_fail, "passes": c0_fail == 0},
        "n_structures": len(structs), "n_columns": len(cols), "shape": list(M.shape),
        "null_dim": int(N.shape[1]) if N.size else 0,
        "exact_rank": {"rank_mod_p": rk, "null_dim": exact_null,
                       "agrees_with_eigh": exact_null == (N.shape[1] if N.size else 0)},
        "eigh": info,
        "n_pairs_occurring": len(pairs_seen),
        "analytic": {
            "n_pair_gauges": sum(1 for k in an if k.startswith("pair_gauge")),
            "all_pair_gauges_null": all(v <= TOL for k, v in an_null.items()
                                        if k.startswith("pair_gauge")),
            "stack_vs_helix_length_max_change": an_null["stack_vs_helix_length"],
            "stack_vs_helix_length_null": an_null["stack_vs_helix_length"] <= TOL,
            "rank_of_analytic_set": rank_A,
            "null_dim_not_explained_by_analytic": extra_dim,
            "unexplained_directions_by_table": extra_loadings,
        },
        "Q3_opening_vs_stacking": {
            "n_pairs_tested": len(q3),
            "min_max_abs_change": min(q3) if q3 else None,
            "any_null": any(x <= TOL for x in q3),
        },
        "C1_random_nonnull_changes": {"max_abs_change": c1, "fails_as_required": c1 > 1e-6},
        "C2_null_basis_unchanged": {"max_abs_change": c2, "passes": c2 <= TOL},
        "C3_row_shuffle": {"null_dim_shuffled": int(Ns.shape[1]) if Ns.size else 0,
                           "changes_dim": (int(Ns.shape[1]) if Ns.size else 0)
                           != (int(N.shape[1]) if N.size else 0)},
        "C4_no_exterior_stem_term": {"n_exterior_capable_pairs": len(ext_pairs),
                                     "max_change_of_pair_gauge": c4,
                                     "detected": c4 > 1e-6},
        "Q6_constant_shift": q6,
    }


def _is_exterior(pt, i):
    """Is the pair opened at i an exterior-loop stem?"""
    depth_open = 0
    for k in range(1, i):
        if pt[k] > k and pt[k] > i:
            depth_open += 1
    return depth_open == 0


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str))


def main():
    t0 = time.time()
    out = {"run_id": RUN_ID, "status": "PRE-REGISTERED, not yet run", "predictions": PRED,
           "mxfold2_commit": MXFOLD2_COMMIT, **git_provenance(R),
           "scope": "positional decomposition of MXfold2 types C (shipped MixC) and M, per "
                    "sequence, exhaustive structure enumeration; MXfold2 not executed"}
    write(out)
    rng = np.random.default_rng(SEED)
    out["results"] = {}
    out["sequences"] = pick_sequences()
    for name, seq in out["sequences"].items():
        assert all(c in "ACGU" for c in seq)
        for mt in ("C", "M"):
            out["results"][f"{name}/{mt}"] = analyse(name, seq, mt, rng)
            write(out)
    res = out["results"].values()
    out["verdicts"] = {
        "Q1": "confirmed" if all(r["analytic"]["all_pair_gauges_null"] for r in res)
              else "refuted",
        "Q2": "confirmed" if all(r["analytic"]["stack_vs_helix_length_null"] for r in res)
              else "refuted",
        "Q3": "confirmed" if not any(r["Q3_opening_vs_stacking"]["any_null"] for r in res)
              else "refuted",
        "Q4": "holds ONLY for the exact null space: the regulariser is a function of one "
              "structure's total, so it is blind to every direction of G_x; it is NOT "
              "invariant under the hinge loss's full invariance, see Q4_correction",
        "Q4_correction": "the regulariser sees the constant-shift direction c_x (it moves the "
                         "reference total by N) and is blind to G_x: it pins exactly one "
                         "direction per sequence and leaves the rest free (user, 2026-10-07)",
        "Q6": "confirmed" if all(r["Q6_constant_shift"]["every_structure_moves_by_N"]
                                 for r in res) else "refuted",
        "null_dims": {k: r["null_dim"] for k, r in out["results"].items()},
        "pairs_occurring": {k: r["n_pairs_occurring"] for k, r in out["results"].items()},
        "controls": {k: {"C0": r["C0_counting_invariants"]["passes"],
                         "C1": r["C1_random_nonnull_changes"]["fails_as_required"],
                         "C2": r["C2_null_basis_unchanged"]["passes"],
                         "C3": r["C3_row_shuffle"]["changes_dim"],
                         "C4": r["C4_no_exterior_stem_term"]["detected"],
                         "exact_rank_agrees": r["exact_rank"]["agrees_with_eigh"]}
                     for k, r in out["results"].items()},
    }
    out["status"] = "complete"
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    write(out)
    print(json.dumps(out["verdicts"], indent=1))


if __name__ == "__main__":
    main()
