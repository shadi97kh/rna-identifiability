#!/usr/bin/env python3
"""The DATA-restricted null space of Turner 2004 on Rfam 15.1 seed structures.

THE QUESTION

`scripts/turner_nullspace.py` found the null space every structure leaves
invisible (the model-intrinsic one): dimension 2 without dangles, 3 with. A real
dataset sees only the structures it contains, so its null space can only be
larger. How much larger, and in which parameters?

THE DATA, AND WHAT IT IS NOT

Rfam 15.1 seed alignments (`data/raw/structure/Rfam_15.1.seed.gz`, registered as
`rfam_15_1_seed`, CC0). Each family's hand-curated consensus structure
(#=GC SS_cons) is PROJECTED onto each seed sequence. The result is a
comparative structure, not one determined experimentally for that molecule.
It is not ArchiveII (BLOCKED item 27) and must never be called that.

Projection rules, fixed before running:
  1. Consensus pairs are the nested brackets <>, (), [], {}. Pseudoknot letters
     (A-Z paired with a-z) are treated as unpaired. Only the #=GC SS_cons line is
     read; additional annotation lines some families carry (#=GC SS_cons_1,
     SS_cons_2, ...) are ignored. A family whose brackets cross
     is reported and its crossing pairs are dropped (counted).
  2. Residues are upper-cased and T->U. A sequence containing any character other
     than ACGU after that is dropped (counted), because the energy model has no
     parameters for it.
  3. A consensus pair is kept for a sequence only if both columns hold a residue in
     that sequence and the two residues form a canonical pair (AU, UA, GC, CG,
     GU, UG). Otherwise both positions become unpaired (counted).
  4. Pairs closing a hairpin with fewer than 3 unpaired nucleotides are removed,
     repeated to a fixpoint (counted). ViennaRNA cannot price them.
  5. Every remaining (sequence, structure) is one row. Duplicates are kept and
     counted; they cannot change a rank.

THE MODEL

Identical to the intrinsic run: ViennaRNA 2.7.2, Turner 2004, 37 C, pinned through
`thermo.vienna_ref`, dangles in {0, 2} x special_hp in {1, 0}, columns = reachable
parameter-file entries. Loops longer than 30 use table[30] plus ViennaRNA's
truncated (int)(lxc*log(u/30)), carried as a fixed offset (it is not linear in lxc).
Foundation check: counts . theta + offset equals ViennaRNA's eval_structure for
EVERY row, exactly, in integer units of 0.01 kcal/mol. If any row fails, stop.

THE DECOMPOSITION

null(M_data) = coverage (+) null of the used columns, where coverage is one
direction per reachable entry that no row uses. Within the used columns, the
intrinsic directions (restricted to used entries) are null by construction;
what is left over is the NONTRIVIAL data null space: combinations of entries the
data does use but never separates. Each part is reported with the tables it
loads on. Per table, an entry is INDIVIDUALLY IDENTIFIABLE when its unit vector is
orthogonal to the null space: the data fixes that one parameter regardless of
gauge.

PRE-REGISTERED PREDICTIONS (written into metrics.json before computing)

R1 (user). The data-restricted null space is strictly larger than the intrinsic
    one: > 2 for d0, > 3 for d2.
R2. Most of the excess is coverage, concentrated in the full lookup tables
    int22, int21, int11 and in the special-hairpin tables.
R3. After coverage and the intrinsic directions are removed, a non-zero remainder
    exists: combinations of used entries the data never separates.
R4. Every intrinsic direction, restricted to the used entries, is null on every
    row. This must hold mathematically; it is a check of the pipeline.
R5. The uniform stack shift is not null on Rfam either.
R6. The r1 generating-set artifact (mismatchM[t][x][A] - mismatchM[t][A][x]) is
    not null on real multiloops (d2).

CONTROLS (each must detect something)
C0 a corrupted parameter entry is caught by the reconstruction check on exactly
   the rows that use it; C1 a random direction with null part removed changes some
   row; C2 every null basis vector changes no row by more than 1e-9; C3 shuffling
   each row's labels among the used columns changes the null dimension; C4 the
   planted size-7 double-mismatch defect makes N1 non-null.

REVISION r2 (2026-10-07). The first run (runs/2026-10-07_turner_nullspace_rfam/)
agreed on every dimension across all three methods, with null and non-null
eigenvalues separated by at least six orders of magnitude, but FAILED C2: its
basis came from eigh of A^T A, which squares A's conditioning, and the vectors
moved some row by 3.6e-9 to 9.1e-9 against the 1e-9 bar. The bar is unchanged.
r2 takes the basis from a streamed QR of A and the SVD of R, at A's own
conditioning, and keeps the Gram eigh as a fourth dimension count. r1's artifact
is left as recorded.

THREE RANK COMPUTATIONS (four in r2: plus QR+SVD, which also supplies the basis)
eigh of the dense Gram matrix of the used columns; singleton elimination followed
by eigh of the reduced Gram matrix with back-substitution; exact rank over GF(p)
for the two largest primes below 2^20, of a random projection of the reduced
matrix, by blocked LU whose updates are exact float64 GEMMs. The projection
equals the rank mod p with probability at least 1 - cols/p per prime, and can
only UNDER-count rank, i.e. over-count null dimension, so an error shows up as
disagreement with the two float methods rather than hiding a direction.

Run: .venv/bin/python scripts/turner_nullspace_rfam.py
"""

from __future__ import annotations

import gzip
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
    PTYPE,
    Extractor,
    _reachable,
    analytic_directions,
    build_matrix,
    dump_params,
    eliminate_singletons,
    theta_from_par,
)
from thermo.vienna_ref import assert_turner2004_params, reference_md  # noqa: E402

RUN_ID = "2026-10-07_turner_nullspace_rfam_r2"
OUT = R / "runs" / RUN_ID
SEED_FILE = R / "data" / "raw" / "structure" / "Rfam_15.1.seed.gz"
TOL = 1e-9
SEED = 20261007
OPEN, CLOSE = "<([{", ">)]}"

PRED = {
    "R1": "data null space strictly larger than intrinsic: >2 (d0), >3 (d2)",
    "R2": "most of the excess is coverage, concentrated in int22/int21/int11 and special hairpins",
    "R3": "a non-zero nontrivial remainder exists beyond coverage and intrinsic",
    "R4": "every intrinsic direction restricted to used entries is null on every row (check)",
    "R5": "uniform stack shift is not null on Rfam",
    "R6": "the r1 generating-set multiloop artifact is not null on real multiloops (d2)",
}


# ----------------------------------------------------------------------------------
# Stockholm parsing and consensus projection
# ----------------------------------------------------------------------------------
def families(path: pathlib.Path):
    acc, ss, seqs, order = None, [], {}, []
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("# STOCKHOLM"):
                acc, ss, seqs, order = None, [], {}, []
            elif line.startswith("#=GF AC"):
                acc = line.split()[2]
            elif line.startswith("#=GC ") and line.split()[1] == "SS_cons":
                ss.append(line.split(None, 2)[2])
            elif line.startswith("//"):
                yield acc, "".join(ss), [(n, "".join(seqs[n])) for n in order]
            elif line and not line.startswith("#"):
                name, s = line.split(None, 1)
                if name not in seqs:
                    seqs[name] = []
                    order.append(name)
                seqs[name].append(s.strip())


def consensus_pairs(ss: str):
    stacks = {c: [] for c in OPEN}
    pairs = []
    for k, c in enumerate(ss):
        if c in OPEN:
            stacks[c].append(k)
        elif c in CLOSE:
            o = OPEN[CLOSE.index(c)]
            if stacks[o]:
                pairs.append((stacks[o].pop(), k))
    pairs.sort()
    # crossing pairs between bracket types: drop the later-opened of each crossing
    crossing = set()
    for a, (i, j) in enumerate(pairs):
        for p, q in pairs[a + 1:]:
            if p > j:
                break
            if i < p < j < q:
                crossing.add((p, q))
    return [p for p in pairs if p not in crossing], len(crossing)


def project(aligned: str, cpairs, stats):
    s = aligned.upper().replace("T", "U")
    cols = [k for k, c in enumerate(s) if c not in "-."]
    seq = "".join(s[k] for k in cols)
    if any(c not in "ACGU" for c in seq):
        stats["dropped_non_ACGU"] += 1
        return None
    pos = {k: n for n, k in enumerate(cols)}
    pt = [-1] * len(seq)
    for i, j in cpairs:
        if i in pos and j in pos:
            a, b = pos[i], pos[j]
            if seq[a] + seq[b] in PTYPE:
                pt[a], pt[b] = b, a
            else:
                stats["pairs_dropped_noncanonical"] += 1
        else:
            stats["pairs_dropped_gap"] += 1
    changed = True
    while changed:
        changed = False
        for a in range(len(seq)):
            b = pt[a]
            if b > a and b - a - 1 < 3 and all(pt[x] == -1 for x in range(a + 1, b)):
                pt[a] = pt[b] = -1
                stats["pairs_dropped_short_hairpin"] += 1
                changed = True
    db = "".join("(" if pt[k] > k else ")" if pt[k] >= 0 else "." for k in range(len(seq)))
    return seq, db


def load_rows():
    stats = {k: 0 for k in ("families", "families_with_crossing", "pairs_dropped_crossing",
                            "sequences", "dropped_non_ACGU", "pairs_dropped_gap",
                            "pairs_dropped_noncanonical", "pairs_dropped_short_hairpin",
                            "rows_without_pairs")}
    rows = []
    for acc, ss, seqs in families(SEED_FILE):
        stats["families"] += 1
        cp, ncross = consensus_pairs(ss)
        if ncross:
            stats["families_with_crossing"] += 1
            stats["pairs_dropped_crossing"] += ncross
        for name, aligned in seqs:
            stats["sequences"] += 1
            assert len(aligned) == len(ss), (acc, name)
            r = project(aligned, cp, stats)
            if r is None:
                continue
            if "(" not in r[1]:
                stats["rows_without_pairs"] += 1
            rows.append((acc, name, r[0], r[1]))
    stats["rows"] = len(rows)
    stats["distinct_rows"] = len({(s, d) for _, _, s, d in rows})
    lens = np.array([len(s) for _, _, s, _ in rows])
    stats["length_min_median_max"] = [int(lens.min()), int(np.median(lens)), int(lens.max())]
    return rows, stats


# ----------------------------------------------------------------------------------
# linear algebra on the used columns
# ----------------------------------------------------------------------------------
def gram_null(A: sp.csr_matrix):
    G = (A.T @ A).astype(np.float64).toarray()
    return eigh_null_from_gram(G)


def eigh_null_from_gram(G):
    w, V = np.linalg.eigh(G)
    tol = max(G.shape) * np.finfo(float).eps * max(w.max(), 1.0)
    null = w < tol
    return V[:, null], {"eig_tol": float(tol),
                        "largest_null_eig": float(w[null].max()) if null.any() else None,
                        "smallest_nonnull_eig": float(w[~null].min()) if (~null).any() else None}


def qr_null(A: sp.csr_matrix, chunk: int = 16384, return_factors: bool = False):
    """Null basis from a streamed QR of A itself, then the SVD of R.

    Works at A's conditioning, not the square of it that eigh of A^T A pays, so the
    basis vectors resolve the 1e-9 row bar (r1 of this run, with Gram eigenvectors,
    left residuals of 3.6e-9 to 9.1e-9 although every dimension count agreed).
    """
    Rm = None
    for a in range(0, A.shape[0], chunk):
        B = A[a:a + chunk].astype(np.float64).toarray()
        Rm = np.linalg.qr(B if Rm is None else np.vstack([Rm, B]), mode="r")
    _, s, Vt = np.linalg.svd(Rm)
    tol = max(A.shape) * np.finfo(float).eps * s.max()
    null = s < tol
    info = {"sv_tol": float(tol),
            "largest_null_sv": float(s[null].max()) if null.any() else None,
            "smallest_nonnull_sv": float(s[~null].min()) if (~null).any() else None}
    if return_factors:  # the full spectrum, for the effective-rank analysis
        return Vt[null].T.copy(), info, Rm, s, Vt
    return Vt[null].T.copy(), info


def reduced_null(A: sp.csr_matrix):
    red, order, keep_c = eliminate_singletons(A)
    Nred, info = (eigh_null_from_gram((red.T @ red).astype(np.float64).toarray())
                  if red.shape[1] else (np.zeros((0, 0)), {}))
    k = Nred.shape[1] if Nred.size else 0
    N = np.zeros((A.shape[1], k))
    if k:
        N[keep_c] = Nred
        Ar = A.tocsr()
        for r, c in reversed(order):
            row = Ar[r]
            N[c] = -(row @ N) / Ar[r, c]
        N, _ = np.linalg.qr(N)
    return N, dict(info, singleton_columns_eliminated=len(order),
                   reduced_shape=list(red.shape)), red


PRIMES = (1_048_573, 1_048_571)  # the two largest primes below 2^20


def rank_mod_p_blocked(A: np.ndarray, p: int, b: int = 512) -> int:
    """Exact rank of an integer matrix over GF(p), p < 2^20, by blocked LU.

    Entries are held as float64 in [0, p). A product of two is below 2^40 and a
    sum of b <= 512 such products below 2^49 < 2^53, so every matrix multiply here
    is exact in float64, which lets the trailing update run as one BLAS GEMM per
    panel instead of one Python-level outer product per pivot.
    """
    assert p < 2**20 and b * p * p < 2**53
    A = np.mod(np.asarray(A, dtype=np.int64), p).astype(np.float64)
    m, n = A.shape
    r, c0 = 0, 0
    while c0 < n and r < m:
        c1 = min(c0 + b, n)
        rows = m - r
        P = A[r:, c0:c1].copy()
        L = np.zeros((rows, c1 - c0))
        perm = np.arange(rows)
        k = 0
        for c in range(c1 - c0):
            if k >= rows:
                break
            nz = np.nonzero(P[k:, c])[0]
            if nz.size == 0:
                continue
            s_ = k + nz[0]
            if s_ != k:
                P[[k, s_]] = P[[s_, k]]
                L[[k, s_]] = L[[s_, k]]
                perm[[k, s_]] = perm[[s_, k]]
            inv = pow(int(P[k, c]), p - 2, p)
            below = np.nonzero(P[k + 1:, c])[0] + k + 1
            if below.size:
                mult = np.fmod(P[below, c] * inv, p)
                L[below, k] = mult
                P[below, c:] = np.fmod(P[below, c:] - np.fmod(np.outer(mult, P[k, c:]), p) + p, p)
            k += 1
        if c1 < n and k:
            T = A[r:, c1:][perm]
            Lk = L[:, :k]
            for i in range(1, k):  # U12 = L11^-1 A12, unit lower triangular
                T[i] = np.fmod(T[i] - np.fmod(Lk[i, :i] @ T[:i], p) + p, p)
            if rows > k:
                T[k:] = np.fmod(T[k:] - np.fmod(Lk[k:] @ T[:k], p) + p, p)
            A[r:, c1:] = T
        r += k
        c0 = c1
    return r


def projected_rank_mod_p(red: sp.csr_matrix, p: int, rng) -> int:
    k = red.shape[1] + 8
    acc = np.zeros((red.shape[1], k), dtype=np.int64)
    redT = red.T.tocsr().astype(np.int64)
    step = 4096
    for a in range(0, red.shape[0], step):  # chunked: bounded memory, no int64 overflow
        Rm = rng.integers(0, p, size=(min(step, red.shape[0] - a), k), dtype=np.int64)
        acc = (acc + (redT[:, a:a + step] @ Rm) % p) % p
    return rank_mod_p_blocked(acc.T, p)


def table_identifiability(N, cols):
    """Per table: used entries, and entries whose unit vector is orthogonal to null."""
    proj = np.linalg.norm(N, axis=1) if N.size else np.zeros(len(cols))
    out = {}
    for k, x in zip(cols, proj, strict=True):
        d = out.setdefault(k[0], {"used": 0, "individually_identifiable": 0})
        d["used"] += 1
        d["individually_identifiable"] += int(x < 1e-8)
    return out


def table_weights(B, cols):
    if not B.size:
        return {}
    w = {}
    for k, x in zip(cols, (B * B).sum(axis=1), strict=True):
        w[k[0]] = w.get(k[0], 0.0) + float(x)
    tot = sum(w.values())
    return {k: round(v / tot, 4) for k, v in sorted(w.items(), key=lambda z: -z[1]) if v > 1e-9}


# ----------------------------------------------------------------------------------
def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str))


def analyse(dangles, special_hp, P, rows, rng):
    theta, special, lxc = P["theta"], P["special"], P["lxc"]
    md = reference_md()
    md.dangles, md.special_hp = dangles, special_hp
    ex = Extractor(special, dangles, special_hp, lxc=lxc)
    counts, energies, offsets, fails = [], [], [], []
    checked = False
    for acc, name, seq, db in rows:
        fc = RNA.fold_compound(seq, md)
        if not checked:
            assert_turner2004_params(fc)
            checked = True
        e = int(round(fc.eval_structure(db) * 100))
        c = ex.counts(seq, db)
        off = c.pop(("_offset",), 0)
        if sum(theta[k] * v for k, v in c.items()) + off != e:
            fails.append({"family": acc, "name": name, "len": len(seq), "vienna": e,
                          "recon": sum(theta[k] * v for k, v in c.items()) + off})
        counts.append(c)
        energies.append(e)
        offsets.append(off)
    res = {"dangles": dangles, "special_hp": special_hp,
           "reconstruction": {"n_checked": len(rows), "n_failed": len(fails),
                              "first_failures": fails[:10], "passes": not fails,
                              "rows_with_long_loop_offset": int(sum(o != 0 for o in offsets))}}
    if fails:
        return res

    c0 = {}
    for probe in [("stack", 1, 2), ("mismatchH", 5, 3, 1), ("ML_ci",), ("TerminalAU",),
                  ("hairpin", 30)]:
        bad = dict(theta)
        bad[probe] += 1
        n_use = sum(1 for c in counts if c.get(probe, 0))
        n_fail = sum(1 for c, e, o in zip(counts, energies, offsets, strict=True)
                     if sum(bad[k] * v for k, v in c.items()) + o != e)
        c0[repr(probe)] = {"rows_using": n_use, "rows_failing": n_fail,
                           "detected": n_fail == n_use > 0}
    res["C0_corrupted_entry_detected"] = c0

    cols = [k for k in sorted(theta, key=repr) if _reachable(k, dangles, special_hp)]
    M = build_matrix(counts, cols)
    nnz_col = np.asarray((M != 0).sum(axis=0)).ravel()
    used = np.nonzero(nnz_col)[0]
    unused = np.nonzero(nnz_col == 0)[0]
    ucols = [cols[i] for i in used]
    A = M[:, used].tocsr()

    t0 = time.time()
    Ng, inf = gram_null(A)
    Nf, inq = qr_null(A)  # the basis every downstream quantity uses
    Nr, inr, red = reduced_null(A)
    ranks = {str(p): projected_rank_mod_p(red, p, rng) for p in PRIMES}
    exact_null = red.shape[1] - max(ranks.values())
    null_used = Nf.shape[1]

    an = analytic_directions(cols, dangles, special_hp)
    an_u = {k: g[used] for k, g in an.items()}
    an_null = {k: float(np.abs(A @ g).max()) for k, g in an_u.items()}
    Au = np.stack([g for g in an_u.values() if np.abs(g).max() > 0], axis=1)
    rank_an = int(np.linalg.matrix_rank(Au))
    Qa, _ = np.linalg.qr(Au)
    resid = Nf - Qa @ (Qa.T @ Nf)
    U_, s_, _ = np.linalg.svd(resid, full_matrices=False)
    nontriv = U_[:, s_ > 1e-8]

    by_table_unused: dict[str, int] = {}
    for i in unused:
        by_table_unused[cols[i][0]] = by_table_unused.get(cols[i][0], 0) + 1
    res.update({
        "shape": list(M.shape), "n_columns_reachable": len(cols),
        "null_dim_total": int(len(unused) + null_used),
        "coverage": {"dim": int(len(unused)), "by_table": dict(sorted(
            by_table_unused.items(), key=lambda z: -z[1]))},
        "used_columns": len(used),
        "null_dim_within_used": {"qr_svd": null_used, "eigh_gram": int(Ng.shape[1]),
                                 "eigh_reduced": int(Nr.shape[1]),
                                 "exact_mod_p": exact_null, "rank_mod_p": ranks,
                                 "agree": null_used == Ng.shape[1] == Nr.shape[1] == exact_null,
                                 "qr": inq, "gram": inf, "reduced": inr,
                                 "seconds": round(time.time() - t0, 1)},
        "intrinsic_restricted": {"max_abs_row_change": an_null,
                                 "all_null": all(v <= TOL for v in an_null.values()),
                                 "rank_on_used_columns": rank_an},
        "nontrivial": {"dim": int(nontriv.shape[1]),
                       "weight_by_table": table_weights(nontriv, ucols)},
        "identifiability_by_table": table_identifiability(Nf, ucols),
    })

    # controls
    g = rng.standard_normal(len(used))
    g -= Nf @ (Nf.T @ g)
    res["C1_random_nonnull_changes"] = {"max_abs_change": float(np.abs(A @ g).max()),
                                        "fails_as_required": float(np.abs(A @ g).max()) > 1e-6}
    ch = float(np.abs(A @ Nf).max()) if Nf.size else 0.0
    res["C2_null_basis_unchanged"] = {"max_abs_change": ch, "passes": ch <= TOL}
    Ac = A.tocoo()
    As = sp.csr_matrix((Ac.data, (Ac.row, rng.integers(0, A.shape[1], Ac.nnz))), shape=A.shape)
    Ns, _ = gram_null(As)
    res["C3_row_shuffle"] = {"null_dim_shuffled": int(Ns.shape[1]),
                             "changes_dim": int(Ns.shape[1]) != null_used}
    exd = Extractor(special, dangles, special_hp, planted_defect=True, lxc=lxc)
    dc = []
    for _, _, s, d in rows:
        c = exd.counts(s, d)
        c.pop(("_offset",), None)
        dc.append(c)
    Md = build_matrix(dc, cols)[:, used]
    d4 = float(np.abs(Md @ an_u["N1_hairpin_mismatch_vs_init"]).max())
    res["C4_planted_defect"] = {"N1_max_abs_change": d4, "detected": d4 > 1e-6}
    gs = np.array([1.0 if k[0] == "stack" else 0.0 for k in ucols])
    res["R5_uniform_stack_shift"] = {"max_abs_change": float(np.abs(A @ gs).max()),
                                     "in_null_space": float(np.abs(A @ gs).max()) <= TOL}
    if dangles == 2:
        idx = {k: n for n, k in enumerate(ucols)}
        g6 = np.zeros(len(ucols))
        for t in range(1, 7):
            for x in (2, 3, 4):
                for key, v in ((("mismatchM", t, x, 1), 1.0), (("mismatchM", t, 1, x), -1.0)):
                    if key in idx:
                        g6[idx[key]] += v
        res["R6_r1_artifact"] = {"max_abs_change": float(np.abs(A @ g6).max()),
                                 "in_null_space": float(np.abs(A @ g6).max()) <= TOL}
    return res


def main():
    t0 = time.time()
    out = {"run_id": RUN_ID, "status": "PRE-REGISTERED, not yet run", "predictions": PRED,
           **git_provenance(R), "viennarna_version": RNA.__version__,
           "data": "rfam_15_1_seed (CC0), consensus structures projected onto seed sequences; "
                   "comparative, not per-molecule experimental; not ArchiveII"}
    write(out)
    P = theta_from_par(dump_params())
    rows, stats = load_rows()
    out["projection"] = stats
    write(out)
    rng = np.random.default_rng(SEED)
    out["configs"] = {}
    for dangles in (0, 2):
        for special_hp in (1, 0):
            name = f"d{dangles}_special_hp{special_hp}"
            out["configs"][name] = analyse(dangles, special_hp, P, rows, rng)
            write(out)
            if not out["configs"][name]["reconstruction"]["passes"]:
                out["status"] = f"STOPPED: reconstruction failed in {name}"
                write(out)
                return 1
    cf = out["configs"]
    intrinsic = {"d0_special_hp1": 2, "d0_special_hp0": 2, "d2_special_hp1": 3, "d2_special_hp0": 3}
    out["verdicts"] = {
        "R1": {"verdict": "confirmed" if all(cf[k]["null_dim_total"] > v
                                             for k, v in intrinsic.items()) else "refuted",
               "null_dim_total": {k: c["null_dim_total"] for k, c in cf.items()}},
        "R2": {k: {"coverage": c["coverage"]["dim"],
                   "excess": c["null_dim_total"] - intrinsic[k],
                   "top_tables": dict(list(c["coverage"]["by_table"].items())[:5])}
               for k, c in cf.items()},
        "R3": {"verdict": "confirmed" if all(c["nontrivial"]["dim"] > 0 for c in cf.values())
               else "refuted",
               "nontrivial_dim": {k: c["nontrivial"]["dim"] for k, c in cf.items()}},
        "R4": all(c["intrinsic_restricted"]["all_null"] for c in cf.values()),
        "R5": not any(c["R5_uniform_stack_shift"]["in_null_space"] for c in cf.values()),
        "R6": not any(c.get("R6_r1_artifact", {}).get("in_null_space", False)
                      for c in cf.values()),
        "controls": {k: {"C0": all(x["detected"]
                                   for x in c["C0_corrupted_entry_detected"].values()),
                         "C1": c["C1_random_nonnull_changes"]["fails_as_required"],
                         "C2": c["C2_null_basis_unchanged"]["passes"],
                         "C3": c["C3_row_shuffle"]["changes_dim"],
                         "C4": c["C4_planted_defect"]["detected"],
                         "rank_methods_agree": c["null_dim_within_used"]["agree"]}
                     for k, c in cf.items()},
    }
    out["status"] = "complete"
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    write(out)
    print(json.dumps(out["verdicts"], indent=1))
    return 0


# ----------------------------------------------------------------------------------
# effective rank: python scripts/turner_nullspace_rfam.py --effective-rank
# ----------------------------------------------------------------------------------
ER_RUN_ID = "2026-10-07_rfam_effective_rank"
ER_OUT = R / "runs" / ER_RUN_ID
SIGMAS = {"kierzek2005_stated_rms": 0.6, "viennarna_table_resolution": 0.01}
DELTAS = [float(x) for x in np.logspace(-3, 1, 41)]  # 0.001 .. 10 kcal/mol, 10 per decade
ANCHOR_DELTAS = [0.001, 0.01, 0.1, 0.6, 1.0, 10.0]
PLANT_EPS = 1e-4

ER_DOC = {
    "question": "The exact null space counts directions with singular value below machine "
                "precision. A direction with a merely small singular value s is also not "
                "recoverable: if each row's free energy carried error sigma (kcal/mol), a "
                "least-squares fit fixes that direction's coefficient only to sigma / s "
                "(kcal/mol; A is a count matrix, so sigma / s carries sigma's units). This "
                "records the whole spectrum and the curve of how many directions are "
                "uncertain beyond delta, for a grid of delta.",
    "lower_bound": "The exact null dimension is a LOWER BOUND on what the data cannot "
                   "determine. Every direction counted at any (sigma, delta) here includes "
                   "the exact null directions (infinite uncertainty) plus the directions whose "
                   "finite uncertainty exceeds delta.",
    "caveats": [
        "Rfam rows are consensus STRUCTURES, not energy measurements. sigma / s is the "
        "uncertainty that this design WOULD leave if every row's free energy were measured "
        "with independent error sigma. It characterises the conditioning of the design; it "
        "is not an error bar Rfam itself delivers.",
        "About 10% of rows duplicate another (sequence, structure) exactly; they count as "
        "repeated measurements here, which can shrink sigma / s along directions they "
        "dominate by at most sqrt(multiplicity).",
        "Directions are right singular vectors of A over the USED columns; the coverage "
        "columns (entries no row uses) are each undetermined in addition and are reported "
        "separately, not folded into the curve.",
    ],
    "sigma_anchors": {"kierzek2005_stated_rms": "0.6 kcal/mol, the RMS Kierzek et al. 2005 "
                      "state for their own fit", "viennarna_table_resolution": "0.01 kcal/mol, "
                      "the resolution of ViennaRNA's integer parameter tables (dcal/mol)"},
}
ER_PRED = {
    "E1": "At sigma = 0.6 and delta = 0.1 kcal/mol, the number of directions (over the used "
          "columns) with uncertainty above delta is at least 10 x the exact null dimension "
          "over the used columns. (Read against the TOTAL null dimension including coverage, "
          "10x would exceed the number of reachable columns, so only the within-used reading "
          "can be tested; recorded before computing.)",
}


def count_matrix(dangles, special_hp, P, rows):
    """Count matrix with r2's exact reconstruction check on every row."""
    md = reference_md()
    md.dangles, md.special_hp = dangles, special_hp
    ex = Extractor(P["special"], dangles, special_hp, lxc=P["lxc"])
    counts, fails, checked = [], 0, False
    for _, _, seq, db in rows:
        fc = RNA.fold_compound(seq, md)
        if not checked:
            assert_turner2004_params(fc)
            checked = True
        e = int(round(fc.eval_structure(db) * 100))
        c = ex.counts(seq, db)
        off = c.pop(("_offset",), 0)
        fails += sum(P["theta"][k] * v for k, v in c.items()) + off != e
        counts.append(c)
    cols = [k for k in sorted(P["theta"], key=repr) if _reachable(k, dangles, special_hp)]
    return build_matrix(counts, cols).tocsr(), cols, fails


def planted_control(Rm, s_tol, n_null, j):
    """Append a near-duplicate of column j on a new row. Because A = Q R, the bordered
    R' = [[R, r_j], [0, eps]] has exactly the singular values of the augmented A, and
    the direction (e_j - e_new)/sqrt(2) has norm eps/sqrt(2). eps = 0 duplicates it."""
    n = Rm.shape[1]
    out = {}
    for eps in (PLANT_EPS, 0.0):
        Rp = np.zeros((n + 1, n + 1))
        Rp[:n, :n] = Rm
        Rp[:n, n] = Rm[:, j]
        Rp[n, n] = eps
        sp_ = np.linalg.svd(Rp, compute_uv=False)
        if eps:
            pred = eps / np.sqrt(2)
            near = sp_[(sp_ > 0.99 * pred) & (sp_ < 1.01 * pred)]
            out["near_duplicate"] = {
                "eps": eps, "predicted_sv": float(pred),
                "observed_sv_within_1pct": [float(x) for x in near],
                "null_count_unchanged": int((sp_ < s_tol).sum()) == n_null,
                "detected": near.size == 1 and int((sp_ < s_tol).sum()) == n_null}
        else:
            out["exact_duplicate"] = {"null_count": int((sp_ < s_tol).sum()),
                                      "expected": n_null + 1,
                                      "detected": int((sp_ < s_tol).sum()) == n_null + 1}
    return out


def main_effective_rank():
    t0 = time.time()
    out = {"run_id": ER_RUN_ID, "status": "PRE-REGISTERED, not yet run", "predictions": ER_PRED,
           **ER_DOC, "deltas": DELTAS, "sigmas": SIGMAS, **git_provenance(R),
           "viennarna_version": RNA.__version__,
           "data": "rfam_15_1_seed rows exactly as in run 2026-10-07_turner_nullspace_rfam_r2"}
    ER_OUT.mkdir(parents=True, exist_ok=True)

    def save():
        (ER_OUT / "metrics.json").write_text(json.dumps(out, indent=1, sort_keys=True,
                                                        default=str))
    save()
    P = theta_from_par(dump_params())
    rows, stats = load_rows()
    out["rows"] = {"n": stats["rows"], "distinct": stats["distinct_rows"]}
    out["configs"] = {}
    for dangles in (0, 2):
        for special_hp in (1, 0):
            name = f"d{dangles}_special_hp{special_hp}"
            M, cols, fails = count_matrix(dangles, special_hp, P, rows)
            res = {"reconstruction_failures": int(fails)}
            out["configs"][name] = res
            if fails:
                out["status"] = f"STOPPED: reconstruction failed in {name}"
                save()
                return 1
            used = np.nonzero(np.asarray((M != 0).sum(axis=0)).ravel())[0]
            ucols = [cols[i] for i in used]
            A = M[:, used].tocsr()
            N, info, Rm, s, Vt = qr_null(A, return_factors=True)
            null = s < info["sv_tol"]
            n_null = int(null.sum())
            res.update({"used_columns": int(len(used)),
                        "coverage_columns": int(len(cols) - len(used)),
                        "exact_null_dim_used": n_null, "qr": info,
                        "singular_values": [float(f"{x:.9g}") for x in s]})
            tables = np.array([k[0] for k in ucols])
            res["sigma"] = {}
            for sname, sigma in SIGMAS.items():
                with np.errstate(divide="ignore"):
                    unc = np.where(null, np.inf, sigma / np.where(null, 1.0, s))
                curve = {f"{d:.6g}": int((unc > d).sum()) for d in DELTAS}
                anchors = {}
                for d in ANCHOR_DELTAS:
                    sel = unc > d
                    w = (Vt[sel] ** 2).sum(axis=0)
                    by = {t: float(w[tables == t].sum()) for t in np.unique(tables)}
                    tot = sum(by.values()) or 1.0
                    anchors[f"{d:g}"] = {
                        "n_directions_used": int(sel.sum()),
                        "n_beyond_exact_null": int(sel.sum()) - n_null,
                        "n_including_coverage": int(sel.sum()) + len(cols) - len(used),
                        "ratio_to_exact_null": float(sel.sum() / n_null) if n_null else None,
                        "weight_by_table": {t: round(v / tot, 4) for t, v in
                                            sorted(by.items(), key=lambda z: -z[1])
                                            if v / tot > 1e-4}}
                # marginal standard error of each individually identifiable entry
                inv = np.where(null, 0.0, 1.0 / np.where(null, 1.0, s))
                se = sigma * np.sqrt(((Vt * inv[:, None]) ** 2).sum(axis=0))
                ident = (Vt[null] ** 2).sum(axis=0) < 1e-16
                per_table = {}
                for t in np.unique(tables):
                    m = (tables == t) & ident
                    if m.any():
                        per_table[t] = {"n_identifiable": int(m.sum()),
                                        "median_se": float(np.median(se[m])),
                                        "max_se": float(se[m].max())}
                res["sigma"][sname] = {
                    "sigma_kcal": sigma,
                    "uncertainty_per_direction": [None if not np.isfinite(x)
                                                  else float(f"{x:.6g}") for x in unc],
                    "curve_n_above_delta_used": curve, "anchors": anchors,
                    "marginal_se_identifiable_entries_by_table": per_table}
            j = ucols.index(("stack", 1, 2))
            res["control_planted_column"] = planted_control(Rm, info["sv_tol"], n_null, j)
            save()
    cf = out["configs"]
    a = {k: c["sigma"]["kierzek2005_stated_rms"]["anchors"]["0.1"] for k, c in cf.items()}
    e1 = all(v["n_directions_used"] >= 10 * cf[k]["exact_null_dim_used"] for k, v in a.items())
    out["verdicts"] = {
        "E1": {"verdict": "confirmed" if e1 else "refuted",
               "per_config": {k: {"exact_null_used": cf[k]["exact_null_dim_used"],
                                  "n_above_0.1_at_sigma_0.6": v["n_directions_used"],
                                  "ratio": v["ratio_to_exact_null"]} for k, v in a.items()}},
        "control_planted": {k: {"near_duplicate": c["control_planted_column"]["near_duplicate"]
                                ["detected"], "exact_duplicate": c["control_planted_column"]
                                ["exact_duplicate"]["detected"]} for k, c in cf.items()},
    }
    out["status"] = "complete"
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    save()
    print(json.dumps(out["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main_effective_rank() if "--effective-rank" in sys.argv else main())
