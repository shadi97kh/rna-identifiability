#!/usr/bin/env python3
"""The null space of the Turner 2004 nearest-neighbour folding model. Stop-or-go check.

THE QUESTION

Every structure's free energy in the nearest-neighbour model is a sum of table
lookups, so E(s) = c(s) . theta exactly, where c(s) counts how many times each
parameter-file entry is used. A direction g with c(s) . g = 0 for every s is
invisible to every measurement of a structure's free energy: it is a gauge. The
Turner-gauge direction (`turner-gauge-plan.md`, step 1) exists only if that set is
not empty. This script decides it.

WHAT IS COMPUTED HERE, AND WHAT IS NOT

The plan asks for the null space of M, the count matrix of ArchiveII's
structures. ArchiveII is BLOCKED (docs/BLOCKED.md item 27: its canonical
distribution is gone and it never stated a licence), so that matrix is not built.

What is built is the MODEL-INTRINSIC null space: the directions that leave every
valid (sequence, structure) pair unchanged. It needs no dataset, and it is a
subset of the null space of any dataset whatsoever, because a data matrix only
has fewer rows. So:
  - intrinsic null space non-empty  =>  every data null space is non-empty, and
    the stop condition of step 1 cannot fire;
  - intrinsic null space empty      =>  the data null space would consist only of
    entries the data never exercises (coverage), which is a weaker object.

To compute it, the rows of M are a GENERATING SET: deterministic constructions
(no RNG) that exercise every reachable parameter-file entry in every loop
context. They are an algebraic device for spanning the model's domain, never
data, never fitted, and never presented as measurements. The computed null space
of any set of valid structures CONTAINS the intrinsic one; the analytic
directions below are PROVED to lie in the intrinsic one. If the two dimensions
are equal, the sandwich closes and the intrinsic null space is exactly the span
of the analytic directions.

THE MODEL

ViennaRNA 2.7.2, Turner 2004, 37 C, model pinned through `thermo.vienna_ref`
(`reference_md`, then `dangles` and `special_hp` varied explicitly, every other
field as pinned). Four configurations: dangles in {0, 2} x special_hp in {1, 0}.
d0 is Krueger et al. 2025's default; d2 is ViennaRNA's default; special_hp=1 is
the full Turner 2004 model, special_hp=0 is this repo's correctness-gate pin.

Columns are PARAMETER-FILE entries (what `RNA.params_save` writes), restricted to
those reachable with an ACGU sequence and canonical pairs. `MLintern` is the
single scalar `ci` of the ML_params line, not ViennaRNA's derived per-type array.
The large-loop extrapolation `(int)(lxc*log(u/30))` is not linear in `lxc`
(truncation), so `lxc` is not a column and the generating set uses loops of at
most 30 unpaired nucleotides; data with longer loops adds a fixed offset only.

The ninio term MIN2(MAX_NINIO, asym*m) is piecewise linear. The count vector is
its linearisation at the published values: asymmetry 1-5 counts `ninio_m` (at 5
the macro returns its second argument, asym*m), 6+ counts `ninio_max`.

THE FOUNDATION CHECK

For every row, counts . theta must equal ViennaRNA's own `eval_structure` total
EXACTLY, in integer units of 0.01 kcal/mol (stricter than the plan's 1e-9 kcal).
If any row fails, the script stops and writes nothing but the failure.

PRE-REGISTERED PREDICTIONS (written into metrics.json before anything is computed)

From the plan, verbatim in substance:
P1. The null space is NOT empty (fingerprint: Zuber et al. 2018 report Monte
    Carlo parameter correlations of exactly +1.0).
P2. The uniform stack shift (add c to every stack entry) is NOT in the null
    space: a structure has (#stacked pairs) stack lookups, which varies.
P3. The surviving directions are trades among loop initiation and per-unpaired
    increments, dangling ends against terminal mismatches, and coaxial terms
    against multiloop terms.

Added before the run, from reading ViennaRNA's loop functions (E_Hairpin,
E_IntLoop, E_MLstem, vrna_E_ext_stem):
A1. N1 = (+1 on every mismatchH entry, -1 on hairpin[4..30]) is null: every
    non-special hairpin of size >= 4 makes exactly one lookup in each. With
    special_hp=0, size-3 hairpins also take mismatchH, so hairpin[3] joins (-1).
A2. N2 = (+1 on every mismatchI, mismatch1nI and mismatch23I entry, -2 on
    internal_loop[4..30]) is null: every loop that uses internal_loop[u] makes
    exactly two mismatch lookups, and the three families share internal_loop[u]
    across loop shapes (u=5: 1x4 and 2x3; u>=6: 1xn and generic), so they cannot
    shift separately.
A3. (d2 only) N3 = (+1 on every mismatchM entry, -1 on ci) is null: every
    multiloop stem, closing pair included, always has both neighbours and so
    always takes exactly one mismatchM and one ci.
A4. Nothing else. In particular there is no per-helix-end ("pair potential")
    gauge: an exterior stem has no per-stem constant in either dangle model (a
    stem pairing nucleotide 1 with nucleotide n takes only TerminalAU), which
    forces it to zero. Predicted dimensions: d0 = 2, d2 = 3, for both special_hp
    settings, and the same after the orientation-symmetry tying below.
A5. P3 is mostly wrong as stated: the survivors are initiation-vs-MISMATCH
    trades, not per-unpaired ones; dangles vs terminal mismatches is not null
    (no counter-term); coaxial stacking does not exist in this model.

NEGATIVE CONTROLS (each must fail, i.e. must detect something)

C1. A random direction with its null component projected out changes at least
    one row's energy by more than 1e-6.
C2. Every reported null basis vector leaves every row unchanged to 1e-9; a
    vector that fails is not null and the rank tolerance is wrong.
C3. Shuffling the motif labels within the count matrix (each row's counts
    reassigned to randomly chosen columns) changes the null dimension. A pure
    column permutation is reported alongside: rank is invariant under it, so it
    is NOT a valid control and must leave the dimension unchanged.
C4. A planted defect: replace the generating set's hairpins of size 7 by a
    model that also charges a second mismatchH lookup (counts only) -- N1 must
    then stop being null. This shows the null check can see a broken identity.

TWO PARAMETERISATIONS

raw   every reachable parameter-file entry is its own column.
tied  entries describing the same motif read from the other strand are merged,
      as Krueger et al. 2025's "minimal symmetries" do: stack[a][b]=stack[b][a],
      int11[a][b][x][y]=int11[b][a][y][x], int22[a][b][i][j][k][l]=
      int22[b][a][k][l][i][j]. (int21 is already orientation-canonical in
      ViennaRNA's lookup.)
Krueger's own 284-parameter strict-extrapolation set and 12,291-value set are
NOT reproduced: their mapping onto table entries is not in the public
jax-rnafold (vendored at 13f08b7), and the rules live in Zuber et al. 2018's
supplement, whose only link is the dead /pub/ directory (BLOCKED item 27).

REVISION r2 (2026-10-07). The first run, runs/2026-10-07_turner_nullspace/, closed
the sandwich for d0 (2 = 2) but not for d2: it found null dimension 4 against the
predicted 3. The fourth direction was mismatchM[t][m5][m3] += g(m3) - g(m5) with
g(A)=1 and 0 otherwise. Around a multiloop this telescopes to zero whenever every
gap between stems is one nucleotide or all A, which was true of every multiloop the
r1 generating set built, and is false for real structures (any gap such as "CG").
It is therefore a coverage gap of the generating set, not a gauge of the model. r2
adds multiloops with two-nucleotide gaps of every composition and directly adjacent
stems, keeps every prediction unchanged, and adds control C5: that exact direction
must now change some row. r1's artifact is left as it is.

Run: .venv/bin/python scripts/turner_nullspace.py
Writes runs/<RUN_ID>/metrics.json. Nothing under models/, datasets/, thermo/ is
modified.
"""

from __future__ import annotations

import itertools
import json
import math
import pathlib
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import RNA  # noqa: E402

from scripts.runmeta import git_provenance  # noqa: E402
from thermo.vienna_ref import assert_turner2004_params, reference_md  # noqa: E402

RUN_ID = "2026-10-07_turner_nullspace_r2"
OUT = R / "runs" / RUN_ID
EXACT_ROW_TOL_KCAL = 1e-9  # the plan's tolerance; the integer check is stricter
NONNULL_MIN_CHANGE = 1e-6  # C1
SHUFFLE_SEED = 20261007

PRED = {
    "P1": "null space NOT empty",
    "P2": "uniform stack shift NOT in the null space",
    "P3": "survivors are initiation/per-unpaired trades, dangles vs terminal mismatches, "
          "coaxial vs multiloop",
    "A1": "N1 = +mismatchH(all) -hairpin[4..30] (and -hairpin[3] when special_hp=0) is null",
    "A2": "N2 = +mismatchI,+mismatch1nI,+mismatch23I(all) -2*internal_loop[4..30] is null",
    "A3": "N3 = +mismatchM(all) -ci is null in d2 only",
    "A4": "nothing else: dim d0=2, d2=3 for both special_hp, raw and tied alike",
    "A5": "P3 mostly wrong: survivors are initiation-vs-mismatch trades; dangles vs terminal "
          "mismatches not null; no coaxial terms in the model",
}

# ----------------------------------------------------------------------------------
# encodings, as ViennaRNA's
# ----------------------------------------------------------------------------------
BASES = "ACGU"
BCODE = {"A": 1, "C": 2, "G": 3, "U": 4}
PAIRS = ["CG", "GC", "GU", "UG", "AU", "UA"]  # pair types 1..6
PTYPE = {p: k + 1 for k, p in enumerate(PAIRS)}
PNAME = {k + 1: p for k, p in enumerate(PAIRS)}
DEF = -50
INF = 10_000_000


def ptype(a: str, b: str) -> int:
    t = PTYPE.get(a + b)
    if t is None:
        raise ValueError(f"non-canonical pair {a}-{b}")
    return t


# ----------------------------------------------------------------------------------
# parameter file
# ----------------------------------------------------------------------------------
def dump_params() -> pathlib.Path:
    """Write ViennaRNA's active Turner 2004 table to a temporary .par file."""
    RNA.params_load_RNA_Turner2004()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="turner_nullspace_")) / "turner2004.par"
    RNA.params_save(str(tmp))
    return tmp


def _tok(x: str) -> int | float:
    if x == "INF":
        return INF
    if x == "DEF":
        return DEF
    v = float(x)
    return int(v) if v.is_integer() else v


def parse_par(path: pathlib.Path) -> dict:
    """Parse a v2.0 parameter file into {section: [(label, [tokens...]), ...]}.

    Block labels (the `/* CG.AA..CG */` comments) are kept so that indices are
    read from the file rather than assumed from block order.
    """
    sections: dict[str, list] = {}
    cur = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("##"):
            continue
        if line.startswith("#"):
            cur = line[1:].strip()
            sections[cur] = []
            continue
        if cur is None or not line:
            continue
        if line.startswith("/*"):
            label = line.strip("/* ").strip()
            sections[cur].append([label, []])
            continue
        if not sections[cur]:
            sections[cur].append([None, []])
        sections[cur][-1][1].extend(line.split())
    return sections


def _pair_or_none(lbl: str) -> int | None:
    return PTYPE.get(lbl)


def _base_idx(c: str) -> int:
    return 0 if c == "@" else BCODE.get(c, -1)


def theta_from_par(path: pathlib.Path) -> dict:
    """Column key -> value (dcal/mol, ints) for every canonical-reachable entry."""
    s = parse_par(path)
    th: dict[tuple, int] = {}

    def flat(name):
        out = []
        for _, toks in s[name]:
            out.extend(toks)
        return out

    # stack: 7x7, pair types 1..7, header comment is a column legend
    v = [_tok(x) for x in flat("stack")]
    assert len(v) == 49, len(v)
    for a in range(1, 7):
        for b in range(1, 7):
            th[("stack", a, b)] = v[(a - 1) * 7 + (b - 1)]

    for sec, key in [("mismatch_hairpin", "mismatchH"), ("mismatch_internal", "mismatchI"),
                     ("mismatch_internal_1n", "mismatch1nI"),
                     ("mismatch_internal_23", "mismatch23I"),
                     ("mismatch_multi", "mismatchM"), ("mismatch_exterior", "mismatchExt")]:
        v = [_tok(x) for x in flat(sec)]
        assert len(v) == 175, (sec, len(v))
        for t in range(1, 7):
            for i in range(1, 5):
                for j in range(1, 5):
                    th[(key, t, i, j)] = v[(t - 1) * 25 + i * 5 + j]

    for sec in ("dangle5", "dangle3"):
        v = [_tok(x) for x in flat(sec)]
        assert len(v) == 35, (sec, len(v))
        for t in range(1, 7):
            for i in range(1, 5):
                th[(sec, t, i)] = v[(t - 1) * 5 + i]

    for lbl, toks in s["int11"]:  # "CG..GC"
        a, b = lbl.split("..")
        ta, tb = _pair_or_none(a), _pair_or_none(b)
        if ta is None or tb is None:
            continue
        v = [_tok(x) for x in toks]
        assert len(v) == 25
        for i in range(1, 5):
            for j in range(1, 5):
                th[("int11", ta, tb, i, j)] = v[i * 5 + j]

    for lbl, toks in s["int21"]:  # "CG.A..CG"
        left, b = lbl.split("..")
        a, x = left.split(".")
        ta, tb, xi = _pair_or_none(a), _pair_or_none(b), _base_idx(x)
        if ta is None or tb is None or xi < 1:
            continue
        v = [_tok(z) for z in toks]
        assert len(v) == 25
        for i in range(1, 5):
            for j in range(1, 5):
                th[("int21", ta, tb, xi, i, j)] = v[i * 5 + j]

    for lbl, toks in s["int22"]:  # "CG.AA..CG", 4x4 with no N row/col
        left, b = lbl.split("..")
        a, xy = left.split(".")
        ta, tb = _pair_or_none(a), _pair_or_none(b)
        if ta is None or tb is None:
            continue
        xi, yi = _base_idx(xy[0]), _base_idx(xy[1])
        v = [_tok(z) for z in toks]
        assert len(v) == 16
        for k in range(1, 5):
            for m in range(1, 5):
                th[("int22", ta, tb, xi, yi, k, m)] = v[(k - 1) * 4 + (m - 1)]

    for sec, key, lo in [("hairpin", "hairpin", 3), ("bulge", "bulge", 1),
                         ("internal", "internal_loop", 4)]:
        v = [_tok(x) for x in flat(sec)]
        assert len(v) == 31, (sec, len(v))
        for u in range(lo, 31):
            th[(key, u)] = v[u]

    v = [_tok(x) for x in flat("ML_params")]
    th[("ML_cu",)], th[("ML_cc",)], th[("ML_ci",)] = v[0], v[2], v[4]
    v = [_tok(x) for x in flat("NINIO")]
    th[("ninio_m",)], th[("ninio_max",)] = v[0], v[2]
    v = [_tok(x) for x in flat("Misc")]
    th[("TerminalAU",)] = v[2]
    lxc = float(v[4])

    special = {}
    for sec, key in [("Tetraloops", "Tetraloop"), ("Triloops", "Triloop"),
                     ("Hexaloops", "Hexaloop")]:
        toks = flat(sec)
        lst = []
        for k in range(0, len(toks), 3):
            seqm, e = toks[k], _tok(toks[k + 1])
            th[(key, seqm)] = e
            lst.append(seqm)
        special[key] = lst
    return {"theta": th, "lxc": lxc, "special": special}


# ----------------------------------------------------------------------------------
# count-vector extractor mirroring ViennaRNA 2.x loop evaluation
# ----------------------------------------------------------------------------------
def pair_table(db: str) -> list[int]:
    pt = [0] * (len(db) + 1)
    st = []
    for k, c in enumerate(db, start=1):
        if c == "(":
            st.append(k)
        elif c == ")":
            i = st.pop()
            pt[i], pt[k] = k, i
    assert not st
    return pt


class Extractor:
    def __init__(self, special: dict, dangles: int, special_hp: int, planted_defect=False,
                 lxc: float | None = None):
        self.special = special
        self.d = dangles
        self.sp = special_hp
        self.planted = planted_defect
        # Loops longer than MAXLOOP=30 are allowed only when lxc is given. ViennaRNA
        # prices them as table[30] + (int)(lxc*log(u/30)); the truncation is not
        # linear in lxc, so it is returned as a fixed offset under the key
        # ("_offset",), never as a column. The generating set never sets lxc.
        self.lxc = lxc

    def _size(self, table, u, add):
        if u <= 30:
            add((table, u))
            return
        assert self.lxc is not None, f"{table} of size {u} > 30 needs lxc"
        add((table, 30))
        add(("_offset",), int(self.lxc * math.log(u / 30.0)))

    def counts(self, seq: str, db: str) -> dict:
        n = len(seq)
        pt = pair_table(db)
        S = [0] + [BCODE[c] for c in seq] + [0]
        c: dict[tuple, int] = {}

        def add(k, v=1):
            c[k] = c.get(k, 0) + v

        def tau(i):  # TerminalAU if the pair type is AU/UA/GU/UG
            return 1 if i > 2 else 0

        # exterior loop
        i = 1
        while i <= n:
            if pt[i] > i:
                j = pt[i]
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
                i = j + 1
            else:
                i += 1

        # every loop closed by a pair
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
            if not inner:
                self._hairpin(seq, S, i, j, t, add, tau)
            elif len(inner) == 1:
                self._interior(seq, S, i, j, inner[0], t, add, tau)
            else:
                self._multi(seq, S, i, j, inner, add, tau)
        return c

    def _hairpin(self, seq, S, i, j, t, add, tau):
        u = j - i - 1
        assert u >= 3, u
        if self.sp:
            if u == 4 and seq[i - 1:j] in self.special["Tetraloop"]:
                add(("Tetraloop", seq[i - 1:j]))
                return
            if u == 6 and seq[i - 1:j] in self.special["Hexaloop"]:
                add(("Hexaloop", seq[i - 1:j]))
                return
            if u == 3:
                if seq[i - 1:j] in self.special["Triloop"]:
                    add(("Triloop", seq[i - 1:j]))
                    return
                add(("hairpin", 3))
                if tau(t):
                    add(("TerminalAU",))
                return
        self._size("hairpin", u, add)
        add(("mismatchH", t, S[i + 1], S[j - 1]))
        if self.planted and u == 7:
            add(("mismatchH", t, S[i + 1], S[j - 1]))  # C4: a deliberately broken identity

    def _interior(self, seq, S, i, j, pq, t, add, tau):
        p, q = pq
        n1, n2 = p - i - 1, j - q - 1
        t2 = ptype(seq[q - 1], seq[p - 1])  # rtype of the inner pair
        si1, sj1, sp1, sq1 = S[i + 1], S[j - 1], S[p - 1], S[q + 1]
        nl, ns = max(n1, n2), min(n1, n2)
        if nl == 0:
            add(("stack", t, t2))
            return
        if ns == 0:
            self._size("bulge", nl, add)
            if nl == 1:
                add(("stack", t, t2))
            else:
                if tau(t):
                    add(("TerminalAU",))
                if tau(t2):
                    add(("TerminalAU",))
            return
        if ns == 1:
            if nl == 1:
                add(("int11", t, t2, si1, sj1))
                return
            if nl == 2:
                if n1 == 1:
                    add(("int21", t, t2, si1, sq1, sj1))
                else:
                    add(("int21", t2, t, sq1, si1, sp1))
                return
            self._size("internal_loop", nl + 1, add)
            self._ninio(nl - ns, add)
            add(("mismatch1nI", t, si1, sj1))
            add(("mismatch1nI", t2, sq1, sp1))
            return
        if ns == 2:
            if nl == 2:
                add(("int22", t, t2, si1, sp1, sq1, sj1))
                return
            if nl == 3:
                add(("internal_loop", 5))
                add(("ninio_m",))
                add(("mismatch23I", t, si1, sj1))
                add(("mismatch23I", t2, sq1, sp1))
                return
        self._size("internal_loop", nl + ns, add)
        self._ninio(nl - ns, add)
        add(("mismatchI", t, si1, sj1))
        add(("mismatchI", t2, sq1, sp1))

    @staticmethod
    def _ninio(asym, add):
        # MIN2(MAX_NINIO, asym*m) linearised at m=60, max=300; at asym=5 the macro
        # returns its second argument
        if asym == 0:
            return
        if asym * 60 <= 300:
            add(("ninio_m",), asym)
        else:
            add(("ninio_max",))

    def _multi(self, seq, S, i, j, inner, add, tau):
        add(("ML_cc",))
        unp = (inner[0][0] - i - 1) + (j - inner[-1][1] - 1)
        unp += sum(inner[k + 1][0] - inner[k][1] - 1 for k in range(len(inner) - 1))
        if unp:
            add(("ML_cu",), unp)
        stems = [(ptype(seq[p - 1], seq[q - 1]), S[p - 1], S[q + 1]) for p, q in inner]
        stems.append((ptype(seq[j - 1], seq[i - 1]), S[j - 1], S[i + 1]))
        for tt, m5, m3 in stems:
            if self.d == 2:
                add(("mismatchM", tt, m5, m3))
            if tau(tt):
                add(("TerminalAU",))
            add(("ML_ci",))


# ----------------------------------------------------------------------------------
# generating set: deterministic constructions spanning the model's domain
# ----------------------------------------------------------------------------------
class Node:
    """A sequence/structure fragment."""

    __slots__ = ("seq", "db")

    def __init__(self, seq, db):
        self.seq, self.db = seq, db


def U(s: str) -> Node:
    return Node(s, "." * len(s))


def cat(*parts: Node) -> Node:
    return Node("".join(p.seq for p in parts), "".join(p.db for p in parts))


def H(pairs: list[str], inner: Node) -> Node:
    """A helix of the given pairs, outermost first, enclosing `inner`."""
    left = "".join(p[0] for p in pairs)
    right = "".join(p[1] for p in reversed(pairs))
    return Node(left + inner.seq + right, "(" * len(pairs) + inner.db + ")" * len(pairs))


CAP = U("GAAA")  # a fixed hairpin body; whatever it costs is the same in every row
CLAMP = ["GC", "CG"]


def seg(n: int, first: str | None = None, last: str | None = None) -> str:
    """n unpaired nt with a fixed first and/or last nucleotide, deterministic filler."""
    s = list(("AAC" * 20)[:n])
    if first is not None:
        s[0] = first
    if last is not None:
        s[-1] = last
    if n == 1 and first is not None and last is not None:
        assert first == last, "one nucleotide cannot carry two different constraints"
    return "".join(s)


def generating_set(special: dict) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    def emit(tag, node, flank5="A", flank3="A"):
        s = cat(U(flank5), node, U(flank3)) if (flank5 or flank3) else node
        rows.append((tag, s.seq, s.db))

    # hairpins: every closing pair x first/last mismatch x sizes 3..30
    for t in PAIRS:
        for a, b in itertools.product(BASES, BASES):
            for u in (3, 4, 5, 6, 7, 9):
                emit("hairpin", H(CLAMP + [t], U(seg(u, a, b))))
        for u in range(3, 31):
            emit("hairpin_size", H(CLAMP + [t], U(seg(u, "G", "A"))))
    for key in ("Tetraloop", "Triloop", "Hexaloop"):
        for m in special[key]:
            pair = m[0] + m[-1]
            if pair in PTYPE:
                emit("special_hairpin", H(CLAMP + [pair], U(m[1:-1])))

    # stacks: every ordered pair-type combination, and helices of varied length
    for a, b in itertools.product(PAIRS, PAIRS):
        emit("stack", H(CLAMP + [a, b] + CLAMP, CAP))
    for k in range(1, 9):
        emit("helix_len", H(["GC"] * k + ["CG"], CAP))
        emit("helix_len", H(["AU"] * k + ["GC"], CAP))

    def interior(t1, t2, left: str, right: str):
        # outer helix ending in pair t1, loop, inner helix starting with pair t2
        inner = H([t2] + CLAMP, CAP)
        return H(CLAMP + [t1], cat(U(left), inner, U(right)))

    # 1x1, 1x2 / 2x1, 2x2: full tables
    for t1, t2 in itertools.product(PAIRS, PAIRS):
        for x, y in itertools.product(BASES, BASES):
            emit("int11", interior(t1, t2, x, y))
        for x, y, z in itertools.product(BASES, BASES, BASES):
            emit("int21", interior(t1, t2, x, y + z))
            if (t1, t2) == ("CG", "AU"):  # the other orientation, as a cross-check only
                emit("int12", interior(t1, t2, x + y, z))
        for w, x, y, z in itertools.product(BASES, BASES, BASES, BASES):
            emit("int22", interior(t1, t2, w + x, y + z))

    # 1xn, 2x3, generic: mismatch coverage on both closing pairs, all sizes
    shapes_1n = [(1, n) for n in range(3, 30)] + [(n, 1) for n in (3, 4, 7)]
    shapes_23 = [(2, 3), (3, 2)]
    shapes_gen = [(2, 4), (4, 2), (3, 3), (3, 4), (2, 5), (4, 4), (3, 8), (2, 10), (5, 10),
                  (6, 6), (2, 8), (3, 9), (9, 3), (5, 5), (4, 12), (14, 14), (6, 24), (2, 28)]
    for shape in shapes_1n + shapes_23 + shapes_gen:
        n1, n2 = shape
        for t in PAIRS:
            for a, b in itertools.product(BASES, BASES):
                # outer closing pair t sees (si1=a, sj1=b)
                emit("il_outer", interior(t, "GC", seg(n1, first=a), seg(n2, last=b)))
                # inner pair t sees (sq1=a, sp1=b): left loop ends in b, right starts with a
                emit("il_inner", interior("GC", t, seg(n1, last=b), seg(n2, first=a)))
    for n1 in range(2, 16):
        for n2 in range(2, 31 - n1):
            emit("il_size", interior("CG", "AU", seg(n1, "G", "A"), seg(n2, "A", "G")))
    # bulges
    for nb in range(1, 31):
        for t1, t2 in itertools.product(PAIRS, PAIRS):
            if nb > 3 and (t1, t2) not in (("CG", "GC"), ("AU", "UA"), ("GU", "CG")):
                continue
            emit("bulge", interior(t1, t2, "A" * nb, ""))
            emit("bulge", interior(t1, t2, "", "A" * nb))

    # multiloops: closing pair x branch pairs x neighbours x unpaired counts
    def branch(t):
        return H([t] + CLAMP, CAP)

    for tc in PAIRS:
        for a, b in itertools.product(BASES, BASES):
            # closing pair tc sees (S[j-1]=a, S[i+1]=b): first unpaired b, last unpaired a
            emit("ml_closing", H(CLAMP + [tc], cat(U(b), branch("GC"), U("A"), branch("CG"), U(a))))
            # a branch of type tc sees (S[p-1]=a, S[q+1]=b)
            emit("ml_branch", H(CLAMP + ["GC"], cat(U(a), branch(tc), U(b), branch("CG"), U("A"))))
    for k in (2, 3, 4):
        for gap in (0, 1, 2, 5):
            body = [U("A" * gap)]
            for m in range(k):
                body += [branch(PAIRS[m % 6]), U("A" * gap)]
            emit("ml_degree", H(CLAMP + ["CG"], cat(*body)))
    # r2: gaps whose first and last nucleotides differ, and directly adjacent stems.
    # Without these every multiloop gap is one nucleotide or all-A, and the r1 run
    # found the spurious d2 direction mismatchM[t][m5][m3] += g(m3) - g(m5) (g(A)=1,
    # else 0), which cancels around any loop built only from such gaps.
    for tb in PAIRS:
        for x, y in itertools.product(BASES, BASES):
            emit("ml_gap2", H(CLAMP + ["CG"], cat(U("A"), branch(tb), U(x + y), branch("GC"),
                                                   U("A"))))
        for t2 in PAIRS:
            emit("ml_adjacent", H(CLAMP + ["CG"], cat(U("C"), branch(tb), branch(t2), U("G"))))

    # exterior loop: stems with both, one or no neighbours, several stems
    for t in PAIRS:
        for a, b in itertools.product(BASES, BASES):
            emit("ext_mm", H([t] + CLAMP, CAP), flank5=a, flank3=b)
        for a in BASES:
            emit("ext_d5", H([t] + CLAMP, CAP), flank5=a, flank3="")
            emit("ext_d3", H([t] + CLAMP, CAP), flank5="", flank3=a)
        emit("ext_none", H([t] + CLAMP, CAP), flank5="", flank3="")
        emit("ext_two", cat(H([t] + CLAMP, CAP), U("A"), H(["GC"] + CLAMP, CAP)), "", "")
        emit("ext_two", cat(H(["CG"] + CLAMP, CAP), H([t] + CLAMP, CAP)), "C", "")
    return rows


# ----------------------------------------------------------------------------------
# linear algebra
# ----------------------------------------------------------------------------------
def build_matrix(rows_counts: list[dict], cols: list[tuple]):
    idx = {k: n for n, k in enumerate(cols)}
    data, ri, ci = [], [], []
    for r, cnt in enumerate(rows_counts):
        for k, v in cnt.items():
            if v:
                data.append(v)
                ri.append(r)
                ci.append(idx[k])
    return sp.csr_matrix((np.array(data, dtype=np.int64), (ri, ci)),
                         shape=(len(rows_counts), len(cols)))


def eigh_null(D: np.ndarray):
    """Null space of a dense matrix via eigh of D^T D (float64)."""
    if D.shape[1] == 0:
        return np.zeros((0, 0)), {"eig_tol": None}
    G = D.T @ D
    w, V = np.linalg.eigh(G)
    tol = max(G.shape) * np.finfo(float).eps * max(w.max(), 1.0)
    null = w < tol
    return V[:, null], {
        "eig_tol": float(tol),
        "largest_null_eig": float(w[null].max()) if null.any() else None,
        "smallest_nonnull_eig": float(w[~null].min()) if (~null).any() else None,
        "largest_eig": float(w.max()),
    }


def eliminate_singletons(M: sp.csr_matrix):
    """Exact reduction. A column that is nonzero in exactly one live row r takes its
    null-vector entry from row r alone, so deleting (r, column) preserves the
    null-space dimension and the rest of every null vector. Repeated to a fixpoint.
    Returns the order of eliminations for back-substitution."""
    Mr, Mc = M.tocsr(), M.tocsc()
    alive_r = np.ones(M.shape[0], bool)
    alive_c = np.ones(M.shape[1], bool)
    col_nnz = np.diff(Mc.indptr).astype(np.int64)
    work = list(np.nonzero(col_nnz == 1)[0])
    order = []
    while work:
        c = work.pop()
        if not alive_c[c] or col_nnz[c] != 1:
            continue
        rs = Mc.indices[Mc.indptr[c]:Mc.indptr[c + 1]]
        live = [r for r in rs if alive_r[r]]
        if len(live) != 1:
            continue
        r = live[0]
        alive_r[r] = False
        alive_c[c] = False
        order.append((int(r), int(c)))
        for c2 in Mr.indices[Mr.indptr[r]:Mr.indptr[r + 1]]:
            if alive_c[c2]:
                col_nnz[c2] -= 1
                if col_nnz[c2] == 1:
                    work.append(c2)
    keep_r, keep_c = np.nonzero(alive_r)[0], np.nonzero(alive_c)[0]
    return Mr[keep_r][:, keep_c], order, keep_c


def null_space(M: sp.csr_matrix):
    """Null basis of M in full coordinates: singleton elimination (exact), eigh on the
    reduced matrix, then back-substitution of the eliminated entries."""
    red, order, keep_c = eliminate_singletons(M)
    D = red.toarray().astype(np.float64)
    Nred, info = eigh_null(D)
    k = Nred.shape[1] if Nred.size else 0
    N = np.zeros((M.shape[1], k))
    if k:
        N[keep_c] = Nred
        Mr = M.tocsr()
        for r, c in reversed(order):
            row = Mr[r]
            N[c] = -(row @ N) / Mr[r, c]  # N[c] is still zero here
        N, _ = np.linalg.qr(N)
    info.update({"singleton_columns_eliminated": len(order), "reduced_shape": list(D.shape),
                 "null_dim": k})
    return N, info, red


def full_eigh_null_dim(M: sp.csr_matrix) -> dict:
    """Independent cross-check with no elimination: eigh of the full M^T M."""
    t0 = time.time()
    N, info = eigh_null(M.astype(np.float64).toarray())
    return dict(info, null_dim=int(N.shape[1]), seconds=round(time.time() - t0, 1))


def rank_mod_p(A: np.ndarray, p: int) -> int:
    """Exact rank of an integer matrix over GF(p) by Gaussian elimination."""
    A = np.mod(A.astype(np.int64), p)
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        piv = np.nonzero(A[r:, c])[0]
        if piv.size == 0:
            continue
        k = r + piv[0]
        if k != r:
            A[[r, k]] = A[[k, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        nz = np.nonzero(A[:, c])[0]
        nz = nz[nz != r]
        if nz.size:
            A[nz] = (A[nz] - np.outer(A[nz, c], A[r])) % p
        r += 1
    return r


def loadings(v: np.ndarray, cols: list[tuple], top: int = 6) -> dict:
    """Weight of a null vector per parameter table, plus its largest entries."""
    by_table: dict[str, float] = {}
    for k, x in zip(cols, v, strict=True):
        by_table[k[0]] = by_table.get(k[0], 0.0) + float(x * x)
    order = np.argsort(-np.abs(v))[:top]
    return {
        "weight_by_table": {k: round(w, 6) for k, w in sorted(by_table.items(),
                                                               key=lambda z: -z[1]) if w > 1e-12},
        "largest_entries": [[repr(cols[i]), float(v[i])] for i in order],
    }


def analytic_directions(cols: list[tuple], dangles: int, special_hp: int) -> dict:
    idx = {k: n for n, k in enumerate(cols)}
    out = {}

    def vec(spec):
        g = np.zeros(len(cols))
        for k, x in spec:
            if k in idx:
                g[idx[k]] += x
        return g

    n1 = [(k, 1.0) for k in cols if k[0] == "mismatchH"]
    n1 += [(("hairpin", u), -1.0) for u in range(4 if special_hp else 3, 31)]
    out["N1_hairpin_mismatch_vs_init"] = vec(n1)
    n2 = [(k, 1.0) for k in cols if k[0] in ("mismatchI", "mismatch1nI", "mismatch23I")]
    n2 += [(("internal_loop", u), -2.0) for u in range(4, 31)]
    out["N2_interior_mismatch_vs_init"] = vec(n2)
    if dangles == 2:
        n3 = [(k, 1.0) for k in cols if k[0] == "mismatchM"] + [(("ML_ci",), -1.0)]
        out["N3_multiloop_mismatch_vs_ci"] = vec(n3)
    return out


def tie_map(cols: list[tuple]) -> tuple[list[tuple], np.ndarray]:
    """Merge orientation-equivalent entries (Krueger's 'minimal symmetries')."""

    def canon(k):
        if k[0] == "stack":
            return ("stack",) + tuple(sorted(k[1:]))
        if k[0] == "int11":
            _, a, b, x, y = k
            return min(k, ("int11", b, a, y, x))
        if k[0] == "int22":
            _, a, b, i, j, kk, m = k
            return min(k, ("int22", b, a, kk, m, i, j))
        return k

    tied = sorted({canon(k) for k in cols}, key=repr)
    tidx = {k: n for n, k in enumerate(tied)}
    T = np.zeros((len(cols), len(tied)))
    for n, k in enumerate(cols):
        T[n, tidx[canon(k)]] = 1.0
    return tied, T


def principal_overlap(B: np.ndarray, A: np.ndarray) -> float:
    """Distance between span(A) and span(B): the largest norm of an orthonormal
    vector of span(A) left over after projecting onto span(B), and vice versa
    (0 = same span; equal to the largest principal-angle sine).

    Computed as a projection residual rather than sqrt(1 - s_min^2) from the SVD of
    Qa^T Qb: the latter cancels catastrophically near equal spans and cannot report
    anything below ~1.5e-8 (sqrt of machine epsilon), which is coarser than the
    1e-8 bar it is compared with. The bar is unchanged; the instrument now resolves it.
    """
    if A.shape[1] != B.shape[1]:
        return float("nan")
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    ra = np.linalg.norm(Qa - Qb @ (Qb.T @ Qa), axis=0).max()
    rb = np.linalg.norm(Qb - Qa @ (Qa.T @ Qb), axis=0).max()
    return float(max(ra, rb))


# ----------------------------------------------------------------------------------
def write(out: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str))


def analyse_config(dangles, special_hp, theta, special, rows, rng, full_check=True):
    md = reference_md()
    md.dangles = dangles
    md.special_hp = special_hp
    assert md.dangles == dangles and md.special_hp == special_hp

    ex = Extractor(special, dangles, special_hp)
    counts, energies, failures = [], [], []
    fc_cache = None
    for tag, seq, db in rows:
        fc = RNA.fold_compound(seq, md)
        if fc_cache is None:
            assert_turner2004_params(fc)
            fc_cache = True
        e = fc.eval_structure(db)
        e_int = int(round(e * 100))
        c = ex.counts(seq, db)
        recon = sum(theta[k] * v for k, v in c.items())
        if recon != e_int:
            failures.append({"tag": tag, "seq": seq, "db": db, "vienna": e_int, "recon": recon})
        counts.append(c)
        energies.append(e_int)

    res: dict = {"dangles": dangles, "special_hp": special_hp, "n_rows": len(rows)}
    res["reconstruction"] = {
        "n_checked": len(rows),
        "n_failed": len(failures),
        "units": "integer 0.01 kcal/mol; exact equality required",
        "first_failures": failures[:10],
        "passes": not failures,
    }
    if failures:
        return res, None

    # C0: the reconstruction comparison must be able to fail. Corrupt one entry by
    # 0.01 kcal/mol; exactly the rows that use it must stop matching ViennaRNA.
    c0 = {}
    for probe in [("stack", 1, 2), ("mismatchH", 5, 3, 1), ("int22", 2, 1, 3, 1, 4, 2),
                  ("ML_ci",), ("TerminalAU",)]:
        bad = dict(theta)
        bad[probe] = bad[probe] + 1
        n_fail = sum(1 for c, e in zip(counts, energies, strict=True)
                     if sum(bad[k] * v for k, v in c.items()) != e)
        n_use = sum(1 for c in counts if c.get(probe, 0))
        c0[repr(probe)] = {"rows_using": n_use, "rows_failing": n_fail,
                           "detected": n_fail == n_use > 0}
    res["C0_corrupted_entry_detected"] = c0

    used = sorted({k for c in counts for k in c}, key=repr)
    reachable = [k for k in sorted(theta, key=repr) if _reachable(k, dangles, special_hp)]
    never = [k for k in reachable if k not in set(used)]
    unexpected = [k for k in used if k not in set(reachable)]
    res["columns"] = {
        "n_reachable": len(reachable),
        "n_used_by_generating_set": len(used),
        "reachable_never_used": [repr(k) for k in never[:50]],
        "n_reachable_never_used": len(never),
        "used_but_not_declared_reachable": [repr(k) for k in unexpected],
    }
    cols = reachable
    M = build_matrix(counts, cols)
    theta_vec = np.array([theta[k] for k in cols], dtype=np.float64) / 100.0  # kcal
    e_kcal = np.array(energies, dtype=np.float64) / 100.0
    res["reconstruction"]["max_abs_kcal_float"] = float(np.abs(M @ theta_vec - e_kcal).max())

    t0 = time.time()
    Nb, spec, red = null_space(M)
    res["eigh_reduced"] = dict(spec, seconds=round(time.time() - t0, 1),
                               rank=len(cols) - int(Nb.shape[1]))
    if full_check:
        res["eigh_full_crosscheck"] = full_eigh_null_dim(M)
        res["eigh_full_crosscheck"]["agrees"] = (
            res["eigh_full_crosscheck"]["null_dim"] == Nb.shape[1])

    dense = red.toarray()
    ranks = {str(p): rank_mod_p(dense, p) for p in (2_147_483_647, 1_000_000_007)}
    res["exact_rank"] = {
        "method": "singleton elimination, then Gaussian elimination over GF(p), two primes",
        "rank_mod_p": ranks,
        "null_dim": dense.shape[1] - max(ranks.values()),
        "rank": spec["singleton_columns_eliminated"] + max(ranks.values()),
        "agrees_with_eigh": dense.shape[1] - max(ranks.values()) == Nb.shape[1],
    }

    # analytic directions: each must be null on every row (A1-A3) and together span
    an = analytic_directions(cols, dangles, special_hp)
    an_res = {}
    for name, g in an.items():
        an_res[name] = {"max_abs_row_change": float(np.abs(M @ g).max()),
                        "null_to_1e-9": bool(np.abs(M @ g).max() <= EXACT_ROW_TOL_KCAL)}
    A = np.stack(list(an.values()), axis=1)
    res["analytic"] = {
        "directions": an_res,
        "n_directions": A.shape[1],
        "dim_equals_null_dim": A.shape[1] == Nb.shape[1],
        "span_mismatch_sine": principal_overlap(Nb, A),
    }

    # C2: every numerical null basis vector is null on every row
    MN = np.abs(M @ Nb)
    res["C2_null_basis_rows_unchanged"] = {
        "max_abs_row_change": float(MN.max()) if Nb.shape[1] else 0.0,
        "tolerance": EXACT_ROW_TOL_KCAL,
        "passes": bool(MN.max() <= EXACT_ROW_TOL_KCAL) if Nb.shape[1] else True,
    }
    # basis loadings, reported on the interpretable analytic basis when it spans
    res["null_basis_loadings"] = {name: loadings(g / np.linalg.norm(g), cols)
                                  for name, g in an.items()}

    # C1: random direction with null component removed must change some row
    g = rng.standard_normal(len(cols))
    if Nb.shape[1]:
        g -= Nb @ (Nb.T @ g)
    ch = float(np.abs(M @ g).max())
    res["C1_random_nonnull_changes_energy"] = {
        "max_abs_row_change": ch, "threshold": NONNULL_MIN_CHANGE,
        "fails_as_required": ch > NONNULL_MIN_CHANGE}

    # P2: uniform stack shift
    gs = np.array([1.0 if k[0] == "stack" else 0.0 for k in cols])
    st = M @ gs
    res["P2_uniform_stack_shift"] = {
        "max_abs_row_change": float(np.abs(st).max()),
        "distinct_row_changes": sorted({int(x) for x in st})[:20],
        "in_null_space": bool(np.abs(st).max() <= EXACT_ROW_TOL_KCAL),
        "component_outside_null_space": float(np.linalg.norm(gs - Nb @ (Nb.T @ gs))
                                              / np.linalg.norm(gs)),
    }

    # C3: per-row shuffle of motif labels vs a pure column permutation
    Mc = M.tocoo()
    shuffled_cols = rng.integers(0, len(cols), size=Mc.nnz)
    Ms = sp.csr_matrix((Mc.data, (Mc.row, shuffled_cols)), shape=M.shape)
    Ns, _, _ = null_space(Ms)
    perm = rng.permutation(len(cols))
    Np, _, _ = null_space(sp.csr_matrix(M[:, perm]))
    res["C3_label_shuffle"] = {
        "null_dim_unshuffled": int(Nb.shape[1]),
        "null_dim_row_shuffled": int(Ns.shape[1]),
        "row_shuffle_changes_dim": int(Ns.shape[1]) != int(Nb.shape[1]),
        "null_dim_column_permuted": int(Np.shape[1]),
        "column_permutation_leaves_dim_unchanged": int(Np.shape[1]) == int(Nb.shape[1]),
        "note": "rank is permutation invariant, so only the within-row shuffle is a control",
    }

    # C4: planted defect -- N1 must stop being null
    exd = Extractor(special, dangles, special_hp, planted_defect=True)
    Md = build_matrix([exd.counts(s, d) for _, s, d in rows], cols)
    g1 = an["N1_hairpin_mismatch_vs_init"]
    dchg = float(np.abs(Md @ g1).max())
    res["C4_planted_defect"] = {
        "N1_max_abs_row_change_under_defect": dchg,
        "detected": dchg > NONNULL_MIN_CHANGE,
    }

    # C5 (d2 only): the spurious direction the r1 generating set left null must now
    # change some row, or the enriched multiloop constructions did not do their job
    if dangles == 2:
        idx = {k: n for n, k in enumerate(cols)}
        g5 = np.zeros(len(cols))
        for t in range(1, 7):
            for x in (2, 3, 4):  # C, G, U; A is 1
                g5[idx[("mismatchM", t, x, 1)]] += 1.0
                g5[idx[("mismatchM", t, 1, x)]] -= 1.0
        ch5 = float(np.abs(M @ g5).max())
        res["C5_r1_spurious_direction_broken"] = {
            "max_abs_row_change": ch5, "broken": ch5 > NONNULL_MIN_CHANGE}

    # tied parameterisation
    tied, T = tie_map(cols)
    Mt = sp.csr_matrix(M @ sp.csr_matrix(T))
    Nt, spt, _ = null_space(Mt)
    res["tied"] = {
        "n_columns": len(tied),
        "null_dim": int(Nt.shape[1]),
        "rank": len(tied) - int(Nt.shape[1]),
        "eigh": spt,
        "C2_max_abs_row_change": float(np.abs(Mt @ Nt).max()) if Nt.shape[1] else 0.0,
    }
    res["null_dim"] = int(Nb.shape[1])
    res["rank"] = len(cols) - int(Nb.shape[1])
    res["shape"] = list(M.shape)
    return res, M


def _reachable(k: tuple, dangles: int, special_hp: int) -> bool:
    name = k[0]
    if name in ("mismatchM", "mismatchExt", "dangle5", "dangle3"):
        return dangles == 2
    if name in ("Tetraloop", "Hexaloop", "Triloop"):
        return special_hp == 1 and (k[1][0] + k[1][-1]) in PTYPE
    return True


def main():
    t_start = time.time()
    out = {"run_id": RUN_ID, "status": "PRE-REGISTERED, not yet run", "predictions": PRED,
           **git_provenance(R), "viennarna_version": RNA.__version__,
           "scope": "MODEL-INTRINSIC null space; the ArchiveII data null space is BLOCKED "
                    "(docs/BLOCKED.md item 27)"}
    write(out)

    par = dump_params()
    P = theta_from_par(par)
    theta, special = P["theta"], P["special"]
    fp = RNA.fold_compound("GGGGAAACCCC", reference_md()).params
    out["params"] = {"n_entries_parsed": len(theta), "lxc": P["lxc"],
                     "n_tetraloops": len(special["Tetraloop"]),
                     "n_triloops": len(special["Triloop"]),
                     "n_hexaloops": len(special["Hexaloop"]),
                     "SaltStack": int(fp.SaltStack),
                     "MLintern_array": list(fp.MLintern)}
    assert fp.SaltStack == 0, "salt correction active at the pinned salt"

    rows = generating_set(special)
    tags: dict[str, int] = {}
    for tag, _, _ in rows:
        tags[tag] = tags.get(tag, 0) + 1
    out["generating_set"] = {"n_rows": len(rows), "by_construction": tags,
                             "max_length": max(len(s) for _, s, _ in rows),
                             "nature": "deterministic constructions spanning the model's "
                                       "domain; not data, never fitted"}
    rng = np.random.default_rng(SHUFFLE_SEED)
    out["configs"] = {}
    for dangles in (0, 2):
        for special_hp in (1, 0):
            name = f"d{dangles}_special_hp{special_hp}"
            res, _ = analyse_config(dangles, special_hp, theta, special, rows, rng)
            out["configs"][name] = res
            write(out)
            if not res["reconstruction"]["passes"]:
                out["status"] = f"STOPPED: reconstruction failed in {name}"
                write(out)
                print(json.dumps(res["reconstruction"], indent=1)[:4000])
                return 1

    # verdicts
    v = {}
    cf = out["configs"]
    dims = {k: c["null_dim"] for k, c in cf.items()}
    v["P1"] = {"verdict": "confirmed" if all(d > 0 for d in dims.values()) else "refuted",
               "null_dims": dims}
    v["P2"] = {"verdict": "confirmed" if not any(c["P2_uniform_stack_shift"]["in_null_space"]
                                                 for c in cf.values()) else "refuted"}
    expected = {"d0_special_hp1": 2, "d0_special_hp0": 2, "d2_special_hp1": 3, "d2_special_hp0": 3}
    v["A1_A2_A3"] = {"verdict": "confirmed" if all(
        all(d["null_to_1e-9"] for d in c["analytic"]["directions"].values())
        for c in cf.values()) else "refuted"}
    v["A4"] = {"verdict": "confirmed" if (dims == expected and all(
        c["tied"]["null_dim"] == expected[k] and c["analytic"]["span_mismatch_sine"] < 1e-8
        for k, c in cf.items())) else "refuted",
        "expected": expected, "got_raw": dims,
        "got_tied": {k: c["tied"]["null_dim"] for k, c in cf.items()}}
    v["P3"] = {"verdict": "refuted in substance (see A5)" if v["A4"]["verdict"] == "confirmed"
               else "undetermined",
               "reason": "every surviving direction is a loop-initiation vs terminal-mismatch "
                         "trade; none involves per-unpaired increments, dangles vs mismatches, "
                         "or coaxial stacking (absent from the model)"}
    v["controls"] = {k: {"C1": c["C1_random_nonnull_changes_energy"]["fails_as_required"],
                         "C2": c["C2_null_basis_rows_unchanged"]["passes"],
                         "C3": c["C3_label_shuffle"]["row_shuffle_changes_dim"],
                         "C4": c["C4_planted_defect"]["detected"],
                         "C5": c.get("C5_r1_spurious_direction_broken", {}).get("broken"),
                         "C0": all(x["detected"]
                                   for x in c["C0_corrupted_entry_detected"].values()),
                         "exact_rank_agrees": c["exact_rank"]["agrees_with_eigh"],
                         "full_eigh_agrees": c.get("eigh_full_crosscheck", {}).get("agrees")}
                     for k, c in cf.items()}
    out["verdicts"] = v
    out["status"] = "complete"
    out["elapsed_seconds"] = round(time.time() - t_start, 1)
    write(out)
    print(json.dumps({"verdicts": v}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
