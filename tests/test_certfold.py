"""Certified folding: the per-loop feature map, which every certificate rests on.

The certificate charges each loop instance its own worst-case shift, so it needs a count vector
PER LOOP, not per structure. `scripts.certfold.LoopCounts` derives that from the Extractor count
model this repository already proves exact against ViennaRNA's `eval_structure` on all 23,741
generating rows. These tests hold the derivation to that model:

  * the sum over a structure's loops equals `Extractor.counts` exactly, keys and values, on
    constructed rows that exercise every `vrna_E_internal` case;
  * each loop's declared subcase agrees with the parameter-table entries that loop actually
    reads, which is the classification a `VRNA_DECOMP_PAIR_IL` callback has to recover from
    (i, j, k, l) alone, since stack, bulge, int11, int21, int22, 1xn, 2x3 and generic interior
    loops all arrive under that one decomposition code;
  * a planted defect in the decomposition is caught, so the identity is not vacuous.

The callback identity against a soft-constrained fold compound is the gate's G1 and lives in the
run artifact, not here: it needs ViennaRNA energies, while these tests need only the count model.
"""

import pathlib
import sys

import pytest

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.certfold import FOUR, LoopCounts, interior_sub  # noqa: E402
from scripts.turner_nullspace import generating_set  # noqa: E402
from scripts.turner_par import template  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

# the table each interior subcase must read, and nothing else of the four
SUB_TABLE = {"int11": "int11", "int21": "int21", "int22": "int22", "bulge": "bulge"}


@pytest.fixture(scope="module")
def pinned():
    P = template()
    return P, LoopCounts(P["special"], 0, 0, lxc=P["lxc"])


@pytest.fixture(scope="module")
def rows(pinned):
    """Generating-set rows, enough of them to reach every interior case."""
    P, _ = pinned
    gen = generating_set(P["special"])
    assert len(gen) > 20000, len(gen)
    return gen[::17]            # a deterministic stride, about 1,400 rows, seconds to check


def test_sum_over_loops_is_the_count_model(pinned, rows):
    """The identity the whole feature map rests on, on every strided generating row."""
    _, lc = pinned
    for _tag, seq, db in rows:
        lc.loops_checked(seq, db)          # raises GuardFailure on any mismatch


def test_every_interior_case_is_exercised(pinned, rows):
    """A green identity test proves nothing if the rows never build the hard loops."""
    _, lc = pinned
    seen = set()
    for _tag, seq, db in rows:
        for rec in lc.loops(seq, db):
            if rec["kind"] == "interior":
                seen.add(rec["sub"])
    assert seen >= {"stack", "bulge", "int11", "int21", "int22", "int1n", "int23", "generic"}, seen


def test_subcase_agrees_with_the_entries_the_loop_reads(pinned, rows):
    """`interior_sub` works from positions; the counts come from the energy model. They must
    agree, because the callback has only the positions and the certificate has only the counts."""
    _, lc = pinned
    for _tag, seq, db in rows:
        for rec in lc.loops(seq, db):
            if rec["kind"] != "interior":
                continue
            tables = {k[0] for k in rec["counts"]}
            sub = rec["sub"]
            assert sub == interior_sub(rec["i"], rec["j"], rec["k"], rec["l"])
            if sub in SUB_TABLE:
                assert SUB_TABLE[sub] in tables, (sub, tables)
            if sub == "stack":
                assert tables == {"stack"}, tables
            if sub in ("int1n", "int23", "generic"):
                assert "internal_loop" in tables, (sub, tables)
            # no interior loop may read a table the four-table projection drops silently
            assert not (tables & {"hairpin", "ML_cc", "ML_cu", "ML_ci"}), (sub, tables)


def test_only_interior_loops_touch_the_four_tables(pinned, rows):
    """The certificate's cheap callback returns 0 outside DECOMP_PAIR_IL. That is only exact if
    no hairpin, multiloop or exterior-stem instance reads one of the four tables B lives in."""
    _, lc = pinned
    for _tag, seq, db in rows:
        for rec in lc.loops(seq, db):
            if rec["kind"] == "interior":
                continue
            assert not ({k[0] for k in rec["counts"]} & set(FOUR)), (rec["kind"], rec["counts"])


def test_a_planted_defect_is_caught(pinned):
    """MUST FAIL: drop one loop's counts and the identity has to notice."""
    _, lc = pinned

    class Broken(type(lc)):
        def loops(self, seq, db):
            per = super().loops(seq, db)
            for rec in per:
                if rec["kind"] == "interior" and rec["counts"]:
                    rec["counts"].popitem()
                    break
            return per

    bad = Broken(lc.special, lc.d, lc.sp, lxc=lc.lxc)
    seq = "GGGCGCAAGCCCAAAGGGCUUGCGCCCAAUAGCAUCGAU"
    db = "((((((((((((...))))))))))))............"
    with pytest.raises(GuardFailure, match="per-loop decomposition"):
        bad.loops_checked(seq, db)


# ----------------------------------------------------------------------------------
# the forbidden-pair fold, which X' is
# ----------------------------------------------------------------------------------
def test_forbidding_a_pair_forbids_only_that_pair(pinned):
    """A forbidden-pair fold must reach the true minimum over the structures lacking the pair.

    This is a regression test for a real false-certificate bug, found by G2 and not by reading
    ViennaRNA's header. `vrna_hc_add_bp(i, j, CONSTRAINT_CONTEXT_NONE)` also removes the pairs
    that CROSS (i, j), which is right when a pair is being enforced and wrong when it is being
    forbidden: the minimisation then runs over too small a space, X' comes back too high, and
    pairs certify that are not robust. The fix is CONSTRAINT_CONTEXT_NO_REMOVE, and what this
    test pins is the consequence rather than the flag -- the fold must agree with brute-force
    enumeration, whatever ViennaRNA's constraint API does next.

    The sequence is the one G2 failed on, with the pair and radius that failed, so a revert
    fails here before it reaches the gate.
    """
    import RNA

    from scripts.certfold import (
        Folder,
        Weights,
        enumerate_structures,
        load_tr0_basis,
        loop_coeff_map,
        offset_from_map,
    )
    from scripts.turner_nullspace import pair_table
    from thermo.vienna_ref import reference_md

    seq, pair, r = "UGCUAAACUCAGGCUGCAUU", (1, 18), 20.0
    P, lc = pinned
    B = load_tr0_basis()
    W1 = Weights(B["a_four"], B["cols"], 1.0)
    plain = RNA.fold_compound(seq, reference_md())
    star, _ = plain.mfe()
    star_map = loop_coeff_map(lc, seq, star, W1)
    best = min(int(round(plain.eval_structure(s) * 100))
               + offset_from_map(star_map, loop_coeff_map(lc, seq, s, W1), r)
               for s in enumerate_structures(seq) if pair_table(s)[pair[0]] != pair[1])
    F = Folder(lc, seq, star, Weights(B["a_four"], B["cols"], r))
    _s, x, _e0 = F.without(*pair)
    assert x == best, (f"the forbidden-pair fold gives {x} dcal, the enumerated minimum over "
                       f"structures without {pair} is {best}")


def test_the_shape_fast_path_charges_nothing_it_should_have_charged(pinned):
    """Every loop whose shape the callback skips must have weight exactly zero, at any radius.

    The callback answers most invocations with one integer test: if the loop's (u1, u2) is not a
    shape one of the reached tables carries, it returns 0 without computing a weight. That is an
    8x speedup and it is only correct if those loops really do have zero weight -- otherwise the
    callback under-charges, the band shrinks, and pairs certify that are not robust.

    This checks it directly, over every interior loop of the generating set, which exercises
    every vrna_E_internal case, and at a radius four times the largest published one so that any
    nonzero coefficient would round up to at least one dcal. It needs no fold: the claim is
    about the weights, so it is tested against the weights.
    """
    from scripts.certfold import Weights, il_keys, load_tr0_basis
    from scripts.turner_nullspace import BCODE

    P, lc = pinned
    B = load_tr0_basis()
    W = Weights(B["a_four"], B["cols"], float(B["radii"]["r_cap_kcal"]))
    skipped, charged, nonzero_skipped = 0, 0, []
    for _tag, seq, db in generating_set(P["special"]):
        S = [0] + [BCODE[c] for c in seq] + [0]
        for d in lc.loops(seq, db):
            if d["kind"] != "interior":
                continue
            i, j, k, ll = d["i"], d["j"], d["k"], d["l"]
            w = W.dcal(il_keys(S, i, j, k, ll))
            if (k - i - 1, j - ll - 1) in W.shapes:
                charged += 1
            else:
                skipped += 1
                if w:
                    nonzero_skipped.append(((i, j, k, ll), (k - i - 1, j - ll - 1), w))
    assert not nonzero_skipped, (f"{len(nonzero_skipped)} loops the callback skips carry a "
                                 f"nonzero weight: {nonzero_skipped[:3]}")
    assert skipped > 10000 and charged > 1000, (f"the test is vacuous: {skipped} skipped, "
                                                f"{charged} charged")


def test_the_compiled_callback_returns_the_python_callbacks_weights(pinned):
    """The compiled callback charges every loop exactly what the Python callback charges.

    The authors' condition of 2026-10-13 for using the compiled callback in the final gate and
    the test-set runs: identical weights on all 116,297 interior loops of the generating set.
    This calls the cfunc's MACHINE CODE through ctypes, not a Python mirror of its logic, at
    both signs (s* = the row's own structure, so every loop is a loop of s*; and s* = the open
    chain, so none is) and at scales that cover both sets: the ball at r = 1, the Andronescu
    radius and r_cap, and the envelope at lambda = 0.5, 1 and 2.

    The compiled callback lives in .venv-certfold, where numba is installed (`make
    test-certfold`); in .venv there is nothing to compare, and the test says so.
    """
    pytest.importorskip("numba", reason="the compiled callback lives in .venv-certfold: run "
                                        "make test-certfold")
    import RNA

    from scripts.certfold import CompiledWeights, Weights, load_tr0_basis, star_interior_loops
    from scripts.turner_nullspace import BCODE

    P, lc = pinned
    B = load_tr0_basis()
    il = int(RNA.DECOMP_PAIR_IL)
    scales = [("ball", B["a_four"], r) for r in
              (1.0, float(B["radii"]["Andronescu2007"]["radius_on_B_four_kcal"]),
               float(B["radii"]["r_cap_kcal"]))]
    scales += [("envelope", B["u_env"], lam) for lam in (0.5, 1.0, 2.0)]
    n_loops, n_nonzero, bad = 0, 0, []
    rows = list(generating_set(P["special"]))
    for _tag, seq, db in rows:
        S = [0] + [BCODE[c] for c in seq] + [0]
        loops = [(d["i"], d["j"], d["k"], d["l"]) for d in lc.loops(seq, db)
                 if d["kind"] == "interior"]
        if not loops:
            continue
        n_loops += len(loops)
        for star_db in (db, "." * len(seq)):
            star = star_interior_loops(lc, seq, star_db)
            for _kind, a, x in scales:
                W = Weights(a, B["cols"], x)
                cw = CompiledWeights(seq, S, star, W)
                for q in loops:
                    w = W.loop_dcal(S, *q)
                    want = (w if q in star else -w) if w else 0
                    got = cw.lookup(*q, il)
                    n_nonzero += want != 0
                    if got != want:
                        bad.append((seq[:20], q, x, want, got))
    assert n_loops == 116297, f"the generating set has {n_loops} interior loops, not 116,297"
    assert n_nonzero > 10000, f"the comparison is vacuous: {n_nonzero} nonzero weights"
    assert not bad, f"{len(bad)} compiled weights differ from the Python callback's: {bad[:3]}"
