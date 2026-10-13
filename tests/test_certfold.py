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
