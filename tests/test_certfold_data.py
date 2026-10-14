"""Certified folding: the test-set loaders, which decide which rows the run phase certifies.

A loader that silently drops, duplicates or reorders rows changes every fraction downstream
without changing any code that computes one, so these tests hold scripts.certfold_data to:

  * the counts the sets are defined by: TS0 1,288 rows, 1,641 held-out families, 5,401
    bpRNA-new list entries;
  * the registered loader: the tarball reader reproduces tr0_ts0_rows on TR0 and TS0 row for
    row and count for count, which is what licenses applying it to bpRNA-new;
  * the filters: every TS0 and bpRNA-new row is ACGU only, canonical, nested, and has no pair
    enclosing fewer than 3 unpaired bases; no bpRNA-new sequence is a TR0 or TS0 sequence;
  * the draws: deterministic for a seed tag, different across tags, and identified by the
    digests their records carry;
  * the strata: TR0_STRATA is the pilot's, and length_stratum covers 1..1000 monotonically.

No accuracy of any prediction is computed against any reference here: the authors forbid it on
TS0, the held-out set and bpRNA-new before the registration. The parser's rules are checked on
small hand-made BPSEQ records instead.
"""

import hashlib
import json
import pathlib
import sys

import pytest

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold_data as cd  # noqa: E402
from scripts.turner_nullspace import PTYPE  # noqa: E402
from scripts.turner_nullspace_tr0 import distinct_rows, tr0_ts0_rows  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

STAT_KEYS = ("raw", "dropped_non_ACGU", "pairs_removed_short_hairpin",
             "structures_touched_by_short_hairpin_rule", "by_list")


@pytest.fixture(scope="module")
def basis_loader():
    """The registered loader's own output, which every comparison here is held to."""
    return tr0_ts0_rows()


@pytest.fixture(scope="module")
def ts0():
    return cd.ts0_rows()


@pytest.fixture(scope="module")
def bprna():
    return cd.bprna_new_rows()


@pytest.fixture(scope="module")
def heldout():
    return cd.heldout_one_per_family()


def pairs_of(db: str) -> list[tuple[int, int]]:
    """1-based pairs of a dot-bracket string; fails on an unbalanced one."""
    stack, out = [], []
    for k, c in enumerate(db, start=1):
        if c == "(":
            stack.append(k)
        elif c == ")":
            assert stack, f"unbalanced ')' at {k}"
            out.append((stack.pop(), k))
        else:
            assert c == ".", f"unexpected character {c!r}"
    assert not stack, "unbalanced '('"
    return sorted(out)


def assert_ts0_filters(rows, tag):
    """What the TS0 loader guarantees of every row it keeps, checked from the row alone."""
    for t, name, seq, db in rows:
        assert t == tag, (t, name)
        assert seq and set(seq) <= set("ACGU"), name
        assert len(db) == len(seq), name
        ps = pairs_of(db)
        assert ps, f"{name}: not pair-bearing"
        for i, j in ps:
            assert seq[i - 1] + seq[j - 1] in PTYPE, f"{name}: non-canonical ({i}, {j})"
            # after the fixpoint every pair encloses a hairpin of >= 3 unpaired bases, so no
            # pair can span fewer than 3 positions between its ends
            assert j - i - 1 >= cd.MIN_HAIRPIN, f"{name}: ({i}, {j}) encloses < 3"


def bpseq_text(seq: str, pairs: list[tuple[int, int]], header: str = "") -> str:
    partner = [0] * (len(seq) + 1)
    for i, j in pairs:
        partner[i], partner[j] = j, i
    body = "".join(f"{k} {seq[k - 1]} {partner[k]}\n" for k in range(1, len(seq) + 1))
    return header + body


# ----------------------------------------------------------------------------------
# the parser, on hand-made records
# ----------------------------------------------------------------------------------
def test_bpseq_row_rules_on_toy_records():
    """Each loader rule on a record built to trigger exactly it."""
    assert cd.bpseq_row(bpseq_text("GGGAAACCC", [(1, 9), (2, 8), (3, 7)]), "keep") == \
        ("GGGAAACCC", "(((...)))", 0)
    # an inner hairpin of 2 is stripped; the outer pair then encloses 4 and stays
    assert cd.bpseq_row(bpseq_text("GGAACC", [(1, 6), (2, 5)]), "inner") == \
        ("GGAACC", "(....)", 1)
    # the fixpoint: stripping the inner pair leaves the outer one closing 2, so it goes too
    assert cd.bpseq_row(bpseq_text("GGCC", [(1, 4), (2, 3)]), "cascade") == ("GGCC", "....", 2)
    # uppercased, T -> U, before the canonical check
    assert cd.bpseq_row(bpseq_text("gggtttccc", [(1, 9)]), "case") == \
        ("GGGUUUCCC", "(.......)", 0)
    assert cd.bpseq_row(bpseq_text("GGGNAACCC", [(1, 9)]), "non-ACGU") is None
    # comment lines and CRLF line ends parse as Path.read_text would give them
    crlf = bpseq_text("GGGAAACCC", [(1, 9)], header="# a comment\n").replace("\n", "\r\n")
    assert cd.bpseq_row(crlf, "crlf") == ("GGGAAACCC", "(.......)", 0)


@pytest.mark.parametrize("text, what", [
    (bpseq_text("AGGAAACCA", [(1, 9)]), "non-canonical"),
    (bpseq_text("GAAGAAACAAACAA", [(1, 8), (4, 12)]), "crossing"),
    ("1 G 9\n2 A 0\n3 A 0\n4 A 0\n5 A 0\n6 A 0\n7 A 0\n8 A 0\n9 C 0\n", "asymmetric"),
    ("1 G 0\n3 A 0\n", "index"),
    ("1 G 0 extra\n", "fields"),
    ("1 G 7\n", "range"),
    ("1 G 1\n2 A 0\n", "self-pair"),
])
def test_bpseq_row_refuses_malformed_records(text, what):
    with pytest.raises(GuardFailure):
        cd.bpseq_row(text, what)


# ----------------------------------------------------------------------------------
# TS0 and the registered loader
# ----------------------------------------------------------------------------------
def test_tarball_reader_reproduces_the_basis_loader(basis_loader):
    """The replicated rules ARE tr0_ts0_rows's: same rows in the same order, same counts."""
    rows, stats = basis_loader
    got, gstats = cd.tarball_rows(("TR0", "TS0"))
    assert got == rows
    for k in STAT_KEYS:
        assert gstats[k] == stats[k], k
    for k in STAT_KEYS[:-1]:
        assert sum(v[k] for v in gstats["per_list"].values()) == gstats[k], k
    assert gstats["tarball_sha256"] == hashlib.sha256((R / cd.TARBALL).read_bytes()).hexdigest()


def test_ts0_rows_count_and_view(ts0, basis_loader):
    """1,288 rows, and exactly the certfold basis loader's filter applied to the TS0 tag."""
    assert len(ts0) == cd.TS0_ROWS == 1288
    rows, _ = basis_loader
    assert ts0 == [r for r in distinct_rows(rows) if r[0] == "TS0" and "(" in r[3]]
    assert len({(r[2], r[3]) for r in ts0}) == len(ts0)


def test_ts0_rows_pass_the_filters(ts0):
    assert_ts0_filters(ts0, "TS0")


def test_unknown_list_tag_is_refused():
    with pytest.raises(GuardFailure):
        cd.tarball_rows(("VL1",))


# ----------------------------------------------------------------------------------
# bpRNA-new
# ----------------------------------------------------------------------------------
def test_bprna_new_counts_and_record(bprna):
    rows, rec = bprna
    assert rec["n_listed"] == cd.BPRNA_NEW_LISTED == 5401
    assert rec["n_rows"] == len(rows) == rec["steps"][-1]["sequences"]
    assert [s["sequences"] for s in rec["steps"]] == sorted(
        (s["sequences"] for s in rec["steps"]), reverse=True)
    assert rec["steps"][1]["sequences"] == rec["n_listed"] - rec["dropped_non_ACGU"]
    assert rec["dropped_identical_to_TR0_or_TS0"] == \
        rec["steps"][-2]["sequences"] - rec["steps"][-1]["sequences"]
    assert len({r[1] for r in rows}) == len(rows)
    assert rec["tarball_sha256"] == hashlib.sha256((R / cd.TARBALL).read_bytes()).hexdigest()
    assert rec["rows_sha256"] == cd.canonical_sha256([[r[1], r[2], r[3]] for r in rows])
    assert "no Rfam family labels" in rec["weighting"]
    assert "per sequence" in rec["weighting"].lower()
    assert "may" in rec["family_overlap_caveat"] and "bpRNA-1m" in rec["family_overlap_caveat"]


def test_bprna_new_rows_pass_the_ts0_filters(bprna):
    rows, _ = bprna
    assert_ts0_filters(rows, "bpRNA-new")
    assert len({(r[2], r[3]) for r in rows}) == len(rows)


def test_bprna_new_has_no_tr0_or_ts0_sequence(bprna, basis_loader):
    rows, _ = bprna
    seen = {r[2] for r in basis_loader[0]}
    assert not [r[1] for r in rows if r[2] in seen]


def test_bprna_new_filters_on_toy_records(monkeypatch):
    """The steps after the parse, on hand-made rows: on the real lists no bpRNA-new sequence is a
    TR0 or TS0 sequence, so only a toy record shows that the identity filter acts at all."""
    toy = [("TR0", "t1", "GGGAAACCC", "(((...)))"),
           ("TS0", "s1", "GGGGAAAACCCC", "((((....))))"),
           ("bpRNA-new", "n1", "GGGAAACCC", "((.....))"),        # TR0's sequence, other db
           ("bpRNA-new", "n2", "GGGGAAAACCCC", "(((......)))"),  # TS0's sequence, other db
           ("bpRNA-new", "n3", "CCCAAAGGG", "(((...)))"),        # kept
           ("bpRNA-new", "n4", "CCCAAAGGG", "(((...)))"),        # duplicate (seq, db) of n3
           ("bpRNA-new", "n5", "AAAAAAAAA", ".........")]        # not pair-bearing
    own = {"raw": 5, "dropped_non_ACGU": 0, "pairs_removed_short_hairpin": 0,
           "structures_touched_by_short_hairpin_rule": 0}
    stats = {"by_list": {"TR0": 1, "TS0": 1, "bpRNA-new": 5}, "per_list": {"bpRNA-new": own},
             "tarball_sha256": "toy"}
    monkeypatch.setattr(cd, "tarball_rows", lambda tags: (list(toy), stats))
    monkeypatch.setattr(cd, "BPRNA_NEW_LISTED", 5)
    rows, rec = cd.bprna_new_rows()
    assert rows == [toy[4]]
    assert [s["sequences"] for s in rec["steps"]] == [5, 5, 4, 3, 1]
    assert (rec["identical_to_TR0"], rec["identical_to_TS0"], rec["identical_to_both"]) == \
        (1, 1, 0)
    assert rec["dropped_identical_to_TR0_or_TS0"] == 2
    monkeypatch.setattr(cd, "BPRNA_NEW_LISTED", 6)
    with pytest.raises(GuardFailure):
        cd.bprna_new_rows()


def test_draws_use_the_specified_tags_and_refuse_bad_ones():
    """Defaults 20 (held-out) and 21 (bpRNA-new) as the task fixed them; a bool, negative or
    numpy tag is refused before any data are read (True would replay tag 1's stream)."""
    import inspect

    import numpy as np

    assert (cd.SEED_TAG_HELDOUT, cd.SEED_TAG_BPRNA_NEW) == (20, 21)
    assert inspect.signature(cd.heldout_one_per_family).parameters["seed_tag"].default == 20
    sig = inspect.signature(cd.bprna_new_subsample).parameters
    assert (sig["n"].default, sig["seed_tag"].default) == (2500, 21)
    for bad in (True, -1, 20.0, "20", np.int64(20)):
        with pytest.raises(GuardFailure):
            cd.heldout_one_per_family(seed_tag=bad)
        with pytest.raises(GuardFailure):
            cd.bprna_new_subsample(seed_tag=bad)


def test_bprna_new_subsample_is_seeded(bprna, monkeypatch):
    """Deterministic for a tag, different across tags, sorted by name, identified by digest."""
    pool, _ = bprna
    a, ra = cd.bprna_new_subsample()
    c, rc = cd.bprna_new_subsample(seed_tag=cd.SEED_TAG_BPRNA_NEW + 1)
    # the repeat draws from the fixture's pool, which test_bprna_new_counts_and_record ties to
    # the tarball; a fresh read would only re-time the same parse
    monkeypatch.setattr(cd, "bprna_new_rows", lambda: (list(pool), dict(bprna[1])))
    b, rb = cd.bprna_new_subsample()
    assert a == b and ra["names_sha256"] == rb["names_sha256"]
    assert a != c and ra["names_sha256"] != rc["names_sha256"]
    for rows, rec, tag in ((a, ra, cd.SEED_TAG_BPRNA_NEW), (c, rc, cd.SEED_TAG_BPRNA_NEW + 1)):
        names = [r[1] for r in rows]
        assert len(rows) == rec["n"] == 2500 and rec["n_pool"] == len(pool)
        assert names == sorted(names) and len(set(names)) == len(names)
        assert set(rows) <= set(pool)
        assert rec["seed"] == [cd.SEED, tag]
        assert rec["names_sha256"] == cd.canonical_sha256(names)
    with pytest.raises(GuardFailure):
        cd.bprna_new_subsample(n=0)
    with pytest.raises(GuardFailure):
        cd.bprna_new_subsample(n=len(pool) + 1)


# ----------------------------------------------------------------------------------
# the held-out families
# ----------------------------------------------------------------------------------
def test_heldout_one_per_family(heldout):
    from scripts.typeb_heldout import heldout_rows

    rows, rec = heldout
    pool = heldout_rows()
    fams = [r[0] for r in rows]
    assert len(rows) == rec["n_families"] == cd.HELDOUT_FAMILIES == 1641
    assert fams == sorted(set(fams)), "one row per family, sorted by family"
    assert fams == sorted({r[0] for r in pool})
    assert rec["n_sequences_in_pool"] == len(pool)
    assert set(rows) <= set(pool)
    assert rec["seed"] == [cd.SEED, cd.SEED_TAG_HELDOUT]
    assert rec["family_name_sha256"] == cd.canonical_sha256([[r[0], r[1]] for r in rows])


def test_heldout_draw_is_seeded(heldout):
    rows, rec = heldout
    again, rec2 = cd.heldout_one_per_family()
    other, rec3 = cd.heldout_one_per_family(seed_tag=cd.SEED_TAG_HELDOUT + 1)
    assert again == rows and rec2 == rec
    assert other != rows and rec3["family_name_sha256"] != rec["family_name_sha256"]
    assert [r[0] for r in other] == [r[0] for r in rows]


# ----------------------------------------------------------------------------------
# strata, seeds, digests
# ----------------------------------------------------------------------------------
def test_tr0_strata_are_the_pilots():
    """Recomputed through certfold.pilot_sequences, and equal to what the pilot recorded."""
    from scripts.certfold import PILOT_STRATA, read_out

    strata = cd.TR0_STRATA
    assert len(strata) == PILOT_STRATA
    recorded = read_out()["pilot"]["strata"]
    assert strata == [(s["min_len"], s["max_len"]) for s in recorded]
    strata.append((0, 0))
    assert len(cd.TR0_STRATA) == PILOT_STRATA, "TR0_STRATA must not be editable by a caller"


def test_length_stratum_on_the_pilots_sequences():
    """Every pilot draw maps to its own stratum, except at a length two strata share."""
    from scripts.certfold import pilot_sequences, tr0_rows

    b = cd.TR0_STRATA
    picks, _ = pilot_sequences(tr0_rows()[0])
    for name, seq, s in picks:
        got = cd.length_stratum(len(seq), b)
        shared = s > 0 and len(seq) == b[s][0] == b[s - 1][1]
        assert got == (s - 1 if shared else s), (name, len(seq), s, got)


def test_length_stratum_covers_1_to_1000_monotonically():
    b = cd.TR0_STRATA
    got = [cd.length_stratum(n, b) for n in range(1, 1001)]
    assert all(x <= y for x, y in zip(got, got[1:], strict=False))
    assert got[0] == 0 and got[-1] == len(b) - 1
    assert set(got) == set(range(len(b)))
    for s, (_lo, hi) in enumerate(b):
        assert cd.length_stratum(hi, b) == s
    assert cd.length_stratum(b[0][0] - 1, b) == 0
    assert cd.length_stratum(b[-1][1] + 1, b) == len(b) - 1


def test_length_stratum_on_toy_bounds():
    b = [(5, 10), (10, 20), (25, 30)]
    assert [cd.length_stratum(n, b) for n in (1, 5, 10, 11, 20, 21, 25, 30, 99)] == \
        [0, 0, 0, 1, 1, 2, 2, 2, 2]


@pytest.mark.parametrize("n, bounds", [
    (5, []),
    (5, [(10, 5)]),
    (5, [(1, 10), (5, 20)]),           # overlaps by more than one shared length
    (5, [(1, 10), (10, 10)]),          # the second stratum could never be returned
    (5, [(10, 20), (1, 9)]),           # unsorted
    (0, [(1, 10)]),
    (True, [(1, 10)]),
    (1.5, [(1, 10)]),
])
def test_length_stratum_refuses(n, bounds):
    with pytest.raises(GuardFailure):
        cd.length_stratum(n, bounds)


def test_seed_is_certfolds():
    from scripts.certfold import SEED

    assert cd.SEED == SEED == 20261013


def test_canonical_sha256():
    want = hashlib.sha256(json.dumps([["RF00001", "x/1-10"]], separators=(",", ":"))
                          .encode("utf-8")).hexdigest()
    assert cd.canonical_sha256([("RF00001", "x/1-10")]) == want
    assert cd.canonical_sha256([["RF00001", "x/1-10"]]) == want
