"""Certified folding: the accuracy test's metrics, on HAND-MADE toy records only.

The registered accuracy test (docs/certified_folding_design.md) ranks MFE pairs by a score and
compares PPV against baselines at matched coverage. Its metrics live in
scripts/certfold_metrics.py and are applied to TS0, the held-out set and bpRNA-new only after
the registration. Until then no prediction may be scored against any of those sets' reference
structures, nor against TR0's, so every record below is built by hand, small enough that each
expected value is worked out in the comment beside it. None of them stands in for a result.

What is held:
  * the slip rule, including i - 1 = 0 and j - 1 = i, and that a malformed pair is refused;
  * the ranking: score, then the nominal gap, and a tie on both taken at its expectation, so
    no number depends on the order records are listed in;
  * unscored records excluded from the curve and counted beside it;
  * PPV at coverage read without interpolation, and AURC of perfect and worst rankings;
  * matched coverage on the common pool, pooled and family-weighted;
  * family weighting differs from pooling where it should;
  * the bootstrap is deterministic, order-free, brackets its point estimate, and counts a
    family drawn twice twice;
  * the disagreement sets' boundaries and the per-stratum fractions.
"""

import itertools
import math
import pathlib
import re
import sys

import numpy as np
import pytest

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.certfold_metrics import (  # noqa: E402
    SEED,
    aurc,
    disagreement_sets,
    family_weighted_ppv,
    fraction_by_stratum,
    matched_coverage_comparison,
    paired_bootstrap,
    pooled_ppv,
    ppv_at_coverage,
    risk_coverage,
    slip_correct,
)
from scripts.turner_typeb import GuardFailure  # noqa: E402


def rec(unit, correct, score, tiebreak, p=0.5, family=None, pair=(1, 10), **extra):
    """One toy MFE-pair record in the registered format."""
    return {"unit": unit, "family": family, "pair": pair, "correct": correct,
            "score": score, "tiebreak": tiebreak, "p": p, **extra}


def shuffled(xs, tag):
    """A fixed reordering of a toy list, to show a result does not depend on list order."""
    return [xs[k] for k in np.random.default_rng(np.random.SeedSequence([SEED, 900, tag]))
            .permutation(len(xs))]


def test_seed_is_certfolds():
    """One seed for the whole study: the metrics module may not drift from certfold.py's."""
    text = (R / "scripts" / "certfold.py").read_text()
    m = re.search(r"^SEED = (\d+)", text, re.M)
    assert m is not None and int(m.group(1)) == SEED == 20261013


# ----------------------------------------------------------------------------------
# slip_correct
# ----------------------------------------------------------------------------------

def test_slip_allows_one_end_to_move_by_one():
    ref = {(5, 20)}
    for q in [(5, 20), (4, 20), (6, 20), (5, 19), (5, 21)]:
        assert slip_correct(q, ref) is True, q
    # both ends moved, or one end moved by two: not the same pair
    for q in [(4, 19), (6, 21), (4, 21), (6, 19), (3, 20), (7, 20), (5, 18), (5, 22)]:
        assert slip_correct(q, ref) is False, q


def test_slip_at_the_first_position():
    """At i = 1 the candidate (i - 1, j) = (0, j) is off the sequence: never a reference pair,
    while the other three slips still count."""
    assert slip_correct((1, 12), {(2, 12)}) is True       # (i + 1, j)
    assert slip_correct((1, 12), {(1, 11)}) is True       # (i, j - 1)
    assert slip_correct((1, 12), {(1, 13)}) is True       # (i, j + 1)
    assert slip_correct((1, 12), {(3, 12)}) is False
    assert slip_correct((2, 12), {(1, 12)}) is True       # i - 1 = 1 is a real position
    # a 0-based reference is refused, not matched against the slipped (0, j)
    with pytest.raises(GuardFailure):
        slip_correct((1, 12), {(0, 12)})


def test_slip_where_the_pair_is_adjacent():
    """At j = i + 1 the slips (i + 1, j) and (i, j - 1) close on themselves; (i, j + 1) and
    (i - 1, j) still count."""
    assert slip_correct((4, 5), {(4, 6)}) is True
    assert slip_correct((4, 5), {(3, 5)}) is True
    assert slip_correct((4, 5), {(5, 6)}) is False
    with pytest.raises(GuardFailure):
        slip_correct((4, 5), {(5, 5)})


def test_slip_refuses_malformed_pairs():
    for bad in [(0, 5), (7, 3), (4, 4), (True, 5), ("a", "b"), (1, 2, 3), 5, (1.0, 5)]:
        with pytest.raises(GuardFailure):
            slip_correct(bad, {(1, 5)})
    with pytest.raises(GuardFailure):
        slip_correct((1, 5), {(12, 1)})           # a reversed reference pair is not reoriented
    # numpy integers and a list of pairs are fine
    assert slip_correct((np.int64(5), np.int64(20)), [(np.int32(6), 20)]) is True
    assert slip_correct((5, 20), []) is False


# ----------------------------------------------------------------------------------
# risk_coverage, ppv_at_coverage, aurc
# ----------------------------------------------------------------------------------

def five_records():
    """A ranks first on score; B, C, D tie on score and are ordered by tiebreak C, D, B; E last.
    Correct in rank order: A T, C T, D F, B F, E T."""
    return [rec("s1", True, 3.0, 0.1),           # A
            rec("s1", False, 2.0, 0.5),          # B
            rec("s2", True, 2.0, 1.5),           # C
            rec("s2", False, 2.0, 1.0),          # D
            rec("s3", True, 1.0, 9.0)]           # E


FIVE_CURVE = [(1 / 5, 1.0), (2 / 5, 1.0), (3 / 5, 2 / 3), (4 / 5, 2 / 4), (5 / 5, 3 / 5)]


def test_curve_ranks_by_score_then_tiebreak():
    curve, n_none = risk_coverage(five_records())
    assert curve == FIVE_CURVE
    assert n_none == 0
    # input order A..E would give T F T F T without the tiebreak: 1, 1/2, 2/3, 1/2, 3/5
    assert curve[1][1] != 0.5
    for tag in range(5):
        assert risk_coverage(shuffled(five_records(), tag)) == (FIVE_CURVE, 0)


def test_curve_takes_a_full_tie_at_its_expectation():
    """X and Y tie on score AND tiebreak, one right and one wrong: the top 1 is right with
    probability 1/2 whichever is listed first. Z (correct) comes last."""
    xs = [rec("s1", True, 1.0, 1.0), rec("s1", False, 1.0, 1.0), rec("s2", True, 0.0, 0.0)]
    want = [(1 / 3, 0.5), (2 / 3, 0.5), (1.0, 2 / 3)]
    assert risk_coverage(xs)[0] == want
    assert risk_coverage(xs[::-1])[0] == want
    assert risk_coverage([xs[1], xs[0], xs[2]])[0] == want


def test_unscored_records_are_excluded_and_counted():
    xs = five_records() + [rec("s4", True, None, 0.3), rec("s4", False, None, 7.0)]
    curve, n_none = risk_coverage(xs)
    assert curve == FIVE_CURVE                  # coverage over the 5 scored, not the 7
    assert n_none == 2
    # an unscored record's tiebreak and label are not read
    xs.append({"unit": "s5", "family": None, "pair": (1, 9), "correct": None, "score": None,
               "tiebreak": None, "p": 0.1})
    assert risk_coverage(xs) == (FIVE_CURVE, 3)


def test_curve_guards():
    good = five_records()
    for field, bad in [("score", float("nan")), ("tiebreak", None), ("tiebreak", float("nan")),
                       ("correct", 1), ("correct", "yes"), ("score", "2.0")]:
        xs = [dict(r) for r in good]
        xs[2][field] = bad
        with pytest.raises(GuardFailure):
            risk_coverage(xs)
    xs = [dict(r) for r in good]
    del xs[0]["tiebreak"]
    with pytest.raises(GuardFailure):
        risk_coverage(xs)
    with pytest.raises(GuardFailure):
        risk_coverage([rec("s1", True, None, 1.0)])       # nothing scored: no curve
    with pytest.raises(GuardFailure):
        risk_coverage(good, score_key="rho")              # no such key


def test_ppv_at_coverage_takes_the_smallest_k_reaching_c():
    c = FIVE_CURVE
    assert ppv_at_coverage(c, 0.2) == 1.0          # k = 1
    assert ppv_at_coverage(c, 0.21) == 1.0         # k = 2, coverage 0.4
    assert ppv_at_coverage(c, 0.5) == 2 / 3        # k = 3, not the 5/6 interpolation would give
    assert ppv_at_coverage(c, 3 / 5) == 2 / 3      # c = m / n selects k = m exactly
    assert ppv_at_coverage(c, 0.61) == 0.5         # k = 4
    assert ppv_at_coverage(c, 1.0) == 0.6
    for bad in [0, 0.0, -0.1, 1.2, float("nan"), True]:
        with pytest.raises(GuardFailure):
            ppv_at_coverage(c, bad)
    with pytest.raises(GuardFailure):
        ppv_at_coverage([], 0.5)
    with pytest.raises(GuardFailure):
        ppv_at_coverage([(0.5, 1.0), (0.4, 1.0)], 0.5)     # not increasing
    with pytest.raises(GuardFailure):
        ppv_at_coverage([(0.25, 1.0), (0.5, 1.0)], 0.75)   # beyond a partial curve


def test_aurc_of_perfect_and_worst_rankings():
    """Three right pairs and one wrong. Perfect order: risks 0, 0, 0, 1/4 at coverage 1/4..1,
    area (1/4)(0 + 1/4)/2 = 1/32. Worst order (wrong first): risks 1, 1/2, 1/3, 1/4, area
    (1/4)[(1 + 1/2) + (1/2 + 1/3) + (1/3 + 1/4)]/2 = 35/96."""
    perfect = [rec("s1", True, 4.0, 0.0), rec("s1", True, 3.0, 0.0),
               rec("s2", True, 2.0, 0.0), rec("s2", False, 1.0, 0.0)]
    assert aurc(risk_coverage(perfect)[0]) == 1 / 32
    worst = [dict(r, score=5.0 - r["score"]) for r in perfect]
    assert aurc(risk_coverage(worst)[0]) == pytest.approx(35 / 96, abs=1e-15)
    # every one of the 24 orderings scores at least the perfect one
    for perm in itertools.permutations([4.0, 3.0, 2.0, 1.0]):
        xs = [dict(r, score=s) for r, s in zip(perfect, perm, strict=True)]
        assert aurc(risk_coverage(xs)[0]) >= 1 / 32
    # all correct: no risk anywhere
    assert aurc(risk_coverage([dict(r, correct=True) for r in perfect])[0]) == 0.0
    assert aurc([(1.0, 0.0)]) == 0.0               # one point: no width, from the first point
    with pytest.raises(GuardFailure):
        aurc([])


# ----------------------------------------------------------------------------------
# matched coverage
# ----------------------------------------------------------------------------------

def matched_records():
    """Score rho against baseline p. r5 has no rho and is out of BOTH rankings, leaving n = 5.
    The p >= 0.99 set in that pool is r1, r3: m = 2, coverage 2/5.
      rho order: r1 T, r2 T, r3 F, r4 F, r6 T  -> top 2 PPV 1,   top 3 PPV 2/3
      p order:   r1 T, r3 F, r2 T, r4 F, r6 T  -> top 2 PPV 1/2, top 3 PPV 2/3
    Families at the top 3: rho and p both select r1 (A, T), r2 (B, T), r3 (B, F), so the
    family-weighted PPV is (1 + 1/2)/2 = 3/4 against the pooled 2/3."""
    return [rec("s1", True, None, 1.0, p=0.999, family="A", rho=4.0),     # r1
            rec("s1", True, None, 2.0, p=0.80, family="B", rho=3.0),      # r2
            rec("s2", False, None, 0.5, p=0.995, family="B", rho=2.0),    # r3
            rec("s2", False, None, 0.2, p=0.60, family="A", rho=1.0),     # r4
            rec("s3", True, None, 0.4, p=0.999, family="A", rho=None),    # r5
            rec("s3", True, None, 0.1, p=0.30, family="B", rho=0.5)]      # r6


def test_matched_coverage_at_the_p099_set():
    out = matched_coverage_comparison(matched_records(), "rho", "p", lambda r: r["p"] >= 0.99)
    assert out["n"] == 5 and out["n_unscored"] == {"score": 1, "baseline": 0}
    assert out["n_in_coverage_set"] == 2 and out["coverage"] == 2 / 5 and out["k"] == 2
    assert (out["ppv_score"], out["ppv_baseline"], out["difference"]) == (1.0, 0.5, 0.5)
    # the same number given directly
    same = matched_coverage_comparison(matched_records(), "rho", "p", 0.4)
    assert (same["k"], same["difference"]) == (2, 0.5)
    # the PPVs are the curves' own, on the common pool
    pool = [r for r in matched_records() if r["rho"] is not None]
    assert out["ppv_score"] == ppv_at_coverage(risk_coverage(pool, "rho")[0], 0.4)
    assert out["ppv_baseline"] == ppv_at_coverage(risk_coverage(pool, "p")[0], 0.4)
    # at full coverage both rankings hold the same pairs
    full = matched_coverage_comparison(matched_records(), "rho", "p", 1.0)
    assert (full["ppv_score"], full["ppv_baseline"], full["difference"]) == (3 / 5, 3 / 5, 0.0)
    for tag in range(3):
        assert matched_coverage_comparison(shuffled(matched_records(), tag), "rho", "p",
                                           lambda r: r["p"] >= 0.99) == out


def test_matched_coverage_by_family():
    pooled = matched_coverage_comparison(matched_records(), "rho", "p", 0.6)
    fam = matched_coverage_comparison(matched_records(), "rho", "p", 0.6, by_family=True)
    assert pooled["k"] == fam["k"] == 3
    assert pooled["ppv_score"] == 2 / 3
    assert fam["ppv_score"] == fam["ppv_baseline"] == 0.75
    assert fam["n_families"] == {"score": 2, "baseline": 2}
    # one family: the family-weighted reading of a tied block is the pooled one
    tied = [rec("s1", True, 1.0, 1.0, family="A", b=0.0), rec("s1", False, 1.0, 1.0,
            family="A", b=1.0), rec("s2", True, 0.0, 0.0, family="A", b=2.0)]
    for c in (1 / 3, 2 / 3, 1.0):
        f = matched_coverage_comparison(tied, "score", "b", c, by_family=True)
        p = matched_coverage_comparison(tied, "score", "b", c)
        assert f["ppv_score"] == pytest.approx(p["ppv_score"], abs=1e-15)
    with pytest.raises(GuardFailure):                   # a record without a family
        matched_coverage_comparison([dict(r, family=None) for r in tied], "score", "b", 0.5,
                                    by_family=True)


def test_matched_coverage_guards():
    with pytest.raises(GuardFailure):
        matched_coverage_comparison(matched_records(), "rho", "rho", 0.5)
    with pytest.raises(GuardFailure):                   # an empty p >= 0.999999 set
        matched_coverage_comparison(matched_records(), "rho", "p", lambda r: r["p"] > 0.9999)
    with pytest.raises(GuardFailure):
        matched_coverage_comparison(matched_records(), "rho", "p", 0.0)
    with pytest.raises(GuardFailure):                   # the predicate must return a bool
        matched_coverage_comparison(matched_records(), "rho", "p", lambda r: r["p"])


# ----------------------------------------------------------------------------------
# pooled and family-weighted PPV
# ----------------------------------------------------------------------------------

def test_family_weighting_differs_from_pooling():
    """Family A: four right pairs. Family B: one wrong pair. Pooled 4/5; each family once,
    (1 + 0)/2."""
    xs = ([rec(f"a{k}", True, 1.0, 0.0, family="A") for k in range(4)]
          + [rec("b0", False, 1.0, 0.0, family="B")])
    assert pooled_ppv(xs) == 0.8
    assert family_weighted_ppv(xs, lambda r: True) == 0.5
    assert family_weighted_ppv(xs, [True] * 5) == 0.5
    # a family with no selected record is left out of the mean, not counted as 0
    assert family_weighted_ppv(xs, lambda r: r["family"] == "A") == 1.0
    assert family_weighted_ppv(xs, [True, False, False, False, True]) == 0.5
    assert pooled_ppv(xs, [True, False, False, False, True]) == 0.5
    assert family_weighted_ppv(shuffled(xs, 0), lambda r: True) == 0.5
    with pytest.raises(GuardFailure):
        family_weighted_ppv(xs, [True] * 4)              # one entry short
    with pytest.raises(GuardFailure):
        family_weighted_ppv(xs, lambda r: False)         # nothing selected
    with pytest.raises(GuardFailure):
        family_weighted_ppv(xs + [rec("c0", True, 1.0, 0.0, family=None)], lambda r: True)
    with pytest.raises(GuardFailure):
        pooled_ppv(xs, lambda r: False)


# ----------------------------------------------------------------------------------
# paired bootstrap
# ----------------------------------------------------------------------------------

def bootstrap_records():
    """Six sequences in three families, each sequence with one p >= 0.99 pair so that no
    resample has an empty matched set. 11 of the 18 pairs are right."""
    xs = []
    rows = [(True, True, False), (True, False, False), (True, True, True), (False, True, False),
            (True, False, True), (True, True, False)]
    for k, (a, b, c) in enumerate(rows):
        u = f"s{k}"
        xs += [rec(u, a, 3.0 - 0.1 * k, 1.0, p=0.995, family=f"F{k % 3}", rho=3.0 - 0.1 * k),
               rec(u, b, 2.0 - 0.1 * k, 0.5, p=0.70 + 0.01 * k, family=f"F{k % 3}",
                   rho=0.5 + 0.1 * k),
               rec(u, c, 1.0 - 0.1 * k, 0.2, p=0.40, family=f"F{k % 3}", rho=1.5 + 0.1 * k)]
    return xs


def test_bootstrap_is_deterministic_and_brackets_its_point():
    xs = bootstrap_records()
    got = paired_bootstrap(xs, pooled_ppv, n_boot=2000)
    assert got == paired_bootstrap(xs, pooled_ppv, n_boot=2000)
    point, lo, hi = got
    assert point == 11 / 18
    assert lo <= point <= hi and lo < hi
    # the draws index units in sorted order, so listing records differently changes nothing
    assert paired_bootstrap(shuffled(xs, 1), pooled_ppv, n_boot=2000) == got
    # a different tag is a different draw (read off the resamples themselves, since a PPV over
    # 18 pairs takes so few values that two draws can share their percentiles)
    def draws(tag):
        seen = []

        def spy(recs):
            seen.append(sorted(str(r["unit"]) for r in recs))
            return 0.0

        paired_bootstrap(xs, spy, n_boot=50, seed_tag=tag)
        return seen

    assert draws(40) == draws(40) and draws(40) != draws(41)
    assert draws(40)[0] == sorted(r["unit"] for r in xs)     # the point: the records as given


def test_bootstrap_of_the_matched_difference():
    """The registered use: the PPV difference at the p >= 0.99 coverage, recomputed with its
    own coverage on every resample."""
    def diff(recs):
        return matched_coverage_comparison(recs, "rho", "p", lambda r: r["p"] >= 0.99)[
            "difference"]

    point, lo, hi = paired_bootstrap(bootstrap_records(), diff, n_boot=1000)
    assert point == diff(bootstrap_records())
    assert lo <= point <= hi
    assert (point, lo, hi) == paired_bootstrap(bootstrap_records(), diff, n_boot=1000)


def test_bootstrap_counts_a_unit_drawn_twice_twice():
    """Resampling three families with replacement: without relabelling, a family drawn twice
    merges with itself and most resamples hold fewer than three. With it, every one holds
    three."""
    xs = bootstrap_records()

    def n_families(recs):
        return float(len({r["family"] for r in recs}))

    assert paired_bootstrap(xs, n_families, n_boot=500, by="family") == (3.0, 3.0, 3.0)

    def n_units(recs):
        return float(len({r["unit"] for r in recs}))

    assert paired_bootstrap(xs, n_units, n_boot=500) == (6.0, 6.0, 6.0)
    # the family-weighted PPV is then a mean over three entries on every resample
    point, lo, hi = paired_bootstrap(xs, lambda rs: family_weighted_ppv(rs, lambda r: True),
                                     n_boot=500, by="family")
    assert lo <= point <= hi


def test_bootstrap_guards():
    xs = bootstrap_records()
    with pytest.raises(GuardFailure):
        paired_bootstrap([r for r in xs if r["unit"] == "s0"], pooled_ppv, n_boot=10)
    with pytest.raises(GuardFailure):
        paired_bootstrap(xs, lambda rs: float("nan"), n_boot=10)
    with pytest.raises(GuardFailure):
        paired_bootstrap(xs, pooled_ppv, n_boot=0)
    with pytest.raises(GuardFailure):
        paired_bootstrap(xs, pooled_ppv, n_boot=10, seed_tag=-1)
    with pytest.raises(GuardFailure):                   # TS0-like records have no family
        paired_bootstrap([dict(r, family=None) for r in xs], pooled_ppv, n_boot=10,
                         by="family")
    with pytest.raises(GuardFailure):                   # a resample stat that raises stops it
        paired_bootstrap(xs, lambda rs: pooled_ppv(rs, lambda r: r["unit"] == "s0"),
                         n_boot=200)


# ----------------------------------------------------------------------------------
# disagreement sets, fractions by stratum
# ----------------------------------------------------------------------------------

def test_disagreement_sets_at_their_boundaries():
    a = rec("s1", True, 1.0, 0.0, p=0.995, certified_l1=False)    # D1
    b = rec("s1", False, 1.0, 0.0, p=0.99, certified_l1=False)    # D1: p >= 0.99 includes 0.99
    c = rec("s1", True, 1.0, 0.0, p=0.999, certified_l1=True)     # both agree: neither
    d = rec("s2", False, 1.0, 0.0, p=0.30, certified_l1=True)     # D2
    e = rec("s2", True, 1.0, 0.0, p=0.5, certified_l1=True)       # p < 0.5 excludes 0.5
    f = rec("s2", True, 1.0, 0.0, p=0.30, certified_l1=False)     # both doubt it: neither
    d1, d2 = disagreement_sets([a, b, c, d, e, f])
    assert d1 == [a, b] and d1[0] is a and d1[1] is b
    assert d2 == [d] and d2[0] is d
    assert pooled_ppv(d1) == 0.5 and pooled_ppv(d2) == 0.0
    with pytest.raises(GuardFailure):
        disagreement_sets([dict(a, certified_l1=None)])
    with pytest.raises(GuardFailure):
        disagreement_sets([dict(a, certified_l1=1)])
    with pytest.raises(GuardFailure):
        disagreement_sets([dict(a, p=99.5)])           # a percentage, not a probability
    with pytest.raises(GuardFailure):
        disagreement_sets([a], p_hi=0.4, p_lo=0.5)


def test_fraction_by_stratum():
    def seq(stratum, whole):
        return {"name": "toy", "stratum": stratum,
                "sets": {"envelope:1": {"whole_structure_certified": whole}}, "whole": whole}

    xs = [seq(2, True), seq(0, True), seq(0, False), seq(1, False), seq(0, True), seq(2, True)]
    want = {0: (2, 3, 2 / 3), 1: (0, 1, 0.0), 2: (2, 2, 1.0)}
    got = fraction_by_stratum(xs, "whole")
    assert got == want and list(got) == [0, 1, 2]
    nested = fraction_by_stratum(
        xs, lambda r: r["sets"]["envelope:1"]["whole_structure_certified"])
    assert nested == want
    assert fraction_by_stratum([seq(None, True), seq(None, False)], "whole") == {
        None: (1, 2, 0.5)}
    with pytest.raises(GuardFailure):
        fraction_by_stratum([seq(None, True), seq(0, False)], "whole")
    with pytest.raises(GuardFailure):
        fraction_by_stratum([dict(seq(0, True), whole=1)], "whole")
    with pytest.raises(GuardFailure):
        fraction_by_stratum([], "whole")
    assert math.isclose(sum(of for _n, of, _f in got.values()), len(xs))


# ----------------------------------------------------------------------------------
# added in review: the held-out families' bootstrap, the pinned seed, a curve passed whole
# ----------------------------------------------------------------------------------

def one_per_family():
    """Six families of one sequence each, as on the held-out one-per-family set: unit is the
    sequence, family its family. Three pairs per sequence; family Fk's correct count varies."""
    return [rec(f"seq{k}", (k + q) % 2 == 0, float(q), 0.0, family=f"F{k}")
            for k in range(6) for q in range(3)]


def test_bootstrap_copies_a_family_its_unit_carries_alone():
    """With unit = sequence (or unit = the family id) and by="unit", a family whose only
    sequence is drawn twice must count twice. Before the fix it merged with itself and most
    resamples held 2 to 5 of the 6 families."""
    def n_families(recs):
        return float(len({r["family"] for r in recs}))

    xs = one_per_family()
    assert paired_bootstrap(xs, n_families, n_boot=300) == (6.0, 6.0, 6.0)
    ys = [dict(r, unit=r["family"]) for r in xs]            # unit is the family id
    assert paired_bootstrap(ys, n_families, n_boot=300) == (6.0, 6.0, 6.0)
    # and the family-weighted PPV's interval is then the by="family" one, draw for draw
    def fw(rs):
        return family_weighted_ppv(rs, lambda r: True)

    assert paired_bootstrap(ys, fw, n_boot=500) == paired_bootstrap(ys, fw, n_boot=500,
                                                                      by="family")


def test_bootstrap_keeps_a_family_spread_over_units():
    """Two sequences per family: a sequence drawn twice is two sequences of the SAME family,
    so no new family label ever appears in a resample."""
    xs = bootstrap_records()
    seen = set()

    def spy(recs):
        seen.update(r["family"] for r in recs)
        return 0.0

    paired_bootstrap(xs, spy, n_boot=300)
    assert seen == {"F0", "F1", "F2"}


def test_bootstrap_interval_against_an_independent_reference():
    """The seed is SeedSequence([20261013, 40]) over the sorted unit ids, one (n_boot, n_units)
    draw, and the interval numpy's linear 2.5th and 97.5th percentiles. Recomputed here from
    the literal seed, so a change to SEED, the tag, or the draw would be caught."""
    xs = bootstrap_records()
    units = sorted({r["unit"] for r in xs})
    hits = {u: sum(r["correct"] for r in xs if r["unit"] == u) for u in units}
    size = {u: sum(r["unit"] == u for r in xs) for u in units}
    rng = np.random.default_rng(np.random.SeedSequence([20261013, 40]))
    draws = rng.integers(0, len(units), size=(700, len(units)))
    vals = [sum(hits[units[x]] for x in row) / sum(size[units[x]] for x in row) for row in draws]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    assert paired_bootstrap(xs, pooled_ppv, n_boot=700) == (11 / 18, float(lo), float(hi))
    # a PPV over 18 pairs takes so few values that another seed or level can land on the same
    # percentiles; a sum of sqrt(prime) unit weights separates every multiset of draws
    w = {u: math.sqrt(q) for u, q in zip(units, (2, 3, 5, 7, 11, 13), strict=True)}

    def weight(recs):                       # a later copy's unit is (unit, copy)
        return math.fsum(w[r["unit"] if isinstance(r["unit"], str) else r["unit"][0]]
                         for r in recs)

    vals = [math.fsum(size[units[x]] * w[units[x]] for x in row) for row in draws]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    assert len(set(vals)) > 100
    assert paired_bootstrap(xs, weight, n_boot=700) == (weight(xs), float(lo), float(hi))


def test_curve_functions_refuse_risk_coverages_whole_return():
    whole = risk_coverage(five_records())                   # (curve, n_unscored)
    with pytest.raises(GuardFailure):
        ppv_at_coverage(whole, 0.5)
    with pytest.raises(GuardFailure):
        aurc(whole)
    with pytest.raises(GuardFailure):
        aurc(risk_coverage(five_records()[:2]))             # a two-point curve, passed whole
