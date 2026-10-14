#!/usr/bin/env python3
"""CERTIFIED FOLDING, THE ACCURACY TEST'S METRICS. Pure functions over per-pair records.

WHAT THIS IS. The metrics of the registered accuracy test in docs/certified_folding_design.md
("Reviewers will score certified radius by PPV at matched coverage on held-out families"): rank
the Turner 2004 MFE pairs by a score, compare PPV against each baseline at matched coverage --
at the coverage of the p >= 0.99 set and along the full risk-coverage curve -- with a
one-nucleotide slip allowed, families weighted equally on the held-out sets, and paired
bootstrap intervals. Raw PPV of certified pairs means nothing on its own (MFE pairs average
65.8%, pairs with p >= 0.99 reach 91.0%; Mathews 2004), which is why every comparison here is
made at the same number of predicted pairs.

WHEN IT MAY RUN. These functions are written and tested BEFORE the registration, on hand-made
toy records only. They are applied to TS0, the held-out one-per-family set and bpRNA-new only
after it. Nothing here reads a file, folds a sequence or sees a reference structure: the caller
scores each pair with slip_correct and passes the records in. So this module pins no energy
model and needs none; the energies and probabilities it ranks were computed under
thermo.vienna_ref.reference_md() by scripts/certfold.py.

THE RECORD, one dict per MFE pair:
  unit      str or int     the sequence or family id the bootstrap resamples
  family    str or None    the family; None where the set has none (TS0)
  pair      (i, j)         1-based, i < j, as ViennaRNA's pair table gives it
  correct   bool           slip_correct(pair, reference pairs), computed by the caller
  score     float or None  the ranking score; None = not computed (rho is budgeted)
  tiebreak  float          the nominal gap, kcal/mol; higher = more confident
  p         float          the Turner 2004 base-pair probability
Baselines' scores and labels such as certified_l1 are further keys, named by the caller.

DECISIONS this module makes, each stated where it is made:
  * ORDER. Score descending, then tiebreak descending: the registered rule, under which ties in
    rho, capped values included, are ordered by the nominal gap. Records tied on BOTH form a
    block whose internal order nothing defines; each member is taken with equal fractional
    weight, which for a pooled PPV is exactly the expectation over every ordering of the block
    (a family-weighted PPV of those weights is order-free but not that expectation; see
    _top_k_weights). Every number is then independent of the order records are listed in.
  * MATCHED COVERAGE is evaluated on the records BOTH rankings score, so the two PPVs share a
    denominator and the same k, and the comparison is paired.
  * NO INTERPOLATION between ranks: PPV at coverage c is read at the smallest k reaching c.
  * THE BOOTSTRAP relabels a unit drawn more than once, and the family of a unit that carries
    its family alone, so a family drawn twice counts twice in a family-weighted mean instead of
    merging with itself.

Guards raise GuardFailure and are never repaired: a NaN score, a missing tiebreak, a label that
is not a bool, a pair that is not 1-based with i < j, a coverage outside (0, 1], a probability
outside [0, 1].
"""

from __future__ import annotations

import bisect
import math
import pathlib
import sys
from collections.abc import Callable, Sequence

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.turner_typeb import GuardFailure  # noqa: E402

SEED = 20261013                  # scripts/certfold.py's SEED; tests/test_certfold_metrics.py holds
                                 # the two equal
BOOTSTRAP_TAG = 40               # SeedSequence([SEED, 40]): the accuracy test's paired bootstrap;
                                 # certfold.py uses tags 1-3 and 10-12
P_HI = 0.99                      # the p >= 0.99 set of Mathews 2004, whose coverage is the matched
                                 # point
P_LO = 0.5                       # D2: certified despite a probability below one half
P_ROUNDING = 1e-6                # ViennaRNA's bpp can exceed 1 by rounding; past 1 + this it is a
                                 # unit error (a percentage, a log), not rounding
BOOT_LEVEL = 95.0                # the percentile interval's level, percent

Record = dict
Curve = list[tuple[float, float]]
Mask = Sequence[bool] | Callable[[dict], bool]

_REAL = (int, float, np.integer, np.floating)
_BOOL = (bool, np.bool_)


# ----------------------------------------------------------------------------------
# guards: every value is checked where it enters, and a bad one stops the analysis
# ----------------------------------------------------------------------------------

def _column(records: Sequence[Record], key: str) -> list:
    """record[key] for every record; a missing key is a malformed record, not a None."""
    try:
        return [r[key] for r in records]
    except KeyError:
        raise GuardFailure(f"a record has no {key!r}") from None
    except TypeError:
        raise GuardFailure("records must be dicts") from None


def _floats(values: list, what: str) -> np.ndarray:
    """Real numbers as float64, refusing NaN and anything that is not a number.

    The type check comes first because numpy would turn the string "1.0" into 1.0 and None
    into NaN without complaint, and a ranking built on either is wrong without looking wrong.
    """
    for t in set(map(type, values)):
        if not issubclass(t, _REAL + _BOOL):     # a bool label may serve as a score
            raise GuardFailure(f"{what}: a value of type {t.__name__} is not a number")
    a = np.asarray(values, dtype=np.float64)
    if np.isnan(a).any():
        raise GuardFailure(f"{what}: NaN; None is the only way to say 'no score'")
    return a


def _bools(values: list, what: str) -> np.ndarray:
    """Booleans as a bool array; 0 and 1 are refused, since a count passed for a label means
    the caller handed over the wrong field."""
    for t in set(map(type, values)):
        if not issubclass(t, _BOOL):
            raise GuardFailure(f"{what}: a value of type {t.__name__} is not a bool")
    return np.asarray(values, dtype=bool).reshape(len(values))


def _probabilities(values: list) -> np.ndarray:
    p = _floats(values, "p")
    if (p < 0).any() or (p > 1 + P_ROUNDING).any():
        raise GuardFailure(f"p outside [0, 1]: {p.min()} to {p.max()}")
    return p


def _mask(records: Sequence[Record], mask: Mask) -> np.ndarray:
    """A mask given as a predicate on a record, or as one bool per record, as a bool array.

    A predicate is what a bootstrap statistic needs, since a resample has no fixed alignment
    with the original list; a sequence of bools must have exactly one entry per record.
    """
    if callable(mask):
        return _bools([mask(r) for r in records], "mask")
    vals = list(mask)
    if len(vals) != len(records):
        raise GuardFailure(f"mask has {len(vals)} entries for {len(records)} records")
    return _bools(vals, "mask")


def _coverage(c) -> float:
    if isinstance(c, bool) or not isinstance(c, _REAL) or not 0 < c <= 1:
        raise GuardFailure(f"coverage {c!r} is not in (0, 1]")
    return float(c)


def _finite(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, _REAL) or not math.isfinite(x):
        raise GuardFailure(f"{what} is {x!r}, not a finite number")
    return float(x)


def _pair(q, what: str) -> tuple[int, int]:
    try:
        i, j = q
    except (TypeError, ValueError):
        raise GuardFailure(f"{what} {q!r} is not a pair (i, j)") from None
    for x in (i, j):
        if isinstance(x, _BOOL) or not isinstance(x, (int, np.integer)):
            raise GuardFailure(f"{what} {q!r}: positions must be integers")
    if not 1 <= i < j:
        raise GuardFailure(f"{what} {q!r} is not 1-based with i < j")
    return int(i), int(j)


# ----------------------------------------------------------------------------------
# scoring one pair
# ----------------------------------------------------------------------------------

def slip_correct(pair, ref_pairs) -> bool:
    """Whether the predicted pair (i, j) is in the reference up to a one-nucleotide slip.

    Correct iff (i, j), (i - 1, j), (i + 1, j), (i, j - 1) or (i, j + 1) is a reference pair:
    one end may slip by one position, never both. This is the rule of Mathews 2004 (RNA
    10:1178-1190, doi:10.1261/rna.7650904, Materials and Methods, Statistics), the paper whose
    65.8% and 91.0% PPVs the certificate is set beside, and the default Mathews 2019
    recommends for benchmarks (Methods 162-163:60-67, doi:10.1016/j.ymeth.2019.04.003,
    section 4; RNAstructure's scorer applies it by default). Scoring the certificate by a
    different rule than the numbers it is compared with would bias the comparison one way or
    the other, so the rule is the published one, not a choice.

    Positions are 1-based with i < j, for the prediction and for every reference pair, as
    ViennaRNA's pair table and BPSEQ give them; anything else is refused rather than
    reoriented. A slipped candidate that leaves the sequence or closes on itself, (0, j) at
    i = 1 or (i, i) at j = i + 1, therefore can never be a reference pair. The reference is
    checked and copied into a set on every call, O(|ref|), so any collection of pairs will do.
    """
    i, j = _pair(pair, "predicted pair")
    ref = {_pair(q, "reference pair") for q in ref_pairs}
    return any(c in ref for c in ((i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)))


# ----------------------------------------------------------------------------------
# ranking, risk-coverage, matched coverage
# ----------------------------------------------------------------------------------

def _scored(records: Sequence[Record], score_key: str, tiebreak_key: str):
    """The records that carry a score, with their (score, tiebreak, correct) arrays in input
    order, and how many carry none."""
    raw = _column(records, score_key)
    keep = [records[k] for k, s in enumerate(raw) if s is not None]
    if not keep:
        raise GuardFailure(f"no record has a {score_key!r}: the curve is undefined")
    sc = _floats([s for s in raw if s is not None], score_key)
    tb = _floats(_column(keep, tiebreak_key), tiebreak_key)
    ok = _bools(_column(keep, "correct"), "correct")
    return keep, sc, tb, ok, len(raw) - len(keep)


def _order(sc: np.ndarray, tb: np.ndarray) -> np.ndarray:
    """Rank order: score descending, then tiebreak descending (lexsort's LAST key is primary)."""
    return np.lexsort((-tb, -sc))


def _curve(sc: np.ndarray, tb: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage k/n and PPV for k = 1..n, a block tied on both keys taken at its expectation.

    Within a block of m records holding h correct ones, entered after a records holding A
    correct ones, the top k (a < k <= a + m) hold A + (k - a) h / m correct in expectation
    over the block's orderings. At every block end that is the exact integer count. The
    product (k - a) h is formed in integers so that the block ends come out exact in floats.
    """
    o = _order(sc, tb)
    s, t, c = sc[o], tb[o], ok[o].astype(np.int64)
    n = len(s)
    new = np.ones(n, dtype=bool)
    new[1:] = (s[1:] != s[:-1]) | (t[1:] != t[:-1])
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, n))
    hits = np.add.reduceat(c, starts)
    before = np.concatenate(([0], np.cumsum(hits)[:-1]))
    blk = np.cumsum(new) - 1
    k = np.arange(1, n + 1)
    got = before[blk] + (k - starts[blk]) * hits[blk] / sizes[blk]
    return k / n, got / k


def risk_coverage(records: Sequence[Record], score_key: str = "score",
                  tiebreak_key: str = "tiebreak") -> tuple[Curve, int]:
    """The risk-coverage curve of one ranking, as (coverage, PPV) at every k; and the number
    of records left out because their score is None.

    Records are ranked by score descending, ties by tiebreak descending (the registered rule:
    ties in rho, capped values included, are ordered by the nominal gap). Records tied on
    both form a block taken at its expectation over orderings (see _curve). Records with score
    None -- rho not computed within its budget -- are excluded, not ranked last, and their
    count is returned beside the curve so it is reported rather than lost: coverage is
    k / n_scored, PPV is correct among the top k over k.
    """
    _keep, sc, tb, ok, n_none = _scored(records, score_key, tiebreak_key)
    cov, ppv = _curve(sc, tb, ok)
    return [(float(c), float(p)) for c, p in zip(cov, ppv, strict=True)], n_none


def _check_curve(curve: Curve) -> list[float]:
    """The curve's coverages, after checking it is a non-empty list of (coverage, PPV) points.
    Passing risk_coverage's whole return value, (curve, n_unscored), is the likely slip."""
    try:
        if not curve:
            raise GuardFailure("empty curve")
        covs = [float(c) for c, _ in curve]
        [float(p) for _, p in curve]
    except (TypeError, ValueError):
        raise GuardFailure("a curve is a list of (coverage, PPV) points; risk_coverage returns "
                           "(curve, n_unscored)") from None
    if covs[0] <= 0 or covs[-1] > 1 or any(b <= a for a, b in zip(covs, covs[1:], strict=False)):
        raise GuardFailure("curve coverage is not strictly increasing within (0, 1]")
    return covs


def ppv_at_coverage(curve: Curve, c: float) -> float:
    """PPV at the smallest k whose coverage reaches c, with no interpolation between ranks.

    Interpolating would report a PPV no selection of pairs achieves; reading the next rank up
    reports one that a threshold on the score does achieve, at a coverage at most one pair
    above c. Coverages are compared as the floats k / n, so c = m / n with the curve's own n
    selects k = m exactly.
    """
    covs = _check_curve(curve)
    c = _coverage(c)
    k = bisect.bisect_left(covs, c)
    if k == len(covs):
        raise GuardFailure(f"coverage {c} lies beyond the curve's last point {covs[-1]}")
    return float(curve[k][1])


def aurc(curve: Curve) -> float:
    """Area under the risk-coverage curve: risk 1 - PPV against coverage, by the trapezoid
    rule from the curve's first point (coverage 1/n) to its last.

    Lower is better. The curve has no point at coverage 0, where PPV is undefined, so the
    area starts at the first rank rather than inventing a value there; a perfect ranking
    scores 0 if every pair is correct and the least any ordering can otherwise.
    """
    covs = _check_curve(curve)
    cov = np.asarray(covs)
    risk = 1.0 - np.asarray([float(p) for _, p in curve])
    if (risk < 0).any() or (risk > 1).any():
        raise GuardFailure("curve PPV outside [0, 1]")
    return float(np.sum(np.diff(cov) * (risk[1:] + risk[:-1]) / 2))


def _top_k_weights(sc: np.ndarray, tb: np.ndarray, k: int) -> np.ndarray:
    """Each record's share of the top-k selection, in input order: 1 above the block holding
    rank k, (k - a)/m for the m members of that block entered after a records, 0 below.

    The shares sum to k, and a pooled PPV built on them equals the curve's PPV at k, which is
    the expectation over the block's orderings. A family-wise PPV built on them is order-free
    too, but it is NOT that expectation: the family mean is not linear in the selection (which
    families are present depends on the ordering), so a family holding only a fractional
    share enters the mean as a whole family. It is a defined, order-free reading of a tied
    block, not the average over its orderings; the two agree when one family holds the block.
    """
    o = _order(sc, tb)
    s, t = sc[o], tb[o]
    tied = np.flatnonzero((s == s[k - 1]) & (t == t[k - 1]))
    a, b = int(tied[0]), int(tied[-1]) + 1
    ws = np.zeros(len(s))
    ws[:a] = 1.0
    ws[a:b] = (k - a) / (b - a)
    w = np.empty_like(ws)
    w[o] = ws
    return w


def _family_mean(families: list, w: np.ndarray, ok: np.ndarray) -> tuple[float, int]:
    """The unweighted mean over families of each family's (weighted) PPV, over the families
    holding any weight; and how many those are. Sums are exactly rounded (math.fsum), so the
    result does not depend on the order the records arrive in."""
    num: dict = {}
    den: dict = {}
    for f, wi, oi in zip(families, w.tolist(), ok.tolist(), strict=True):
        if f is None:
            raise GuardFailure("family weighting needs a family on every record; one has None")
        if wi > 0:
            num.setdefault(f, []).append(wi if oi else 0.0)
            den.setdefault(f, []).append(wi)
    if not den:
        raise GuardFailure("no family holds a selected record: the family mean is undefined")
    vals = [math.fsum(num[f]) / math.fsum(den[f]) for f in den]
    return math.fsum(vals) / len(vals), len(vals)


def matched_coverage_comparison(records: Sequence[Record], score_key: str, baseline_key: str,
                                coverage: float | Callable[[dict], bool],
                                tiebreak_key: str = "tiebreak",
                                by_family: bool = False) -> dict:
    """PPV of the score's ranking and of the baseline's at the same coverage, and the
    difference (score minus baseline).

    Both rankings are taken over the records BOTH score, so they share n and, at a given
    coverage, the same k: the comparison is between two selections of the same number of
    pairs from the same pool, which is what "matched" has to mean for it to be paired. How
    many records each key leaves unscored is reported.

    `coverage` is a number in (0, 1], or a predicate on a record, in which case the coverage
    is that set's fraction of the common pool -- `lambda r: r["p"] >= 0.99` gives the
    coverage of the p >= 0.99 set, recomputed on every bootstrap resample. Each PPV is read
    as ppv_at_coverage reads it (smallest k reaching the coverage). With by_family, each
    selection's PPV is the unweighted mean over families of the PPV of its pairs in that
    family, over the families it selects from (the held-out sets' weighting).
    """
    if score_key == baseline_key:
        raise GuardFailure(f"score and baseline are both {score_key!r}")
    s_raw = _column(records, score_key)
    b_raw = _column(records, baseline_key)
    common = [r for r, s, b in zip(records, s_raw, b_raw, strict=True)
              if s is not None and b is not None]
    if not common:
        raise GuardFailure(f"no record carries both {score_key!r} and {baseline_key!r}")
    n = len(common)
    if callable(coverage):
        m = int(_mask(common, coverage).sum())
        if m == 0:
            raise GuardFailure("the coverage set is empty on the records both rankings score")
        c = m / n
    else:
        m, c = None, _coverage(coverage)
    out = {"score_key": score_key, "baseline_key": baseline_key,
           "weighting": "family" if by_family else "pooled", "n": n,
           "n_unscored": {"score": sum(s is None for s in s_raw),
                          "baseline": sum(b is None for b in b_raw)},
           "coverage": c, "n_in_coverage_set": m}
    if by_family:
        out["n_families"] = {}
    for role, key in (("score", score_key), ("baseline", baseline_key)):
        _keep, sc, tb, ok, _ = _scored(common, key, tiebreak_key)
        cov, ppv = _curve(sc, tb, ok)
        k = int(np.searchsorted(cov, c, side="left")) + 1
        out["k"] = k
        if by_family:
            w = _top_k_weights(sc, tb, k)
            out[f"ppv_{role}"], out["n_families"][role] = _family_mean(
                _column(common, "family"), w, ok)
        else:
            out[f"ppv_{role}"] = float(ppv[k - 1])
    out["difference"] = out["ppv_score"] - out["ppv_baseline"]
    return out


# ----------------------------------------------------------------------------------
# PPV over a set, pooled and family-weighted
# ----------------------------------------------------------------------------------

def pooled_ppv(records: Sequence[Record], mask: Mask | None = None) -> float:
    """Correct pairs over all pairs, among the records the mask selects (all, if None).
    An empty selection has no PPV and is refused rather than reported as 0 or 1."""
    ok = _bools(_column(records, "correct"), "correct")
    sel = np.ones(len(ok), dtype=bool) if mask is None else _mask(records, mask)
    if not sel.any():
        raise GuardFailure("no record selected: PPV is undefined")
    return float(ok[sel].sum() / sel.sum())


def family_weighted_ppv(records: Sequence[Record], mask: Mask) -> float:
    """PPV per family over the records the mask selects, then the unweighted mean over the
    families that have at least one selected record.

    Each family counts once however many pairs it contributes (Mathews 2019, section 3:
    "each family contributes equally to the mean"), so one large family cannot carry the
    held-out result. A family with no selected record has no PPV and is left out of the mean,
    not counted as 0. Every record must carry a family: a None means the set has no family
    annotation and this is the wrong function for it.
    """
    ok = _bools(_column(records, "correct"), "correct")
    sel = _mask(records, mask)
    mean, _n = _family_mean(_column(records, "family"), sel.astype(np.float64), ok)
    return mean


# ----------------------------------------------------------------------------------
# paired bootstrap
# ----------------------------------------------------------------------------------

def paired_bootstrap(records: Sequence[Record], stat: Callable[[list], float],
                     n_boot: int = 10000, seed_tag: int = BOOTSTRAP_TAG,
                     by: str = "unit") -> tuple[float, float, float]:
    """stat on the records, and its 95% percentile interval over resamples of whole units.

    Units (record[by]: sequences, or families on the held-out sets) are drawn with
    replacement, as many as there are, and stat is recomputed on the records of the drawn
    units. Resampling units rather than pairs keeps the pairs of one sequence together, since
    they are not independent. The bootstrap is PAIRED because a comparison is made inside
    stat on each resample (e.g. the PPV difference at matched coverage), so score and
    baseline always see the same pairs.

    A unit drawn more than once has its later copies relabelled (record[by] becomes
    (unit, copy)) on copied dicts, so a family drawn twice counts twice in a family-weighted
    statistic instead of merging with itself. The same holds when the unit is not the family
    key but carries a whole family alone -- unit set to the family id, or the held-out set's
    one sequence per family -- so a later copy of such a unit also has record["family"]
    relabelled (family, copy). Without that, a family drawn twice merged with itself, each
    resample's family mean ran over only the distinct families drawn (about 63% of them),
    and the interval came out too narrow. A family spread over several units keeps its label:
    a sequence drawn twice is two sequences of the same family. stat must not mutate records.
    The draws come
    from default_rng(SeedSequence([SEED, seed_tag])) over the unit ids in sorted order, so the
    result depends on neither the order of the records nor anything outside the call. The
    interval is numpy's percentile (linear) at 2.5 and 97.5. A resample on which stat raises
    stops the bootstrap: dropping it would condition the interval on the resamples that work.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise GuardFailure(f"n_boot {n_boot!r} is not a positive integer")
    if isinstance(seed_tag, bool) or not isinstance(seed_tag, int) or seed_tag < 0:
        raise GuardFailure(f"seed_tag {seed_tag!r} is not a non-negative integer")
    groups: dict = {}
    holders: dict = {}                   # family -> the units holding its records
    for r, u in zip(records, _column(records, by), strict=True):
        if isinstance(u, _BOOL) or not isinstance(u, (str, int, np.integer)):
            raise GuardFailure(f"{by} {u!r} is not a str or int id")
        groups.setdefault(u, []).append(r)
        f = r.get("family")
        if by != "family" and f is not None:
            holders.setdefault(f, set()).add(u)
    if len(groups) < 2:
        raise GuardFailure(f"{len(groups)} distinct {by}: a bootstrap needs at least two")
    try:
        units = sorted(groups)
    except TypeError:
        raise GuardFailure(f"{by} ids mix str and int") from None
    sole = {f for f, us in holders.items() if len(us) == 1}   # families one unit carries whole

    def copy(r: dict, u, c: int) -> dict:
        out = {**r, by: (u, c)}
        if r.get("family") in sole:
            out["family"] = (r["family"], c)
        return out

    point = _finite(stat(list(records)), "stat on the full records")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, seed_tag]))
    draws = rng.integers(0, len(units), size=(n_boot, len(units)))
    vals = np.empty(n_boot)
    for b, row in enumerate(draws.tolist()):
        sub: list = []
        copies: dict = {}
        for x in row:
            c = copies.get(x, 0)
            copies[x] = c + 1
            u = units[x]
            sub.extend(groups[u] if c == 0 else (copy(r, u, c) for r in groups[u]))
        vals[b] = _finite(stat(sub), f"stat on resample {b}")
    half = (100.0 - BOOT_LEVEL) / 2
    lo, hi = np.percentile(vals, [half, 100.0 - half])
    return point, float(lo), float(hi)


# ----------------------------------------------------------------------------------
# the disagreement sets and the sequence-level fractions
# ----------------------------------------------------------------------------------

def disagreement_sets(records: Sequence[Record], rho_certified_key: str = "certified_l1",
                      p_hi: float = P_HI, p_lo: float = P_LO) -> tuple[list, list]:
    """D1, MFE pairs with p >= p_hi NOT certified at lambda = 1; D2, MFE pairs certified at
    lambda = 1 with p < p_lo. The records themselves, in input order.

    These are the result the design says can carry the paper: D1 is thermally confident but
    fragile along unpinned directions, D2 robust despite a moderate probability; their PPVs
    say whether the certificate catches something probability misses. The certified label
    must be a bool -- labels are always complete, so a None here is a bug upstream.
    """
    if not 0 <= p_lo <= p_hi <= 1:
        raise GuardFailure(f"need 0 <= p_lo <= p_hi <= 1, got p_lo={p_lo}, p_hi={p_hi}")
    cert = _bools(_column(records, rho_certified_key), rho_certified_key).tolist()
    p = _probabilities(_column(records, "p")).tolist()
    d1 = [r for r, pi, ci in zip(records, p, cert, strict=True) if pi >= p_hi and not ci]
    d2 = [r for r, pi, ci in zip(records, p, cert, strict=True) if ci and pi < p_lo]
    return d1, d2


def fraction_by_stratum(records_by_seq: Sequence[Record], label_key: str | Callable[[dict], bool],
                        strata_key: str = "stratum") -> dict:
    """{stratum: (n with the label, of in the stratum, fraction)}, strata in sorted order.

    For sequence-level labels, such as the fraction of sequences whose whole s* is certified
    at the published-table radius, by length stratum. `label_key` is a key or a predicate on
    a per-sequence record, since certfold.py nests the label under rec["sets"][set]. Strata
    holding no sequence do not appear. A stratum of None is a stratum of its own only if
    every sequence has it; a mixture means some records were stratified and some not.
    """
    if not records_by_seq:
        raise GuardFailure("no sequence records")
    raw = ([label_key(r) for r in records_by_seq] if callable(label_key)
           else _column(records_by_seq, label_key))
    labels = _bools(raw, f"label {label_key!r}").tolist()
    strata = _column(records_by_seq, strata_key)
    n_none = sum(s is None for s in strata)
    if 0 < n_none < len(strata):
        raise GuardFailure(f"{n_none} of {len(strata)} sequences carry no {strata_key!r}")
    tally: dict = {}
    try:
        for s, lab in zip(strata, labels, strict=True):
            n, of = tally.get(s, (0, 0))
            tally[s] = (n + int(lab), of + 1)
        keys = sorted(tally) if n_none == 0 else [None]
    except TypeError:
        raise GuardFailure(f"{strata_key!r} values are not hashable and mutually ordered") from None
    return {s: (tally[s][0], tally[s][1], tally[s][0] / tally[s][1]) for s in keys}
