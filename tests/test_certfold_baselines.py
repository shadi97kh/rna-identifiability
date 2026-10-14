"""Certified folding, the baselines: scores and predictions the accuracy test compares against.

`scripts.certfold_baselines` never reads a reference structure, and neither do these tests: the
TR0 rows are used for their SEQUENCES only, and nothing here computes an accuracy. What is held:

  * the ensemble: the probability matrix is one (shape, triangle, per-base totals); MEA returns
    balanced structures over admissible pairs; the gamma-centroid at gamma = 1 is ViennaRNA's own
    centroid, and at every gamma it is a secondary structure inside the threshold set;
  * the gamma > 1 DP against brute force on small hand-made and seeded toy matrices, since on
    TR0 the threshold set is usually NOT a structure there and the DP is what answers;
  * the draws: envelope draws inside both constraints to 1e-9, a pure rescaling across lam,
    reproducible by tag; noise draws leave an INF entry exactly alone;
  * the frequencies: a zero shift reproduces the Turner 2004 MFE exactly (frequency 1 on its
    pairs, 0 elsewhere), drawn tables give fractions in [0, 1], and Turner 2004 is back after.
"""

import itertools
import pathlib
import sys

import numpy as np
import pytest

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold as certfold  # noqa: E402
from scripts.certfold_baselines import (  # noqa: E402
    GAMMAS,
    SEED,
    _energy,
    _nested,
    _ninio_correction,
    _pairs,
    envelope_draws,
    gamma_centroid_pairs,
    mea_structures,
    noise_draws,
    pair_frequencies,
    pair_probabilities,
    shifted_theta,
)
from scripts.turner_nullspace import PTYPE  # noqa: E402
from scripts.turner_par import template  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402
from scripts.typeb_common import INF_DCAL, ninio_correction  # noqa: E402

SEED_TAG_TOYS = 39               # test-only: the random toy probability matrices


@pytest.fixture(scope="module")
def tr0_seqs():
    """TR0 sequences by (length, name); the reference column is never read."""
    rows, _ = certfold.tr0_rows()
    return [r[2] for r in sorted(rows, key=lambda r: (len(r[2]), r[1]))]


@pytest.fixture(scope="module")
def short3(tr0_seqs):
    return tr0_seqs[:3]


@pytest.fixture(scope="module")
def few(tr0_seqs):
    """The three shortest and three at the length quartiles: a few TR0 sequences, seconds."""
    m = len(tr0_seqs)
    return tr0_seqs[:3] + [tr0_seqs[m // 4], tr0_seqs[m // 2], tr0_seqs[3 * m // 4]]


@pytest.fixture(scope="module")
def B():
    if not (certfold.SHARDS / "basis_tr0.npz").exists():
        pytest.skip("the TR0 basis shard is gitignored; rerun certfold --phase build-basis")
    return certfold.load_tr0_basis()


def test_seed_is_certfolds():
    assert SEED == certfold.SEED == 20261013


# ----------------------------------------------------------------------------------
# the ensemble
# ----------------------------------------------------------------------------------
def test_pair_probabilities_is_a_probability_matrix(short3):
    for seq in short3:
        P = pair_probabilities(seq)
        n = len(seq)
        assert P.shape == (n + 1, n + 1) and P.dtype == np.float64
        assert not np.tril(P).any() and not P[0].any() and not P[:, 0].any()
        assert P.min() >= 0.0 and P.max() <= 1.0 + 1e-9
        assert (P.sum(axis=0) + P.sum(axis=1)).max() <= 1.0 + 1e-9
        assert P.max() > 0.0


def test_mea_structures_are_balanced(few):
    for seq in few:
        P = pair_probabilities(seq)
        got = mea_structures(seq)
        assert set(got) == set(GAMMAS)
        for g, s in got.items():
            assert len(s) == len(seq), g
            for i, j in _pairs(s):          # raises on anything unbalanced
                assert seq[i - 1] + seq[j - 1] in PTYPE and j - i > 3, (g, i, j)
                assert P[i, j] > 0.0, (g, i, j)
        # every gamma reads the same partition function: asking in another order changes nothing
        assert mea_structures(seq, tuple(reversed(GAMMAS))) == got


def test_gamma_centroid_at_one_is_viennarnas_centroid(few):
    import RNA

    from thermo.vienna_ref import reference_md

    for seq in few:
        fc = RNA.fold_compound(seq, reference_md())
        _s, e = fc.mfe()
        fc.exp_params_rescale(e)
        fc.pf()
        centroid, _dist = fc.centroid()
        assert gamma_centroid_pairs(pair_probabilities(seq), (1.0,))[1.0] == set(_pairs(centroid))


def test_gamma_centroid_is_a_structure_inside_the_threshold_set(few):
    for seq in few:
        P = pair_probabilities(seq)
        for g, got in gamma_centroid_pairs(P).items():
            ii, jj = np.nonzero(P > 1.0 / (g + 1.0))
            above = set(zip(ii.tolist(), jj.tolist(), strict=True))
            assert _nested(got) and got <= above, g
            if g <= 1.0:
                assert got == above, g


def _brute_max_gain(P, thr):
    """The maximum-gain nested subset of the threshold set by enumeration, and whether unique."""
    ii, jj = np.nonzero(np.triu(P, 1) > thr)
    cand = list(zip(ii.tolist(), jj.tolist(), strict=True))
    assert len(cand) <= 12, "brute force is for toys"
    best, gains = None, []
    for m in range(len(cand) + 1):
        for sub in itertools.combinations(cand, m):
            if _nested(sub):
                gain = sum(P[i, j] - thr for i, j in sub)
                gains.append(gain)
                if best is None or gain > best[0]:
                    best = (gain, set(sub))
    gains.sort()
    unique = len(gains) < 2 or gains[-1] - gains[-2] > 1e-12
    return best[1], unique


def _toy(n, entries):
    P = np.zeros((n + 1, n + 1))
    for (i, j), p in entries.items():
        P[i, j] = p
    return P


def test_gamma_centroid_resolves_conflicts_by_max_gain():
    shared = _toy(16, {(1, 8): 0.45, (1, 9): 0.40, (2, 7): 0.5, (10, 15): 0.9})
    crossing = _toy(16, {(1, 10): 0.4, (5, 15): 0.45, (2, 9): 0.3, (6, 14): 0.25,
                         (11, 16): 0.35})
    for P in (shared, crossing):
        got = gamma_centroid_pairs(P)
        for g in GAMMAS:
            want, unique = _brute_max_gain(P, 1.0 / (g + 1.0))
            assert unique, g
            assert got[g] == want, (g, got[g], want)
    # the DP path really ran: at gamma = 2 the threshold set of `shared` is not a structure
    assert not _nested({(1, 8), (1, 9), (2, 7), (10, 15)})
    assert gamma_centroid_pairs(shared, (2.0,))[2.0] == {(1, 8), (2, 7), (10, 15)}


def test_gamma_centroid_dp_against_brute_force_on_seeded_toys():
    rng = np.random.default_rng(np.random.SeedSequence([SEED, SEED_TAG_TOYS]))
    n, checked = 14, 0
    for _ in range(60):
        pairs = {}
        for _k in range(int(rng.integers(3, 10))):
            i = int(rng.integers(1, n - 3))
            j = int(rng.integers(i + 4, n + 1))
            pairs[(i, j)] = float(rng.uniform(0.15, 0.6))
        P = _toy(n, pairs)
        tot = P.sum(axis=0) + P.sum(axis=1)
        if tot.max() > 1.0:
            P = P / tot.max()               # a valid probability matrix: no base above 1
        got = gamma_centroid_pairs(P, (2.0, 4.0))
        for g in (2.0, 4.0):
            want, unique = _brute_max_gain(P, 1.0 / (g + 1.0))
            if unique:
                assert got[g] == want, (g, pairs)
                checked += 1
    assert checked > 60


def test_gamma_centroid_refuses_what_is_not_a_probability_matrix():
    with pytest.raises(GuardFailure):        # two pairs above 0.5 that cross
        gamma_centroid_pairs(_toy(16, {(1, 10): 0.6, (5, 15): 0.6}), (1.0,))
    lower = _toy(16, {(1, 10): 0.6})
    lower[10, 1] = 0.6
    with pytest.raises(GuardFailure):
        gamma_centroid_pairs(lower)
    with pytest.raises(GuardFailure):        # base 1 paired with total probability 1.2
        gamma_centroid_pairs(_toy(16, {(1, 10): 0.6, (1, 12): 0.6}))


# ----------------------------------------------------------------------------------
# the draws
# ----------------------------------------------------------------------------------
def test_envelope_draws_satisfy_both_constraints(B):
    shifts, rec = envelope_draws(B)
    B4, used = B["B4"], B["used"]
    r_max, delta = B["envelope"]["r_max_kcal"], B["envelope"]["delta_kcal"]
    off = np.ones(len(B["cols"]), dtype=bool)
    off[used] = False
    assert len(shifts) == rec["n_draws"] == 200
    for s in shifts:
        assert s.shape == (len(B["cols"]),) and not s[off].any()
        t = B4.T @ s[used]
        assert np.abs(B4 @ t - s[used]).max() <= 1e-9          # in span(B4)
        assert np.linalg.norm(t) <= r_max + 1e-9               # the ball
        assert np.abs(s).max() <= delta + 1e-9                 # the box
        assert np.all(np.abs(s) <= B["u_env"] + 1e-9)          # hence u_e on every entry
    assert rec["n_proposals"] >= 200
    assert rec["acceptance_rate"] == 200 / rec["n_proposals"]
    assert 0.0 < rec["acceptance_rate"] <= 1.0
    assert rec["seed"] == [SEED, 30]


def test_envelope_draws_rescale_with_lam_and_reproduce_by_tag(B):
    one, _ = envelope_draws(B, n_draws=20)
    half, rec = envelope_draws(B, n_draws=20, lam=0.5)
    r_max, delta = B["envelope"]["r_max_kcal"], B["envelope"]["delta_kcal"]
    for a, h in zip(one, half, strict=True):
        assert np.abs(h - 0.5 * a).max() <= 1e-15 * max(1.0, np.abs(a).max())
        assert np.abs(h).max() <= 0.5 * delta + 1e-9
        assert np.linalg.norm(B["B4"].T @ h[B["used"]]) <= 0.5 * r_max + 1e-9
    assert rec["lam"] == 0.5
    again, _ = envelope_draws(B, n_draws=20)
    other, _ = envelope_draws(B, n_draws=20, seed_tag=32)
    assert all(np.array_equal(a, b) for a, b in zip(one, again, strict=True))
    assert not any(np.array_equal(a, b) for a, b in zip(one, other, strict=True))


def test_draws_refuse_a_tag_or_count_that_is_not_a_plain_int():
    cols = list(template()["theta"])[:5]
    for bad in (True, -1, 30.0):            # True would replay tag 1's stream
        with pytest.raises(GuardFailure):
            noise_draws(cols, n_draws=1, seed_tag=bad)
    with pytest.raises(GuardFailure):
        noise_draws(cols, n_draws=2.5)


def test_noise_draws_leave_inf_entries_alone():
    cols = list(template()["theta"])
    planted = dict(template()["theta"])
    inf_keys = [("int22", 1, 1, 1, 1, 1, 2), ("hairpin", 7)]
    for k in inf_keys:
        planted[k] = INF_DCAL
    shifts, rec = noise_draws(cols, n_draws=4, theta0=planted)
    at = [cols.index(k) for k in inf_keys]
    rest = np.ones(len(cols), dtype=bool)
    rest[at] = False
    for s in shifts:
        assert np.all(s[at] == 0.0)
        assert np.all(s[rest] != 0.0)
    assert rec["n_entries_inf_left_alone"] == 2
    assert rec["n_entries_perturbed"] == len(cols) - 2
    # on Turner 2004 itself the rule is vacuous, and the record says so
    _s, rec04 = noise_draws(cols, n_draws=1)
    assert rec04["n_entries_inf_left_alone"] == 0 and rec04["seed"] == [SEED, 31]


def test_shifted_theta_rounds_and_never_moves_inf():
    T = template()["theta"]
    cols = [("stack", 1, 2), ("int11", 1, 1, 1, 1), ("hairpin", 7)]
    theta, resid, n = shifted_theta(T, cols, np.array([0.1234, -0.005, 0.0]))
    assert theta[cols[0]] == T[cols[0]] + 12 and theta[cols[1]] == T[cols[1]]   # -0.5 -> 0
    assert n == 1 and abs(resid - 0.005) < 1e-12 and len(theta) == len(T)
    planted = dict(T)
    planted[cols[2]] = INF_DCAL
    theta, _r, _n = shifted_theta(planted, cols, np.array([0.1, 0.0, 0.0]))
    assert theta[cols[2]] == INF_DCAL
    with pytest.raises(GuardFailure):
        shifted_theta(planted, cols, np.array([0.1, 0.0, 0.2]))


# ----------------------------------------------------------------------------------
# the frequencies
# ----------------------------------------------------------------------------------
def _turner_restored():
    import RNA

    from thermo.vienna_ref import assert_turner2004_params, reference_md

    assert_turner2004_params(RNA.fold_compound("GGGAAACCCAGCUUCGGCUG", reference_md()))


def test_pair_frequencies_zero_shift_is_the_turner2004_mfe(short3, tmp_path):
    import RNA

    from thermo.vienna_ref import reference_md

    cols = list(template()["theta"])
    freq, rec = pair_frequencies(short3, [np.zeros(len(cols))], template()["theta"], tmp_path,
                                 cols=cols)
    for seq, f in zip(short3, freq, strict=True):
        star, _e = RNA.fold_compound(seq, reference_md()).mfe()
        assert f == {p: 1.0 for p in _pairs(star)}
    d = rec["draws"][0]
    assert d["params_match_pass"] and d["params_match"] == {}
    assert d["max_rounding_kcal"] == 0.0 and d["n_entries_changed_dcal"] == 0
    assert d["carry_rows"] == [] and d["n_sequences_mfe_differs_from_turner2004"] == 0
    assert rec["theta0_equals_turner2004"] and rec["turner2004_restored"]
    assert list(tmp_path.iterdir()) == []                      # the table file is removed
    _turner_restored()


def test_pair_frequencies_are_fractions_under_drawn_tables(short3, B, tmp_path):
    env, _ = envelope_draws(B, n_draws=3)
    noise, _ = noise_draws(B["cols"], n_draws=3)
    before = [pair_probabilities(s) for s in short3]
    freq, rec = pair_frequencies(short3, env + noise, template()["theta"], tmp_path,
                                 cols=B["cols"])
    assert rec["n_draws"] == 6 and len(freq) == 3
    for f in freq:
        for (i, j), x in f.items():
            assert 1 <= i < j and 0.0 < x <= 1.0
            assert abs(x * 6 - round(x * 6)) < 1e-12
    assert rec["all_params_match_pass"]
    assert rec["max_rounding_kcal"] <= 0.005 + 1e-12
    assert rec["mfe_energy_identity"] == {"n_folds": 18, "n_mismatch": 0}
    for d in rec["draws"]:
        assert d["carry_rows"] and d["n_entries_changed_dcal"] > 0
    _turner_restored()
    # the Boltzmann table is Turner 2004's again too: the same probabilities, bit for bit
    for s, P in zip(short3, before, strict=True):
        assert np.array_equal(pair_probabilities(s), P)


def test_pair_frequencies_refuses_a_shift_without_its_columns(short3, tmp_path):
    cols = list(template()["theta"])
    with pytest.raises(GuardFailure):
        pair_frequencies(short3, [np.zeros(len(cols) - 1)], template()["theta"], tmp_path,
                         cols=cols)
    _turner_restored()


def test_ninio_correction_is_exact_where_the_shared_one_is_not(few, tmp_path):
    """A 2x3 loop adds ninio_m unclipped in ViennaRNA; a 1x4 loop adds MIN2(cap, 3 m).

    With a written table at m = 400 > cap = 300, eval_structure must equal counts . theta +
    _ninio_correction on both, and typeb_common.ninio_correction is 100 dcal off on the 2x3.
    At Turner 2004's m <= cap the two corrections agree on every structure.
    """
    import RNA

    from scripts.turner_nullspace import Extractor
    from scripts.turner_par import loaded, write_par
    from thermo.vienna_ref import reference_md

    T = template()
    th = dict(T["theta"])
    th[("ninio_m",)] = 400
    assert th[("ninio_max",)] == 300
    ex = Extractor(T["special"], 0, 0, lxc=T["lxc"])
    two3 = ("GGGAAGCGAAAACGCAAACCC", "(((..(((....)))...)))")
    one4 = ("GGGAGCGAAAACGCAAAACCC", "(((.(((....)))....)))")
    with loaded(write_par(th, tmp_path / "ninio.par")):
        for seq, db in (two3, one4):
            got = int(round(RNA.fold_compound(seq, reference_md()).eval_structure(db) * 100))
            assert got == _energy(ex.counts(seq, db), th) + _ninio_correction(db, 400, 300), db
    assert _ninio_correction(two3[1], 400, 300) == 0
    assert ninio_correction(two3[1], 400, 300) == -100
    assert _ninio_correction(one4[1], 400, 300) == ninio_correction(one4[1], 400, 300) == -900
    _turner_restored()
    # through pair_frequencies: m past the cap, and every carry row really moves (the cap
    # absorbs the linear change on the asym >= 5 rows, so those must not be the ones checked)
    _f, rec = pair_frequencies(few[:2], [np.array([3.4])], T["theta"], tmp_path,
                               cols=[("ninio_m",)])
    rows = rec["draws"][0]["carry_rows"]
    assert len(rows) == 3 and all(r["change_vs_turner2004_dcal"] != 0 for r in rows)
    _turner_restored()
    for seq in few:
        for db in mea_structures(seq).values():
            assert _ninio_correction(db, 60, 300) == ninio_correction(db, 60, 300) == 0
            assert _ninio_correction(db, 70, 330) == ninio_correction(db, 70, 330)


def test_noise_moves_both_ends_of_one_parameter_together():
    """A stacking parameter seen from either end is ONE parameter: its two file entries get the
    same noise, so the 180-degree flip invariance survives the perturbation."""
    from scripts.certfold import load_tr0_basis
    from scripts.certfold_baselines import noise_draws
    from scripts.turner_nullspace import tie_map

    B = load_tr0_basis()
    cols = B["cols"]
    shifts, rec = noise_draws(cols, n_draws=3, seed_tag=31)
    tied, T = tie_map(cols)
    cls = T.argmax(axis=1)
    assert rec["n_tie_classes"] == len(tied) < len(cols)
    for s in shifts:
        for c in range(len(tied)):
            members = s[cls == c]
            assert np.all(members == members[0]), f"class {tied[c]} drew different noise"
