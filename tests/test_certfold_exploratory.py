"""Certified folding, the exploratory analyses: their wiring, on HAND-MADE toy records only.

Held here:
  * E2's statistic is PPV(D1) - PPV(certified) among p >= 0.99 pairs, by hand;
  * the stratified bootstrap draws each stratum's units from that stratum only, is
    deterministic, pools by pair count, and reports an undefined interval instead of dropping
    resamples;
  * the length score is attached to MFE records only and leaves the inputs alone.
None of these records stands in for a result.
"""

import pathlib
import sys

import pytest

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold_exploratory as ce  # noqa: E402


def mfe(index, correct, rho, p=0.995, cert=True, gap=1.0, pair=(1, 10)):
    return {"kind": "mfe", "unit": index, "family": None, "index": index, "pair": pair,
            "correct": correct, "rho_env": rho, "neg_rho": -rho, "capped": False,
            "tiebreak": gap, "neg_tiebreak": -gap, "gap": gap, "flat": 0.0, "p": p,
            "freq_env": 1.0, "freq_noise": 1.0, "certified_l1": cert}


def unit(index):
    return {"kind": "unit", "unit": index, "family": None, "index": index}


def test_e2_is_d1_ppv_minus_certified_ppv():
    # p >= 0.99: D1 (not certified) = 2 pairs, 1 correct (0.5); certified = 3 pairs, all
    # correct (1.0). The p = 0.5 pair is outside both. Difference -0.5.
    recs = [mfe(0, True, 1.0, cert=False), mfe(0, False, 1.0, cert=False),
            mfe(1, True, 2.0), mfe(1, True, 2.0), mfe(2, True, 2.0),
            mfe(2, False, 2.0, p=0.5, cert=False)]
    assert ce.e2_stat(recs) == pytest.approx(-0.5)
    assert ce.e2_weight(recs) == 5


def _groups():
    # stratum 0: sequences 0-3; stratum 1: sequences 10-13 (one with no pair at all)
    g = {0: {}, 1: {}}
    for k in range(4):
        g[0][k] = [unit(k), mfe(k, k % 2 == 0, 0.1 * (k + 1)), mfe(k, True, 1.0 + k),
                   mfe(k, k != 1, 0.5, cert=k > 1)]
    for k in range(10, 14):
        g[1][k] = [unit(k)] + ([] if k == 13 else
                               [mfe(k, k != 11, 0.2 * (k - 9)), mfe(k, True, 3.0 + k),
                                mfe(k, True, 0.4, cert=k != 10)])
    return g


def test_stratified_bootstrap_is_deterministic_and_stays_in_strata(monkeypatch):
    seen = []
    real = ce.e1_stat

    def spy(recs):
        seen.append({r["index"] for r in recs})
        return real(recs)

    monkeypatch.setitem(ce.STRATIFIED, "E1_r2_within_strata", (spy, ce.e1_weight))
    a = ce.stratified_bootstrap(_groups(), n_boot=50)
    for idx in seen:                         # every call sees one stratum's sequences only
        assert idx <= {0, 1, 2, 3} or idx <= {10, 11, 12, 13}
    monkeypatch.setitem(ce.STRATIFIED, "E1_r2_within_strata", (real, ce.e1_weight))
    b = ce.stratified_bootstrap(_groups(), n_boot=50)
    c = ce.stratified_bootstrap(_groups(), n_boot=50)
    assert b == c
    assert a["E1_r2_within_strata"]["pooled_point"] == b["E1_r2_within_strata"]["pooled_point"]


def test_pooled_point_is_weighted_by_pair_count():
    out = ce.stratified_bootstrap(_groups(), n_boot=20)["E1_r2_within_strata"]
    per = out["by_stratum"]
    want = sum(per[s]["n_pairs"] * per[s]["point"] for s in out["strata_pooled"]) / sum(
        per[s]["n_pairs"] for s in out["strata_pooled"])
    assert out["pooled_point"] == pytest.approx(want)
    assert {s: per[s]["n_pairs"] for s in per} == {0: 12, 1: 9}


def test_an_undefined_resample_leaves_the_interval_undefined():
    # stratum 1 with one D1 pair in one sequence: resamples that miss it have an empty D1
    g = {1: {10: [unit(10), mfe(10, True, 1.0, cert=False), mfe(10, True, 1.0)],
             11: [unit(11), mfe(11, False, 1.0)], 12: [unit(12), mfe(12, True, 1.0)]}}
    out = ce.stratified_bootstrap(g, n_boot=200)["E2_d1_within_strata"]
    assert out["by_stratum"][1]["point"] is not None
    assert out["by_stratum"][1]["resamples_undefined"] > 0
    assert out["by_stratum"][1]["interval"] is None
    assert out["pooled_interval"] is None and out["pooled_resamples_undefined"] > 0


def test_length_score_only_on_mfe_records_and_inputs_untouched():
    recs = [unit(0), mfe(0, True, 1.0), {"kind": "mea:1", "unit": 0, "family": None,
                                         "index": 0, "pair": (1, 10), "correct": True}]
    out = ce.with_length(recs, {0: 77})
    assert [r.get("neg_len") for r in out] == [None, -77.0, None]
    assert "neg_len" not in recs[1]
