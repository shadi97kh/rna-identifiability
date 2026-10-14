"""Certified folding: the test-set run's own wiring, on HAND-MADE toy records only.

scripts/certfold_run.py defines no metric: every number comes from scripts/certfold_metrics.py.
What it does decide is which records go where, and those decisions are held here:
  * the shard assignment is deterministic and spreads lengths evenly;
  * R2's lowest quartile, read off the REVERSED frozen ranking, is the hand-computed quartile,
    and a block tied across the boundary is shared so the two sides add up to the whole;
  * a sampling baseline read with fractional weights alone shares k and n with rho_env;
  * the per-pair records carry the unit PRED names (the family on the held-out set) and refuse
    frequencies that are not over the MFE pairs;
  * --resume drops a torn LAST line and refuses a torn line anywhere else.
None of these records stands in for a result.
"""

import json
import pathlib
import sys

import pytest

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold_run as cr  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402


def mfe(unit, correct, rho, gap, p=0.995, capped=False, family=None, pair=(1, 10)):
    return {"kind": "mfe", "unit": unit, "family": family, "pair": pair, "correct": correct,
            "rho_env": rho, "neg_rho": None if rho is None else -rho, "capped": capped,
            "tiebreak": gap, "neg_tiebreak": -gap, "gap": gap, "flat": 0.0, "p": p,
            "freq_env": 1.0, "freq_noise": 1.0, "certified_l1": True}


def test_shards_are_deterministic_and_balanced():
    rows = [{"index": k, "seq": "A" * (20 + (k * 37) % 400)} for k in range(101)]
    a, b = cr.shard_of(rows, 8), cr.shard_of(list(reversed(rows)), 8)
    assert a == b                                   # the input order does not matter
    sizes = [sum(v == w for v in a.values()) for w in range(8)]
    assert max(sizes) - min(sizes) <= 1
    longest = sorted(rows, key=lambda r: -len(r["seq"]))[:8]
    assert sorted(a[r["index"]] for r in longest) == list(range(8))   # one each


def test_r2_quartile_is_the_hand_computed_one():
    # 8 pairs, distinct rho; the lowest quartile is the 2 smallest rho: one correct of two (0.5);
    # the rest, 6 pairs, hold 5 correct (5/6). Difference 0.5 - 5/6.
    recs = [mfe(k, c, rho, gap=1.0) for k, (c, rho) in enumerate(
        [(True, 0.1), (False, 0.2), (True, 0.3), (True, 0.4), (True, 0.5), (False, 0.6),
         (True, 0.7), (True, 0.8)])]
    assert cr.stat_r2_quartile(recs) == pytest.approx(0.5 - 5 / 6, abs=1e-15)


def test_r2_quartile_shares_a_block_tied_across_the_boundary():
    # 4 pairs, all tied on rho and gap: k_low = 1. Each side takes the block at its expectation,
    # the block's PPV (3 of 4), so the difference is exactly 0 whatever order they are listed in.
    recs = [mfe(k, c, 1.0, 2.0) for k, c in enumerate([True, True, False, True])]
    assert cr.stat_r2_quartile(recs) == 0.0
    assert cr.stat_r2_quartile(list(reversed(recs))) == 0.0


def test_r2_pool_is_p_ge_099_with_a_rho():
    recs = [mfe(0, True, 0.1, 1.0, p=0.98), mfe(1, True, None, 1.0), mfe(2, False, 0.3, 1.0)]
    assert [r["unit"] for r in cr._r2_pool(recs)] == [2]


def test_fractional_reading_shares_k_and_n_with_rho():
    # p >= 0.99 holds for 2 of 4, so k = 2. Under "flat" the four baseline scores (all 1.0) tie
    # in one block of PPV 1/2; rho_env's top 2 are both correct. Difference 1 - 1/2.
    recs = [mfe(0, True, 3.0, 1.0), mfe(1, True, 2.0, 1.0), mfe(2, False, 1.0, 1.0, p=0.5),
            mfe(3, False, 0.5, 1.0, p=0.5)]
    assert cr.stat_matched(recs, "freq_env", "flat", False) == pytest.approx(0.5)
    # with the registered tiebreak the frequencies' ties go by the gap, here also all tied
    assert cr.stat_matched(recs, "freq_env", "tiebreak", False) == pytest.approx(0.5)


def _seq_rec(index, tset, family=None):
    return {"index": index, "set": tset, "family": family, "stratum": 0,
            "ref_db": "((((....))))", "mea": {"1": "((((....))))"},
            "centroid": {"1": [[1, 12], [2, 11]]},
            "pairs": [{"pair": [i, 13 - i], "p": 0.9, "nominal_gap_kcal": 0.5 * i,
                       "margins_dcal": {"envelope:1": 10 * (i - 2)},
                       "rho_env": None if i == 4 else {"rho": 0.25 * i, "capped": False,
                                                       "tied_at_zero": False}}
                      for i in range(1, 5)]}


def test_pair_records_units_and_keys():
    freq = {k: {"env": [[i, 13 - i, 1.0] for i in range(1, 5)],
                "noise": [[i, 13 - i, 0.5] for i in range(1, 5)]} for k in (0, 1)}
    out = cr.pair_records([_seq_rec(0, "TS0"), _seq_rec(1, "held-out", "RF00001")], freq)
    ts0 = [r for r in out["TS0"] if r["kind"] == "mfe"]
    ho = [r for r in out["held-out"] if r["kind"] == "mfe"]
    assert {r["unit"] for r in ts0} == {0} and {r["unit"] for r in ho} == {"RF00001"}
    assert [r["rho_status"] for r in ts0] == ["finite"] * 3 + ["budget"]
    assert [r["certified_l1"] for r in ts0] == [False, False, True, True]
    assert all(r["neg_tiebreak"] == -r["tiebreak"] for r in ts0)
    assert {r["kind"] for r in out["TS0"]} == {"unit", "mfe", "mea:1", "centroid:1"}
    assert sum(r["kind"] == "unit" for r in out["TS0"]) == 1
    assert all(r["correct"] for r in out["TS0"] if r["kind"] == "mea:1")


def test_pair_records_refuse_frequencies_off_the_mfe_pairs():
    freq = {0: {"env": [[1, 12, 1.0]], "noise": [[i, 13 - i, 0.5] for i in range(1, 5)]}}
    with pytest.raises(GuardFailure):
        cr.pair_records([_seq_rec(0, "TS0")], freq)


def test_resume_drops_a_torn_last_line_only(tmp_path):
    f = tmp_path / "run_0.jsonl"
    f.write_text(json.dumps({"index": 3}) + "\n" + json.dumps({"index": 7}) + "\n" + '{"ind')
    assert cr._done_indices(f) == {3, 7}
    assert f.read_text().count("\n") == 2
    f.write_text('{"ind\n' + json.dumps({"index": 7}) + "\n")
    with pytest.raises(GuardFailure):
        cr._done_indices(f)
    assert cr._done_indices(tmp_path / "absent.jsonl") == set()


def test_pump_drops_only_the_zuker_line(tmp_path):
    import subprocess
    script = ("import sys\n"
              "print('[ERROR]   Added 19 bps', file=sys.stderr)\n"
              "print('[ERROR]   Added 19 bps and more', file=sys.stderr)\n"
              "print('Traceback (most recent call last):', file=sys.stderr)\n"
              "print('[ERROR]   Added 0 bps', file=sys.stderr)\n")
    p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    dropped = [0]
    cr._pump(p, tmp_path / "w.log", False, dropped, 0)
    p.wait()
    assert dropped == [2]
    assert (tmp_path / "w.log").read_text().splitlines() == [
        "[ERROR]   Added 19 bps and more", "Traceback (most recent call last):"]


def test_task_records_change_no_statistic_and_keep_every_unit():
    freq = {k: {"env": [[i, 13 - i, 1.0] for i in range(1, 5)],
                "noise": [[i, 13 - i, 0.5] for i in range(1, 5)]} for k in range(6)}
    seqs = [_seq_rec(k, "TS0") for k in range(6)]
    seqs[5]["pairs"] = []                              # an open-chain MFE: still a unit
    freq[5] = {"env": [], "noise": []}
    recs = cr.pair_records(seqs, freq)["TS0"]
    units = {r["unit"] for r in recs}
    assert units == set(range(6))
    for stat, args in (("aurc", ("p", "tiebreak")), ("disagreement", ("D1", False)),
                       ("point", ("mea:1", False))):
        sub = cr.task_records(recs, stat, args)
        assert {r["unit"] for r in sub} == units
        assert len(sub) < len(recs)
        assert {r["kind"] for r in sub} <= {"unit", "mfe", "mea:1"}
        try:
            want = cr.STATS[stat](recs, *args)
        except GuardFailure:
            with pytest.raises(GuardFailure):
                cr.STATS[stat](sub, *args)
            continue
        assert cr.STATS[stat](sub, *args) == want
