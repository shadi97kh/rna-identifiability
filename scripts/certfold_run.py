#!/usr/bin/env python3
"""CERTIFIED FOLDING, THE TEST-SET RUN AND ITS ANALYSIS. Registration 1.

WHAT THIS IS. The module PRED's code_claude describes: "The test-set run's driver and the
analysis's wiring are in a module that imports certfold and is not imported by it, written after
this registration and before any test-set fold beyond G5's, its commit recorded in the run's
record; it may only call the frozen metrics." So nothing here defines a label, a radius, a
baseline or a metric. Labels and radii come from certfold.certify_sequence, the baselines from
scripts/certfold_baselines.py, and every accuracy number from scripts/certfold_metrics.py. This
module decides only which rows go where and in what order, and it states each such decision
where it is made.

PHASES
  run           every test sequence -- TS0 (1,288), the held-out one-per-family set (1,641) and
                the bpRNA-new subsample (2,500), exactly the rows G5 attacked -- labelled under
                every set to completion, rho within the registered budget, MEA and the
                gamma-centroid beside them; then, per shard, the two sampling baselines (200
                envelope draws, tag 30; 200 independent-noise draws, tag 31). Refuses unless G5
                passed on the registration and the artifact carrying it is committed.
  analyse       R1 to R3, the disagreement sets, the descriptive fractions by length stratum and
                the exposure quartiles, into metrics.json["analysis"]. Refuses unless the run is
                committed and the code is the run's.

Run, in .venv-certfold (the registration requires the compiled callback):
  .venv-certfold/bin/python scripts/certfold_run.py --phase run       # then commit
  .venv-certfold/bin/python scripts/certfold_run.py --phase analyse   # then commit

Guards raise GuardFailure and are never repaired.
"""

from __future__ import annotations

import collections
import json
import multiprocessing
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold as cf  # noqa: E402
from scripts.certfold_metrics import (  # noqa: E402
    P_HI,
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
from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

ME = "scripts/certfold_run.py"
RUN_WORKERS = 8                  # of 16 cores, at nice 10: the house pattern
RHO_ORDER_TAG = 12               # PRED runtime_claude: rho's pairs in the order [SEED, 12, index]
N_DRAWS = 200                    # PRED accuracy_test_claude: 200 envelope draws, 200 noise draws
N_BOOT = 10000                   # PRED: 10,000 paired bootstrap resamples (seed tag 40)
BOOT_PROCS = 12                  # processes the analysis spreads its bootstraps over
SETS = ("TS0", "held-out", "bpRNA-new")
PRIMARY = "envelope:1"           # the primary set's label key (certfold.run_sets)
# the registered baselines the rho_env ranking is compared with: (name, score key, tiebreak key).
# EVERY ranking breaks ties by the nominal gap; the two sampling baselines are ALSO read with
# fractional weights alone ("flat", a constant), beside the registered reading (PRED ties)
BASELINES = (("p", "p", "tiebreak"), ("nominal_gap", "gap", "tiebreak"),
             ("freq_envelope", "freq_env", "tiebreak"), ("freq_noise", "freq_noise", "tiebreak"),
             ("freq_envelope_fractional", "freq_env", "flat"),
             ("freq_noise_fractional", "freq_noise", "flat"))
CURVE_GRID = 100                 # risk-coverage curves are stored at coverage 1/100, ..., 1


# ----------------------------------------------------------------------------------
# the rows: exactly G5's, with the references and families G5 did not need
# ----------------------------------------------------------------------------------
def test_rows() -> tuple[list[dict], dict]:
    """Every test sequence in g5_test_rows's order, with its reference, family and stratum.

    Refuses unless (set, name, sequence) equals g5_test_rows row for row, so the run labels the
    sequences G5 attacked and no others. The family is the held-out row's first field (an Rfam
    accession); TS0 and bpRNA-new carry none. The stratum is the pilot's TR0 length stratum,
    a length on a boundary going to the lower one (certfold_data.length_stratum).
    """
    import scripts.certfold_data as cd

    ts0 = cd.ts0_rows()
    ho, _ = cd.heldout_one_per_family()
    bn, _ = cd.bprna_new_subsample(cf.BPRNA_NEW_N)
    bounds = cd.TR0_STRATA
    rows = []
    for tset, src in (("TS0", ts0), ("held-out", ho), ("bpRNA-new", bn)):
        for r in src:
            rows.append({"index": len(rows), "set": tset,
                         "family": r[0] if tset == "held-out" else None,
                         "name": r[1], "seq": r[2], "ref_db": r[3],
                         "stratum": cd.length_stratum(len(r[2]), bounds)})
    g5, draws = cf.g5_test_rows(with_records=True)
    if [(r["set"], r["name"], r["seq"]) for r in rows] != [tuple(x) for x in g5]:
        raise GuardFailure("the run's rows are not the rows G5 attacked")
    return rows, draws


def shard_of(rows: list[dict], n_w: int) -> dict[int, int]:
    """{index: worker}: longest first, dealt round-robin.

    Every worker gets the same mix of lengths, so the 1800 s rho budgets of the longest
    sequences spread evenly instead of piling onto one worker; ties in length go by index, so
    the assignment depends on the rows alone.
    """
    order = sorted(rows, key=lambda r: (-len(r["seq"]), r["index"]))
    return {r["index"]: k % n_w for k, r in enumerate(order)}


def _in_certfold_env() -> None:
    """The registration's own predicate: the compiled callback, in .venv-certfold (the venv's
    directory, unresolved, because a venv's bin/python is a symlink out of it)."""
    env = cf.env_record()
    exe = pathlib.Path(env.get("python", "")).absolute()
    if (env.get("callback_backend") != "compiled" or env.get("numba") is None
            or exe.parent.parent != (R / ".venv-certfold").absolute()):
        raise SystemExit("the run belongs in .venv-certfold, with the compiled callback")


# ----------------------------------------------------------------------------------
# run: the worker
# ----------------------------------------------------------------------------------
def sequence_record(lc, B: dict, sets: dict, r: dict) -> dict:
    """certify_sequence's record for one row, with the per-sequence baselines beside it.

    p is recorded by certify_sequence for every MFE pair; pair_probabilities computes the same
    matrix by the same discipline, and the two are required to agree exactly, so the centroid
    is read off the very probabilities the ranking uses.
    """
    from scripts.certfold_baselines import (
        gamma_centroid_pairs,
        mea_structures,
        pair_probabilities,
    )

    rec = cf.certify_sequence(lc, B, sets, r["name"], r["seq"], r["ref_db"], r["stratum"],
                              cf.RHO_BUDGET_S, r["index"], seed_tag=RHO_ORDER_TAG)
    P = pair_probabilities(r["seq"])
    for q in rec["pairs"]:
        i, j = q["pair"]
        if q["p"] != float(P[i, j]):
            raise GuardFailure(f"{r['name']} ({i}, {j}): certify_sequence's p {q['p']} is not "
                               f"pair_probabilities' {float(P[i, j])}")
    rec["mea"] = {f"{g:g}": s for g, s in mea_structures(r["seq"]).items()}
    rec["centroid"] = {f"{g:g}": sorted([i, j] for i, j in s)
                       for g, s in gamma_centroid_pairs(P).items()}
    rec["set"], rec["family"] = r["set"], r["family"]
    return rec


def _done_indices(dst: pathlib.Path) -> set[int]:
    """Indices already written to a shard, for --resume. A torn last line (a kill mid-write) is
    dropped from the file; a torn line anywhere else is a damaged shard and stops the run."""
    if not dst.exists():
        return set()
    lines = dst.read_text().split("\n")
    keep, done = [], set()
    for k, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            if k == len(lines) - 1:
                break
            raise GuardFailure(f"{dst.name}: line {k + 1} is not JSON") from None
        keep.append(ln)
        done.add(rec["index"])
    dst.write_text("".join(x + "\n" for x in keep))
    return done


def run_worker() -> int:
    w = int(sys.argv[sys.argv.index("--shard") + 1])
    n_w = int(sys.argv[sys.argv.index("--of") + 1])
    resume = "--resume" in sys.argv
    if cf.callback_backend() != "compiled":
        raise GuardFailure("the run folds with the compiled callback only")
    B = cf.load_tr0_basis()
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    sets = cf.run_sets(B)
    rows, _ = test_rows()
    assign = shard_of(rows, n_w)
    mine = sorted((r for r in rows if assign[r["index"]] == w),
                  key=lambda r: (-len(r["seq"]), r["index"]))
    dst = cf.SHARDS / f"run_{w}.jsonl"
    done = _done_indices(dst) if resume else set()
    with open(dst, "a" if resume else "w") as fh:
        for r in mine:
            if r["index"] in done:
                continue
            fh.write(json.dumps(sequence_record(lc, B, sets, r)) + "\n")
            fh.flush()
    freq_dst = cf.SHARDS / f"run_freq_{w}.json"
    if not (resume and freq_dst.exists()):
        freq_dst.write_text(json.dumps(sampling_baselines(B, P, mine, w)))
    return 0


def sampling_baselines(B: dict, P: dict, mine: list[dict], w: int) -> dict:
    """Pair frequencies of every MFE pair of every sequence in the shard, under the 200 envelope
    draws (lambda = 1, tag 30) and the 200 noise draws (sigma 0.3 kcal/mol per physical
    parameter, tag 31). Every shard draws the same tables (fixed seeds); the record's digests
    let the driver check that.

    The noise is drawn over B["cols"], the 12,607 entries the d0 energy model reads; the
    template's other entries (multiloop and exterior mismatches, which dangles 0 never reads)
    enter no energy, so perturbing them could change no fold.
    """
    import RNA

    from scripts.certfold_baselines import envelope_draws, noise_draws, pair_frequencies
    from thermo.vienna_ref import assert_turner2004_params, reference_md

    seqs = [r["seq"] for r in mine]
    stars = []
    for s in seqs:
        fc = RNA.fold_compound(s, reference_md())
        assert_turner2004_params(fc)
        stars.append(fc.mfe()[0])
    scratch = cf.SHARDS / f"run_tables_{w}"
    scratch.mkdir(parents=True, exist_ok=True)
    out = {"by_index": {str(r["index"]): {} for r in mine}}
    for kind, (shifts, drec) in (("env", envelope_draws(B, N_DRAWS, lam=1.0)),
                                 ("noise", noise_draws(B["cols"], N_DRAWS))):
        freq, frec = pair_frequencies(seqs, shifts, P["theta"], scratch, cols=B["cols"])
        for r, star, f in zip(mine, stars, freq, strict=True):
            pt = cf.pair_table(star)
            out["by_index"][str(r["index"])][kind] = [
                [i, int(pt[i]), f.get((i, int(pt[i])), 0.0)]
                for i in range(1, len(star) + 1) if pt[i] > i]
        out[f"{kind}_draws"] = {k: v for k, v in drec.items() if k != "draws"}
        out[f"{kind}_frequencies"] = {
            **{k: v for k, v in frec.items() if k != "draws"},
            "par_sha256_by_draw": [d["par_sha256"] for d in frec["draws"]],
            "n_sequences_mfe_differs_by_draw":
                [d["n_sequences_mfe_differs_from_turner2004"] for d in frec["draws"]]}
    return out


# ----------------------------------------------------------------------------------
# run: the driver
# ----------------------------------------------------------------------------------
def _quantiles(xs) -> dict | None:
    xs = np.asarray(xs, dtype=float)
    return None if not len(xs) else {
        "n": int(len(xs)), "min": float(xs.min()), "p25": float(np.percentile(xs, 25)),
        "median": float(np.median(xs)), "p75": float(np.percentile(xs, 75)),
        "p95": float(np.percentile(xs, 95)), "max": float(xs.max())}


ZUKER_NOISE = re.compile(r"\[ERROR\]   Added \d+ bps")


def _pump(proc, log: pathlib.Path, append: bool, dropped: list, w: int) -> None:
    """Copy a worker's output to its log, dropping exactly one known line.

    subopt_zuker writes "[ERROR]   Added N bps" to stderr once per returned structure (the plan
    records it: no driver may decide failure from it). The run makes seven Zuker calls per test
    sequence, and G5, which logs them unfiltered, wrote about 11 MB a minute across its workers;
    a day of that would exceed what the shared disk has free. That one line, matched whole, is
    dropped and counted; every other line, a traceback included, is kept.
    """
    with log.open("a" if append else "w") as fh:
        for line in proc.stdout:
            if ZUKER_NOISE.fullmatch(line.rstrip("\n")):
                dropped[w] += 1
            else:
                fh.write(line)
                fh.flush()


def read_shards(n_w: int) -> tuple[list[dict], dict]:
    """Every sequence record and every shard's frequencies, keyed by index."""
    seq_recs, freqs = [], {}
    for w in range(n_w):
        with open(cf.SHARDS / f"run_{w}.jsonl") as fh:
            seq_recs += [json.loads(ln) for ln in fh if ln.strip()]
        freqs[w] = json.loads((cf.SHARDS / f"run_freq_{w}.json").read_text())
    return sorted(seq_recs, key=lambda r: r["index"]), freqs


def run() -> int:
    """The registered test-set run. Labels to completion, rho within budget, the baselines."""
    resume = "--resume" in sys.argv
    out = cf.read_out()
    if out.get("status") != "G5 passed":
        raise SystemExit(f"the run follows G5; status is {out.get('status')!r}")
    if not cf._pred_ok(out):
        raise SystemExit("PRED changed since the registration")
    head = cf._committed(unmodified=True)
    if head.get("predictions") != json.loads(json.dumps(cf.PRED)):
        raise SystemExit("the committed artifact does not carry this registration")
    g5 = head.get("gate", {}).get("G5", {})
    if not g5.get("pass") or g5.get("n_violations") != 0:
        raise SystemExit("G5 has not passed on the committed artifact")
    if "run" in head:
        raise SystemExit("the run exists")
    prov = git_provenance(R)
    if prov["git_dirty_excluding_runs"]:
        raise SystemExit(f"commit first: {prov.get('git_dirty_paths_outside_runs')}")
    if cf._git("ls-files", "--error-unmatch", ME).returncode != 0:
        raise SystemExit(f"{ME} is not committed")
    cf._same_code(head["registration_provenance"], "registration")
    cf._same_code(g5["provenance"], "G5")
    _in_certfold_env()
    rows, draws = test_rows()
    if json.loads(json.dumps(draws)) != g5.get("test_set_draws"):
        raise GuardFailure("the test-set draws differ from the ones G5 recorded")
    me_blob = cf._git("rev-parse", f"HEAD:{ME}").stdout.strip()
    load, t0 = os.getloadavg(), time.time()
    cf.SHARDS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    procs = [subprocess.Popen(
        ["nice", "-n", "10", sys.executable, str(R / ME), "--phase", "run-worker",
         "--shard", str(w), "--of", str(RUN_WORKERS)] + (["--resume"] if resume else []),
        cwd=R, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        errors="replace") for w in range(RUN_WORKERS)]
    dropped = [0] * RUN_WORKERS
    pumps = [threading.Thread(target=_pump, args=(p, cf.SHARDS / f"run_{w}.log", resume,
                                                  dropped, w), daemon=True)
             for w, p in enumerate(procs)]
    for t in pumps:
        t.start()
    pids = [p.pid for p in procs]
    while True:
        codes = [p.poll() for p in procs]
        if any(c not in (None, 0) for c in codes):
            for q in procs:
                if q.poll() is None:
                    q.terminate()
            codes = [q.wait() for q in procs]
            break
        if all(c == 0 for c in codes):
            break
        time.sleep(30)
    for t in pumps:
        t.join()
    if any(codes):
        # kept, not reset: a later attempt (--resume, after this record is committed) sits
        # beside it, and the artifact says how many tries the run took
        out = cf.read_out()
        out.setdefault("run_failed_attempts", []).append({
            "exit_codes": codes, "pids": pids, "resumed": resume, "provenance": prov,
            "load_at_start": [round(x, 2) for x in load],
            "wall_seconds": round(time.time() - t0, 1),
            "zuker_stderr_lines_dropped_by_shard": dropped,
            "sequences_written": {f"run_{w}.jsonl": len(_done_indices(
                cf.SHARDS / f"run_{w}.jsonl")) for w in range(RUN_WORKERS)}})
        cf.write(out)
        raise GuardFailure(f"run workers exited {codes}; shards kept at {cf.SHARDS}; the "
                           f"attempt is recorded under run_failed_attempts")

    seq_recs, freqs = read_shards(RUN_WORKERS)
    if [r["index"] for r in seq_recs] != list(range(len(rows))):
        raise GuardFailure(f"the run labelled {len(seq_recs)} of {len(rows)} test sequences")
    viol = sum(r["recursion"]["violations"] for r in seq_recs)
    if viol:
        raise GuardFailure(f"{viol} recursion-side violations in the run")
    for w in range(1, RUN_WORKERS):
        for kind in ("env", "noise"):
            for part in ("draws", "frequencies"):
                a, b = freqs[0][f"{kind}_{part}"], freqs[w][f"{kind}_{part}"]
                keep = ("par_sha256_by_draw",) if part == "frequencies" else tuple(a)
                if any(a.get(k) != b.get(k) for k in keep if k not in ("seconds",)):
                    raise GuardFailure(f"shard {w} drew different {kind} tables from shard 0")
    by_set = {}
    for tset in SETS:
        mine = [r for r in seq_recs if r["set"] == tset]
        strata = collections.defaultdict(lambda: [0, 0, 0])
        for r in mine:
            s = strata[r["stratum"]]
            s[0] += 1
            s[1] += r["rho_coverage"]["n_with_rho"]
            s[2] += r["rho_coverage"]["n_pairs"]
        by_set[tset] = {
            "n_sequences": len(mine),
            "n_mfe_pairs": sum(r["n_pairs_star"] for r in mine),
            "n_rho_timed_out": sum(r["rho_coverage"]["timed_out"] for r in mine),
            "rho_coverage_by_stratum": {
                str(k): {"n_sequences": v[0], "n_with_rho": v[1], "n_pairs": v[2],
                         "fraction": v[1] / v[2] if v[2] else None}
                for k, v in sorted(strata.items())},
            "seconds_per_sequence": _quantiles([r["seconds"] for r in mine]),
            "seconds_labels": _quantiles([r["seconds_labels"] for r in mine]),
            "seconds_rho": _quantiles([r["seconds_rho"] for r in mine])}
    files = {f"run_{w}.jsonl": cf._sha(cf.SHARDS / f"run_{w}.jsonl") for w in range(RUN_WORKERS)}
    files.update({f"run_freq_{w}.json": cf._sha(cf.SHARDS / f"run_freq_{w}.json")
                  for w in range(RUN_WORKERS)})
    rec = {
        "label": "the registered test-set run (registration 1)",
        "module": ME, "module_blob": me_blob,
        "sets": by_set, "n_sequences": len(seq_recs), "test_set_draws": draws,
        "workers": RUN_WORKERS, "pids": pids, "nice": 10, "resumed": resume,
        "zuker_stderr_lines_dropped_by_shard": dropped,
        "rho_budget_seconds_per_sequence": cf.RHO_BUDGET_S,
        "rho_pair_order_seed": [cf.SEED, RHO_ORDER_TAG, "index"],
        "label_sets": {k: list(v) for k, v in cf.run_sets(cf.load_tr0_basis()).items()},
        "recursion_side": {k: sum(r["recursion"][k] for r in seq_recs)
                           for k in ("folds_checked", "zuker_structures_checked", "violations")},
        "folds_total": sum(r["folds"] for r in seq_recs),
        "sampling_baselines": {f"{kind}_{part}": freqs[0][f"{kind}_{part}"]
                               for kind in ("env", "noise") for part in ("draws", "frequencies")},
        "sampling_seconds_by_shard": {str(w): {k: freqs[w][f"{k}_frequencies"]["seconds"]
                                               for k in ("env", "noise")}
                                      for w in range(RUN_WORKERS)},
        "load_at_start": [round(x, 2) for x in load],
        "wall_seconds": round(time.time() - t0, 1),
        "cpu_seconds_labels_and_rho": round(sum(r["seconds"] for r in seq_recs), 1),
        "files_sha256": files, "provenance": prov, "environment": cf.env_record(),
    }
    out = cf.read_out()                 # re-read: hours have passed since the start
    out["run"] = rec
    out["status"] = "run done"
    cf.write(out)
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("provenance", "files_sha256", "sampling_baselines",
                                   "environment", "test_set_draws")}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------------
# analyse: the per-pair records the frozen metrics take
# ----------------------------------------------------------------------------------
def _pairs_of(db: str) -> list[tuple[int, int]]:
    pt = cf.pair_table(db)
    return [(i, int(pt[i])) for i in range(1, len(db) + 1) if pt[i] > i]


def pair_records(seq_recs: list[dict], freq_by_index: dict) -> dict[str, list[dict]]:
    """{test set: records} in certfold_metrics's record format.

    One record per Turner 2004 MFE pair (kind "mfe") and one per pair of each MEA and
    gamma-centroid prediction (kind "mea:<gamma>", "centroid:<gamma>"), every one scored
    against the reference by slip_correct. The bootstrap unit is the sequence (its index) on
    TS0 and bpRNA-new and the family on the held-out set, which holds one sequence per family,
    so there the two coincide and the family is the unit PRED names. Every sequence also gets
    one record of kind "unit" carrying nothing but its unit and family, so that a sequence
    contributing no pair to some statistic -- an open-chain MFE, an empty prediction -- is still
    a unit the bootstrap draws: "sequences resampled" means all of them. Every statistic
    selects its records by kind, so these records enter no number.

    rho_env is the VERIFIED radius (rho_of_pair's "rho"), None where it was not computed within
    the budget or did not converge; the two causes are counted apart. tiebreak is the nominal
    gap X0(p) - E0(s*), the registered tiebreak of every ranking, and "flat" is a constant, the
    fractional-weights-alone reading of the sampling baselines. neg_rho and neg_tiebreak rank
    the same pairs in exactly the reverse order, which is how R2's lowest quartile is read off
    the frozen risk-coverage curve.
    """
    out: dict[str, list[dict]] = {s: [] for s in SETS}
    for s in seq_recs:
        ref = _pairs_of(s["ref_db"])
        fam = s["family"]
        unit = fam if s["set"] == "held-out" else s["index"]
        out[s["set"]].append({"kind": "unit", "unit": unit, "family": fam, "index": s["index"]})
        fq = {kind: {(i, j): f for i, j, f in freq_by_index[s["index"]][kind]}
              for kind in ("env", "noise")}
        mfe_pairs = [tuple(q["pair"]) for q in s["pairs"]]
        for kind in ("env", "noise"):
            if set(fq[kind]) != set(mfe_pairs):
                raise GuardFailure(f"sequence {s['index']}: the {kind} frequencies are not over "
                                   f"its MFE pairs")
        for q in s["pairs"]:
            pair = tuple(q["pair"])
            e = q["rho_env"]
            rho = None if e is None or e["rho"] is None else float(e["rho"])
            out[s["set"]].append({
                "kind": "mfe", "unit": unit, "family": fam, "pair": pair, "index": s["index"],
                "stratum": s["stratum"], "correct": slip_correct(pair, ref),
                "rho_env": rho, "neg_rho": None if rho is None else -rho,
                "rho_status": ("budget" if e is None else "unconverged" if e["rho"] is None
                               else "capped" if e["capped"] else "tied_at_zero"
                               if e["tied_at_zero"] else "finite"),
                "capped": bool(e is not None and e["capped"]),
                "tiebreak": float(q["nominal_gap_kcal"]),
                "neg_tiebreak": -float(q["nominal_gap_kcal"]),
                "gap": float(q["nominal_gap_kcal"]), "flat": 0.0, "p": float(q["p"]),
                "freq_env": float(fq["env"][pair]), "freq_noise": float(fq["noise"][pair]),
                "certified_l1": q["margins_dcal"][PRIMARY] > 0})
        preds = {f"mea:{g}": _pairs_of(db) for g, db in s["mea"].items()}
        preds.update({f"centroid:{g}": [tuple(x) for x in ps] for g, ps in s["centroid"].items()})
        for kind, ps in preds.items():
            for pair in ps:
                out[s["set"]].append({"kind": kind, "unit": unit, "family": fam, "pair": pair,
                                      "index": s["index"], "correct": slip_correct(pair, ref)})
    return out


# ----------------------------------------------------------------------------------
# analyse: the statistics, each a function of a record list (a resample) to one number
# ----------------------------------------------------------------------------------
def _p_hi(r: dict) -> bool:
    return r["p"] >= P_HI


def _everything(_r: dict) -> bool:
    return True


def _mfe(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r["kind"] == "mfe"]


def _rho_pool(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r["kind"] == "mfe" and r["rho_env"] is not None]


def _ppv(recs: list[dict], by_family: bool) -> float:
    return family_weighted_ppv(recs, _everything) if by_family else pooled_ppv(recs)


def stat_matched(recs, baseline: str, tb: str, by_family: bool) -> float:
    """rho_env's PPV minus the baseline's at the coverage of the p >= 0.99 set, both over the
    MFE pairs carrying a rho_env. With a baseline tiebreak other than the registered one, the
    baseline's PPV comes from a second frozen call with that tiebreak; both calls share the
    pool, the coverage and therefore k, which is checked."""
    mfe = _mfe(recs)
    a = matched_coverage_comparison(mfe, "rho_env", baseline, _p_hi, by_family=by_family)
    if tb == "tiebreak":
        return a["difference"]
    b = matched_coverage_comparison(mfe, baseline, "rho_env", _p_hi, tiebreak_key=tb,
                                    by_family=by_family)
    if (a["k"], a["n"]) != (b["k"], b["n"]):
        raise GuardFailure(f"the two readings of {baseline} matched different k or n")
    return a["ppv_score"] - b["ppv_score"]


def stat_aurc(recs, baseline: str, tb: str) -> float:
    """AURC of rho_env minus AURC of the baseline, both over the MFE pairs carrying a rho_env
    (lower is better, so a negative difference favours rho_env). Pooled on every set: the
    frozen risk-coverage curve has no family weighting; on the held-out set the families are
    still the resampled unit."""
    pool = _rho_pool(recs)
    return (aurc(risk_coverage(pool, "rho_env", "tiebreak")[0])
            - aurc(risk_coverage(pool, baseline, tb)[0]))


def stat_point(recs, kind: str, by_family: bool) -> float:
    """rho_env's PPV over its top k MFE pairs minus a prediction's PPV over its own k pairs, k =
    the prediction's pair count (PRED: MEA and centroid points, PPV over THEIR OWN pairs).
    Refused where k exceeds the rho pool, which is then not a matched comparison."""
    own = [r for r in recs if r["kind"] == kind]
    pool = _rho_pool(recs)
    k = len(own)
    if not 0 < k <= len(pool):
        raise GuardFailure(f"{kind}: {k} pairs against a rho pool of {len(pool)}")
    m = matched_coverage_comparison(pool, "rho_env", "p", k / len(pool), by_family=by_family)
    if m["k"] != k:
        raise GuardFailure(f"{kind}: the matched selection took {m['k']} pairs, not {k}")
    return m["ppv_score"] - _ppv(own, by_family)


def stat_disagreement(recs, which: str, by_family: bool) -> float:
    """Slip-PPV of D1 (p >= 0.99, not certified at lambda = 1) or D2 (certified, p < 0.5)."""
    d1, d2 = disagreement_sets(_mfe(recs))
    return _ppv(d1 if which == "D1" else d2, by_family)


def _r2_pool(recs) -> list[dict]:
    return [r for r in _rho_pool(recs) if r["p"] >= P_HI]


def stat_r2_quartile(recs) -> float:
    """R2: PPV of the lowest rho_env quartile minus PPV of the rest, among MFE pairs with
    p >= 0.99 carrying a rho_env. The lowest quartile is the top of the REVERSED ranking (rho
    ascending, then the nominal gap ascending) at coverage 1/4, read as ppv_at_coverage reads
    it: the smallest k_low with k_low / n >= 1/4. The rest is the top n - k_low of the forward
    ranking. A block tied on both keys across the boundary is shared by fractional weights,
    and the two readings take complementary shares of it."""
    pool = _r2_pool(recs)
    n = len(pool)
    covs = np.arange(1, n + 1) / n
    k_low = int(np.searchsorted(covs, 0.25, side="left")) + 1
    if k_low >= n:
        raise GuardFailure(f"R2: {n} pairs leave no 'rest' beside a lowest quartile of {k_low}")
    low = ppv_at_coverage(risk_coverage(pool, "neg_rho", "neg_tiebreak")[0], 0.25)
    rest = ppv_at_coverage(risk_coverage(pool, "rho_env", "tiebreak")[0], (n - k_low) / n)
    return low - rest


def stat_r2_capped(recs) -> float:
    """R2's registered fallback: PPV(uncapped) - PPV(capped) among the same pairs."""
    pool = _r2_pool(recs)
    return (pooled_ppv(pool, [not r["capped"] for r in pool])
            - pooled_ppv(pool, [r["capped"] for r in pool]))


STATS = {"matched": stat_matched, "aurc": stat_aurc, "point": stat_point,
         "disagreement": stat_disagreement, "r2_quartile": stat_r2_quartile,
         "r2_capped": stat_r2_capped}
_BOOT_RECORDS: dict[str, list[dict]] = {}       # set in the parent before the pool forks


def task_records(recs: list[dict], stat: str, args: tuple) -> list[dict]:
    """The records one statistic reads, plus every unit record.

    The unit records keep the resampling universe the whole set (every sequence or family),
    so every task draws the same units with the same seed, exactly as on the full list; the
    statistic selects by kind, so its value on any resample is unchanged. Only the records it
    never reads are left behind -- above all the MEA and centroid pairs, eight predictions'
    worth, which would otherwise be copied through every one of 10,000 resamples of every task.
    """
    if stat == "point":
        keep = {args[0]}
    elif stat == "disagreement":
        keep = {"mfe"}                      # all MFE pairs: labels are complete
    else:
        keep = {"mfe:rho"}                  # matched, aurc, R2: pairs carrying a rho_env
    out = []
    for r in recs:
        k = r["kind"]
        if (k == "unit" or k in keep
                or (k == "mfe" and r["rho_env"] is not None and ("mfe:rho" in keep
                                                                  or stat == "point"))):
            out.append(r)
    return out


def _boot_task(task: tuple) -> tuple:
    """One paired bootstrap: (task id, (point, lo, hi) or None, why None)."""
    tid, tset, stat, args = task

    def f(recs):
        return STATS[stat](recs, *args)

    try:
        recs = task_records(_BOOT_RECORDS[tset], stat, args)
        return tid, paired_bootstrap(recs, f, n_boot=N_BOOT), None
    except GuardFailure as e:
        # a resample on which the statistic is undefined (an empty disagreement set, a
        # prediction outgrowing the pool) stops the bootstrap, as paired_bootstrap requires;
        # the interval is then reported as undefined, with the reason, never patched
        return tid, None, str(e)


def run_bootstraps(records: dict[str, list[dict]], tasks: list[tuple], procs: int) -> dict:
    """Every bootstrap, spread over `procs` forked processes (the records are inherited, not
    pickled). Each is seeded by certfold_metrics alone (SeedSequence([SEED, 40])), so the
    result does not depend on the process it ran in or the order tasks finish."""
    _BOOT_RECORDS.clear()
    _BOOT_RECORDS.update(records)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(procs) as pool:
        got = pool.map(_boot_task, tasks, chunksize=1)
    return {tid: {"interval": None if res is None else [res[1], res[2]],
                  "point_bootstrap": None if res is None else res[0],
                  "interval_undefined_because": why}
            for tid, res, why in got}


# ----------------------------------------------------------------------------------
# analyse: the descriptive side, counts only
# ----------------------------------------------------------------------------------
def describe(seq_recs: list[dict], label_sets: list[str]) -> dict:
    """Certified fractions under every set, by length stratum; rho's status and coverage; the
    exposure quartiles (the mechanism). Counts over labels, which are always complete."""
    strata = sorted({r["stratum"] for r in seq_recs})
    prof = {str(s): {"n_sequences": sum(r["stratum"] == s for r in seq_recs),
                     "n_mfe_pairs": sum(r["n_pairs_star"] for r in seq_recs
                                        if r["stratum"] == s),
                     "lengths": [min(r["n_nt"] for r in seq_recs if r["stratum"] == s),
                                 max(r["n_nt"] for r in seq_recs if r["stratum"] == s)]}
            for s in strata}
    by_label = {}
    for k in label_sets:
        def frac(sel, num, den):
            a = sum(num(r) for r in sel)
            b = sum(den(r) for r in sel)
            return {"n": a, "of": b, "fraction": a / b if b else None}

        parts = {
            "certified_present": (lambda r, k=k: r["sets"][k]["n_certified_present"],
                                  lambda r: r["n_pairs_star"]),
            "certified_absent_contested": (
                lambda r, k=k: r["sets"][k]["n_certified_absent_contested"],
                lambda r: r["n_contested"]),
            "certified_absent_admissible": (
                lambda r, k=k: r["sets"][k]["n_certified_absent_admissible"],
                lambda r: r["n_admissible_outside_star"])}
        by_label[k] = {
            name: {"pooled": frac(seq_recs, *fn),
                   "by_stratum": {str(s): frac([r for r in seq_recs if r["stratum"] == s], *fn)
                                  for s in strata}}
            for name, fn in parts.items()}
        whole = fraction_by_stratum(
            seq_recs, lambda r, k=k: bool(r["sets"][k]["whole_structure_certified"]))
        by_label[k]["whole_structure_certified"] = {
            "pooled": {"n": sum(v[0] for v in whole.values()),
                       "of": sum(v[1] for v in whole.values())},
            "by_stratum": {str(s): {"n": v[0], "of": v[1], "fraction": v[2]}
                           for s, v in whole.items()}}

    def rho_status(kind):
        st = collections.Counter()
        vals, moved, cyc = [], 0, 0
        cover = collections.defaultdict(lambda: [0, 0])
        for r in seq_recs:
            for q in r["pairs"]:
                e = q[f"rho_{kind}"]
                cover[r["stratum"]][1] += 1
                if e is None:
                    st["budget"] += 1
                    continue
                cover[r["stratum"]][0] += e["rho"] is not None
                if e["rho"] is None:
                    st["unconverged"] += 1
                elif e["tied_at_zero"]:
                    st["tied_at_zero"] += 1
                elif e["capped"]:
                    st["capped"] += 1
                else:
                    st["finite"] += 1
                    vals.append(e["rho"])
                moved += bool(e["verify_moved"])
                cyc += bool(e["cycle"])
        return {"status": dict(st), "finite": _quantiles(vals), "verify_moved": moved,
                "cycles": cyc,
                "coverage_by_stratum": {str(s): {"n_with_rho": c[0], "n_pairs": c[1],
                                                 "fraction": c[0] / c[1] if c[1] else None}
                                        for s, c in sorted(cover.items())}}

    expo = np.array([r["exposure"] for r in seq_recs])
    qs = np.quantile(expo, [0.25, 0.5, 0.75])
    bins = np.digitize(expo, qs)            # the pilot's quartile rule, unchanged
    mech = {"quartile_edges": [float(x) for x in qs], "by_set": {}}
    for k in [x for x in label_sets if x.startswith("envelope:")] + ["ball:r1"]:
        rows = []
        for b in range(4):
            sel = [r for r, bb in zip(seq_recs, bins, strict=True) if bb == b]
            n = sum(r["n_pairs_star"] for r in sel)
            c = sum(r["sets"][k]["n_certified_present"] for r in sel)
            rows.append({"quartile": b, "n_sequences": len(sel), "n_mfe_pairs": n,
                         "exposure_median": float(np.median(expo[bins == b]))
                         if (bins == b).any() else None,
                         "length_median": float(np.median([r["n_nt"] for r in sel]))
                         if sel else None,
                         "uncertain_present_fraction": (n - c) / n if n else None})
        mech["by_set"][k] = rows
    return {"length_profile": prof, "labels": by_label,
            "rho_env": rho_status("env"), "rho_ball": rho_status("ball"),
            "exposure_quartiles": mech,
            "exposure": _quantiles(expo),
            "recursion_side": {k: sum(r["recursion"][k] for r in seq_recs)
                               for k in ("folds_checked", "zuker_structures_checked",
                                         "violations")}}


# ----------------------------------------------------------------------------------
# analyse: the whole analysis on records, and the gated phase
# ----------------------------------------------------------------------------------
def _defined(point, tasks: list, task: tuple | None) -> dict:
    """point(), and its bootstrap task queued; or, where a frozen metric refuses the full
    records (an empty coverage set, an empty pool), the refusal recorded as the result and no
    interval attempted. A refusal is reported, never replaced by a number."""
    try:
        got = point()
    except GuardFailure as e:
        return {"undefined_because": str(e)}
    if task is not None:
        tasks.append(task)
    return got


def _matched_point(mfe: list[dict], key: str, tb: str, fam: bool) -> dict:
    """The full matched-coverage record behind stat_matched's number."""
    m = matched_coverage_comparison(mfe, "rho_env", key, _p_hi, by_family=fam)
    if tb == "tiebreak":
        return m
    b = matched_coverage_comparison(mfe, key, "rho_env", _p_hi, tiebreak_key=tb, by_family=fam)
    return {**m, "ppv_baseline": b["ppv_score"], "difference": m["ppv_score"] - b["ppv_score"],
            "baseline_tiebreak": tb}


def _aurc_point(pool: list[dict], key: str, tb: str) -> dict:
    """The two areas behind stat_aurc's number."""
    a = aurc(risk_coverage(pool, "rho_env", "tiebreak")[0])
    b = aurc(risk_coverage(pool, key, tb)[0])
    return {"aurc_rho_env": a, "aurc_baseline": b, "difference": a - b,
            "baseline_tiebreak": tb, "n": len(pool)}


def _prediction_point(own: list[dict], pool: list[dict], fam: bool) -> dict:
    """An MEA or centroid point: its PPV over its own k pairs, and rho_env's over its top k."""
    k = len(own)
    pt = {"k": k, "n_rho_pool": len(pool), "ppv_own_pairs": _ppv(own, fam)}
    if k <= len(pool):
        m = matched_coverage_comparison(pool, "rho_env", "p", k / len(pool), by_family=fam)
        return {**pt, "matched": True, "ppv_rho_env_top_k": m["ppv_score"],
                "difference": m["ppv_score"] - pt["ppv_own_pairs"]}
    # PRED: reported beside rho_env's whole-pool PPV with the coverage gap stated, and not a
    # matched comparison, so it gets no interval
    return {**pt, "matched": False, "ppv_rho_env_whole_pool": _ppv(pool, fam),
            "coverage_gap_pairs": k - len(pool)}


def _curve_points(pool: list[dict], key: str, tb: str, grid: list[float]) -> dict:
    curve = risk_coverage(pool, key, tb)[0]
    return {"ppv": [ppv_at_coverage(curve, c) for c in grid]}


def analyse_set(tset: str, recs: list[dict], mine: list[dict], label_sets: list[str],
                tasks: list) -> dict:
    """One test set's points, queueing each defined point's bootstrap in `tasks`."""
    fam = tset == "held-out"
    mfe, pool = _mfe(recs), _rho_pool(recs)
    d1, d2 = disagreement_sets(mfe)
    st: dict = {
        "weighting": "families weighted equally; families resampled" if fam else
                     "pairs pooled; sequences resampled",
        "n_sequences": len(mine), "n_mfe_pairs": len(mfe), "n_rho_pool": len(pool),
        "rho_env_unscored": dict(collections.Counter(
            r["rho_status"] for r in mfe if r["rho_env"] is None)),
        "describe": describe(mine, label_sets),
        "matched": {}, "aurc": {}, "points": {}, "disagreement": {}, "curves": {}}
    for name, key, tb in BASELINES:
        st["matched"][name] = _defined(
            lambda key=key, tb=tb: _matched_point(mfe, key, tb, fam), tasks,
            (f"{tset}|matched|{name}", tset, "matched", (key, tb, fam)))
        st["aurc"][name] = _defined(
            lambda key=key, tb=tb: _aurc_point(pool, key, tb), tasks,
            (f"{tset}|aurc|{name}", tset, "aurc", (key, tb)))
    for kind in sorted({r["kind"] for r in recs} - {"mfe", "unit"}):
        own = [r for r in recs if r["kind"] == kind]
        pt = _defined(lambda own=own: _prediction_point(own, pool, fam), [], None)
        if pt.get("matched"):
            tasks.append((f"{tset}|point|{kind}", tset, "point", (kind, fam)))
        st["points"][kind] = pt
    for which, d in (("D1", d1), ("D2", d2)):
        st["disagreement"][which] = {
            "n": len(d), "of_mfe_pairs": len(mfe),
            "by_stratum": dict(collections.Counter(str(r["stratum"]) for r in d)),
            **_defined(lambda d=d: {"ppv": _ppv(d, fam)}, tasks,
                       (f"{tset}|disagreement|{which}", tset, "disagreement", (which, fam)))}
    grid = [(c + 1) / CURVE_GRID for c in range(CURVE_GRID)]
    for name, key, tb in (("rho_env", "rho_env", "tiebreak"),) + BASELINES:
        st["curves"][name] = _defined(lambda key=key, tb=tb: _curve_points(pool, key, tb, grid),
                                      [], None)
    st["curves"]["coverage"] = grid
    return st


def analyse_records(seq_recs: list[dict], freq_by_index: dict, label_sets: list[str],
                    procs: int = BOOT_PROCS) -> dict:
    """Every registered number from the run's records. Points first, then every interval."""
    recs = pair_records(seq_recs, freq_by_index)
    res: dict = {"sets": {}}
    tasks: list = []
    for tset in SETS:
        if recs[tset]:
            res["sets"][tset] = analyse_set(tset, recs[tset],
                                            [r for r in seq_recs if r["set"] == tset],
                                            label_sets, tasks)
    # R2, on TS0 alone
    if recs["TS0"]:
        pool = _r2_pool(recs["TS0"])
        capped = sum(r["capped"] for r in pool)
        frac = capped / len(pool) if pool else None
        fallback = frac is not None and frac > 0.75
        stat = "r2_capped" if fallback else "r2_quartile"
        res["R2"] = {"n_pairs": len(pool), "n_capped": capped, "capped_fraction": frac,
                     "comparison": "PPV(uncapped) - PPV(capped)" if fallback else
                                   "PPV(lowest rho_env quartile) - PPV(rest)",
                     **_defined(lambda: {"point": STATS[stat](recs["TS0"])}, tasks,
                                ("R2", "TS0", stat, ()))}
    t0 = time.time()
    boots = run_bootstraps(recs, tasks, procs)
    if "R2" in res:
        res["R2"]["pass"] = False           # until an interval says otherwise
    for tid, b in boots.items():
        if tid == "R2":
            res["R2"].update(b)
            res["R2"]["pass"] = b["interval"] is not None and b["interval"][1] < 0
            tgt = res["R2"]
        else:
            tset, part, name = tid.split("|")
            sec = res["sets"][tset]
            tgt = sec["disagreement" if part == "disagreement" else
                      "points" if part == "point" else part][name]
            tgt.update(b)
        if b["point_bootstrap"] is not None:
            want = tgt.get("difference", tgt.get("ppv", tgt.get("point")))
            if want != b["point_bootstrap"]:
                raise GuardFailure(f"{tid}: the bootstrap's point {b['point_bootstrap']} is not "
                                   f"the reported {want}")
    res["bootstrap"] = {"n_boot": N_BOOT, "seed": [cf.SEED, 40], "n_tasks": len(tasks),
                        "processes": procs, "seconds": round(time.time() - t0, 1)}
    return res


def analyse() -> int:
    """R1 to R3 and the descriptive results, from the committed run, on the run's code."""
    out = cf.read_out()
    if out.get("status") != "run done":
        raise SystemExit(f"analyse follows the run; status is {out.get('status')!r}")
    if not cf._pred_ok(out):
        raise SystemExit("PRED changed since the registration")
    head = cf._committed(unmodified=True)
    run_rec = head.get("run")
    if not run_rec:
        raise SystemExit("commit the run first")
    prov = git_provenance(R)
    if prov["git_dirty_excluding_runs"]:
        raise SystemExit(f"commit first: {prov.get('git_dirty_paths_outside_runs')}")
    cf._same_code(run_rec["provenance"], "run")
    d = cf._git("diff", "--name-only", run_rec["provenance"]["git_sha"], "HEAD", "--", ME)
    if d.stdout.strip():
        raise SystemExit(f"{ME} changed since the run")
    for f, h in run_rec["files_sha256"].items():
        if cf._sha(cf.SHARDS / f) != h:
            raise GuardFailure(f"{f} changed since the run")
    t0 = time.time()
    seq_recs, freqs = read_shards(run_rec["workers"])
    freq_by_index = {int(k): v for w in freqs.values() for k, v in w["by_index"].items()}
    res = analyse_records(seq_recs, freq_by_index, list(run_rec["label_sets"]))
    gate = head["gate"]
    r1 = {"G2_false_certificates_ball_by_radius": gate["G2"]["false_certificates"],
          "G2_false_certificates_envelope": gate["G2"]["envelope"]["n_false_certificates"],
          "G4_false_certificates": gate["G4"]["n_false_certificates"],
          "G5_violations": gate["G5"]["n_violations"]}
    counts = (list(r1["G2_false_certificates_ball_by_radius"].values())
              + [r1[k] for k in ("G2_false_certificates_envelope", "G4_false_certificates",
                                 "G5_violations")])
    r1["pass"] = (all(gate[g]["pass"] for g in ("G2", "G4", "G5"))
                  and all(c == 0 for c in counts))
    res["R1"] = r1
    res["relaxation_loss"] = {
        "from": "gate.G4: TR0, n <= 150 (decision 7)",
        "note": gate["G4"]["relaxation_loss_note"],
        "by_set": {k: {kk: v[kk] for kk in ("relaxation_loss_present", "relaxation_loss_absent",
                                            "n_robust_present", "n_certified_present",
                                            "n_robust_absent", "n_certified_absent", "n_run",
                                            "n_skipped")}
                   for k, v in gate["G4"]["by_radius"].items()}}
    res["provenance"] = prov
    res["seconds"] = round(time.time() - t0, 1)
    out = cf.read_out()
    out["analysis"] = res
    out["status"] = "analysed"
    cf.write(out)
    print(json.dumps({"R1": res["R1"], "R2": res["R2"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"run": run, "run-worker": run_worker, "analyse": analyse}[ph]())
