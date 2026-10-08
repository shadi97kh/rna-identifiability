#!/usr/bin/env python3
"""Did overriding MXfold2's torch <2 and numpy <2 pins change its numerics?

THE QUESTION

The seed study runs MXfold2 (51b2136) under torch 2.13.0+cu130 and numpy 2.4.6,
against its own pins (torch ^1.4, numpy ^1.18). This check runs the SHIPPED weights
(models/TrainSetAB.conf / TrainSetAB.pth) over all 1,305 TS0 sequences, the seed
study's fixed evaluation set.

WHY NOT AGAINST THE PAPER'S TS0 NUMBER

The paper reports TS0 accuracy (Table 2: PPV 0.520, SEN 0.682, F 0.575) for an MXfold2
TRAINED ON TR0. The shipped model was trained on TrainSetA+B (its config is named
TrainSetAB; the README says the default parameters come from TrainSetA and TrainSetB).
The paper gives no TS0 figure for it. A difference between our TS0 F and 0.575 would
mix training data with numerics and could not isolate the pin override. Both numbers
are recorded, labelled as not comparable.

THE TEST THAT DOES ISOLATE IT

The same shipped weights, the same MXfold2 source, run in a second environment that
honours the pins: .venv-mxfold2-pins (Python 3.10.20, torch 1.13.1+cpu, numpy 1.26.4).
Predictions are compared sequence by sequence with ours:
  A_cpu  .venv-mxfold2, torch 2.13 / numpy 2.4.6, CPU
  A_gpu  .venv-mxfold2, same, GPU
  B_cpu  .venv-mxfold2-pins, torch 1.13.1 / numpy 1.26.4, CPU

PRE-REGISTERED (written into metrics.json before any prediction)
N1. A_cpu and B_cpu predict the identical structure for every TS0 sequence, and their
    scores agree within 1e-3 * max(1, |score|). If so, the override did not change
    the numerics that matter. Any disagreement is listed sequence by sequence.
N2. A_gpu agrees with A_cpu to the same tolerance (GPU convolution algorithms may
    reorder sums; structures may differ only where the score gap is within tolerance).

Accuracy is computed with the paper's definitions: PPV = TP/(TP+FP), SEN = TP/(TP+FN),
F = 2*PPV*SEN/(PPV+SEN), against the canonical-pair TS0 reference. Per-sequence mean
(sequences with at least one reference pair; F = 0 when TP = 0) and pooled over all
pairs are both reported.

Writes runs/2026-10-07_mxfold2_ts0_inference/metrics.json.
Run: .venv/bin/python scripts/mxfold2_ts0_inference.py
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402

RUN_ID = "2026-10-07_mxfold2_ts0_inference"
OUT = R / "runs" / RUN_ID
DATA = R / ".cache" / "mxfold2-data"
MODELS = R / "vendor" / "mxfold2" / "mxfold2" / "models"
ENVS = {"A": R / ".venv-mxfold2" / "bin" / "python",
        "B": R / ".venv-mxfold2-pins" / "bin" / "python"}
RUNNER = R / "scripts" / "mxfold2_predict_fullprec.py"  # full-precision scores
REL_TOL = 1e-3
PUBLISHED_TR0_MODEL_TS0 = {"PPV": 0.520, "SEN": 0.682, "F": 0.575,
                           "source": "Sato et al. 2021 Nat Commun 12:941, Table 2, "
                                     "sequence-wise CV on TS0, MXfold2 trained on TR0"}


def bpseq(path):
    rows = [r.split() for r in path.read_text().split("\n") if r.strip() and not r.startswith("#")]
    seq = "".join(r[1] for r in rows).upper().replace("T", "U")
    pairs = {(int(r[0]), int(r[2])) for r in rows if int(r[2]) > int(r[0])}
    return seq, pairs


def db_pairs(db):
    st, out = [], set()
    for k, c in enumerate(db, start=1):
        if c == "(":
            st.append(k)
        elif c == ")":
            out.add((st.pop(), k))
    return out


def predict(env, gpu, fasta, out_txt):
    cmd = [str(ENVS[env]), str(RUNNER), "predict", "@./TrainSetAB.conf", "--gpu", str(gpu),
           str(fasta)]
    e = dict(os.environ)
    if gpu >= 0:
        e["CUDA_VISIBLE_DEVICES"] = e.get("MXF_GPU", "1")
    t = time.time()
    with out_txt.open("w") as fh:
        subprocess.run(cmd, cwd=MODELS, env=e, stdout=fh, stderr=subprocess.DEVNULL, check=True)
    res = {}
    for ln in out_txt.read_text().split("\n"):
        if ln.startswith("{"):
            d = json.loads(ln)
            res[d["name"]] = (d["structure"], d["score"])
    return res, time.time() - t


def accuracy(pred, refs):
    per, tp_all, fp_all, fn_all = [], 0, 0, 0
    for name, (_, ref) in refs.items():
        p = db_pairs(pred[name][0])
        tp, fp, fn = len(p & ref), len(p - ref), len(ref - p)
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
        if not ref:
            continue
        ppv = tp / (tp + fp) if tp + fp else 0.0
        sen = tp / (tp + fn)
        per.append((ppv, sen, 2 * ppv * sen / (ppv + sen) if tp else 0.0))
    n = len(per)
    pp = tp_all / (tp_all + fp_all)
    ss = tp_all / (tp_all + fn_all)
    return {"n_scored": n, "n_without_reference_pairs": len(refs) - n,
            "mean_per_sequence": {"PPV": sum(x[0] for x in per) / n,
                                  "SEN": sum(x[1] for x in per) / n,
                                  "F": sum(x[2] for x in per) / n},
            "pooled": {"PPV": pp, "SEN": ss, "F": 2 * pp * ss / (pp + ss)}}


def compare(a, b):
    diff_struct, bad_score, worst = [], [], 0.0
    for name in a:
        (sa, xa), (sb, xb) = a[name], b[name]
        d = abs(xa - xb)
        worst = max(worst, d / max(1.0, abs(xa)))
        if sa != sb:
            diff_struct.append({"name": name, "score_a": xa, "score_b": xb})
        if d > REL_TOL * max(1.0, abs(xa)):
            bad_score.append({"name": name, "score_a": xa, "score_b": xb})
    return {"n": len(a), "n_identical_structures": len(a) - len(diff_struct),
            "differing_structures": diff_struct[:50], "n_scores_outside_tol": len(bad_score),
            "scores_outside_tol": bad_score[:50], "max_relative_score_diff": worst,
            "passes": not diff_struct and not bad_score}


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))


def main():
    out = {"run_id": RUN_ID, "status": "PRE-REGISTERED, no prediction yet",
           "predictions": {"N1": "A_cpu and B_cpu: identical structures for every sequence, "
                                 f"scores within {REL_TOL} * max(1,|score|)",
                           "N2": "A_gpu and A_cpu agree to the same tolerance"},
           "published_ts0_for_a_different_model": PUBLISHED_TR0_MODEL_TS0,
           "model": "shipped TrainSetAB (models/TrainSetAB.conf, TrainSetAB.pth), MXfold2 51b2136",
           **git_provenance(R)}
    write(out)
    refs = {}
    fasta = OUT / "ts0.fa"
    with fasta.open("w") as fh:
        for ln in (DATA / "data" / "TS0-canonicals.lst").read_text().split("\n"):
            if ln.strip():
                f = ln.split()[0]
                name = pathlib.Path(f).stem
                seq, pairs = bpseq(DATA / f)
                refs[name] = (seq, pairs)
                fh.write(f">{name}\n{seq}\n")
    runs, secs = {}, {}
    for tag, env, gpu in (("A_gpu", "A", 0), ("A_cpu", "A", -1), ("B_cpu", "B", -1)):
        runs[tag], secs[tag] = predict(env, gpu, fasta, OUT / f"pred_{tag}.txt")
        assert set(runs[tag]) == set(refs), tag
        out[f"accuracy_{tag}"] = accuracy(runs[tag], refs)
        write(out)
    out["seconds"] = secs
    out["N1_A_cpu_vs_B_cpu"] = compare(runs["A_cpu"], runs["B_cpu"])
    out["N2_A_gpu_vs_A_cpu"] = compare(runs["A_gpu"], runs["A_cpu"])
    out["verdicts"] = {"N1": "confirmed" if out["N1_A_cpu_vs_B_cpu"]["passes"] else "refuted",
                       "N2": "confirmed" if out["N2_A_gpu_vs_A_cpu"]["passes"] else "refuted"}
    out["status"] = "complete"
    write(out)
    print(json.dumps({"verdicts": out["verdicts"],
                      "F_mean": {t: out[f"accuracy_{t}"]["mean_per_sequence"]["F"] for t in runs},
                      "published_TR0_model": PUBLISHED_TR0_MODEL_TS0["F"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
