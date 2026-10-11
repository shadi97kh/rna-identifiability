#!/usr/bin/env python3
"""Two defects in MXfold2's long-helix branch, reproduced on the RELEASED model.

MXfold2 v0.1.2 (vendor/mxfold2 at 51b2136), released TrainSetAB weights (MixC,
--max-helix-length 30), CPU. Test sequences are perfect hairpins: a stem of m G/C pairs
drawn from default_rng(SEED + m), a GAAA loop, and the stem's complement.

DEFECT 1 (output side; zuker.cpp:386-397 and 562-576). For m >= 31 the traceback never
records the 30th pair from the outer end, so the REPORTED structure carries a spurious 1x1
internal loop, while the reported score is that of the full helix. Measured per m: the
predicted structure, its score, the pairs of the full helix it lacks, and the constrained
scores of the printed structure and of the full helix; for m = 35 also the command-line
`mxfold2 predict` output.

DEFECT 2 (training side; the long-helix branch's loss-augmentation sum lp, zuker.cpp:139-147
and 565-572, accumulated in the same k < m loop). Measured per m: the DP constrained to the
full helix, once plain and once with an all-unpaired reference and neg_paired = 1 (every
other loss term 0). Every pair is absent from that reference, so the augmented minus the
plain score should be m, one unit per pair; if the branch omits the 30th pair's term it is
m - 1 for m >= 31. m = 30 is the control (no long-helix branch).

Run (CPU only): CUDA_VISIBLE_DEVICES= .venv-mxfold2/bin/python scripts/mxfold2_release_repro.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402

RUN_ID = "2026-10-10_mxfold2_release_repro"
OUT = R / "runs" / RUN_ID
MODELS = R / "vendor" / "mxfold2" / "mxfold2" / "models"
CONF, PARAM = MODELS / "TrainSetAB.conf", MODELS / "TrainSetAB.pth"
ARCH = R / "runs" / "2026-10-07_mxfold2_seed_study" / "arch.conf"  # TrainSetAB.conf minus --param
SEED = 20261010
STEMS = (30, 31, 35, 41)
LOOP = "GAAA"
CLI_STEM = 35


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def hairpin(m):
    rng = np.random.default_rng(SEED + m)
    stem = "".join(rng.choice(list("GC"), m))
    return stem + LOOP + "".join({"G": "C", "C": "G"}[c] for c in reversed(stem))


def partner(db):
    p, st = [0] * (len(db) + 1), []
    for k, c in enumerate(db, 1):
        if c == "(":
            st.append(k)
        elif c == ")":
            i = st.pop()
            p[i], p[k] = k, i
    return p


def main():
    import torch
    from mxfold2 import interface

    from scripts.mxfold2_decompose import build_model

    torch.set_num_threads(2)
    out = {"run_id": RUN_ID, "status": "running", "mxfold2_commit": "51b2136",
           "param_sha256": hashlib.sha256(PARAM.read_bytes()).hexdigest(),
           "conf": CONF.read_text().split(), **git_provenance(R)}
    write(out)
    model = build_model(ARCH)
    sd = torch.load(PARAM, map_location="cpu", weights_only=True)
    model.load_state_dict(sd.get("model_state_dict", sd))
    model.eval()

    def call(seq, **kw):
        with torch.no_grad():
            a = model.make_param([seq])[0]
        p = {k: model.clear_count(dict(v)) for k, v in a.items()}
        v, db, pair = interface.predict_mxfold(seq, p, max_helix_length=30, **kw)
        return float(v), db, list(pair)

    rows = {}
    for m in STEMS:
        seq = hairpin(m)
        L = len(seq)
        full = "(" * m + "." * len(LOOP) + ")" * m
        fp = partner(full)
        v, db, pair = call(seq, max_internal_length=30)
        missing = [[i, fp[i]] for i in range(1, L + 1) if fp[i] > i and pair[i] != fp[i]]
        e_printed, _, _ = call(seq, max_internal_length=L, constraint=partner(db))
        e_full, _, _ = call(seq, max_internal_length=L, constraint=fp)
        e_aug, _, _ = call(seq, max_internal_length=L, constraint=fp, reference=[0] * (L + 1),
                           loss_pos_paired=0.0, loss_neg_paired=1.0, loss_pos_unpaired=0.0,
                           loss_neg_unpaired=0.0)
        rows[str(m)] = {
            "sequence": seq, "length": L, "stem_pairs": m,
            "predicted_structure": db, "predicted_score": v,
            "predicted_equals_full_helix": db == full,
            "full_helix_pairs_missing_from_prediction": missing,
            "constrained_score_of_printed_structure": e_printed,
            "constrained_score_of_full_helix": e_full,
            "printed_score_minus_printed_structure_score": v - e_printed,
            "loss_augmentation": {"augmented_minus_plain": e_aug - e_full, "expected_if_all": m,
                                  "pairs_unpenalised": m - round(e_aug - e_full)}}
        write(out | {"hairpins": rows})
    out["hairpins"] = rows

    # the command-line tool on the m = 35 hairpin
    fa = OUT / f"hairpin{CLI_STEM}.fa"
    fa.write_text(f">hairpin{CLI_STEM}\n{hairpin(CLI_STEM)}\n")
    cli = subprocess.run([str(R / ".venv-mxfold2" / "bin" / "mxfold2"), "predict",
                          f"@./{CONF.name}", str(fa)], cwd=MODELS, capture_output=True,
                         text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=600)
    lines = [ln for ln in cli.stdout.splitlines() if ln.strip()]
    out["cli"] = {"exit_code": cli.returncode, "output": lines,
                  "structure_matches_api": bool(lines) and lines[-1].split()[0] ==
                  rows[str(CLI_STEM)]["predicted_structure"]}

    out["summary"] = {
        "defect1_long_helix_reported_without_its_30th_pair": {
            m: r["full_helix_pairs_missing_from_prediction"] for m, r in rows.items()},
        "defect1_printed_score_is_the_full_helix_score": {
            m: abs(r["predicted_score"] - r["constrained_score_of_full_helix"]) < 1e-3
            for m, r in rows.items()},
        "defect2_pairs_left_unpenalised_by_loss_augmentation": {
            m: r["loss_augmentation"]["pairs_unpenalised"] for m, r in rows.items()},
    }
    out["status"] = "complete"
    write(out)
    print(json.dumps(out["summary"], indent=1))
    print("\n".join(out["cli"]["output"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
