#!/usr/bin/env python3
"""V3 directions 1 and 2 for the pairs where direction 0 missed (amendment 5, 2026-10-11).

The registered rule (PREREG analysis.V3_amendment_2026_10_11): a (sequence x, model k) pair
misses only if three independent directions orthogonal to G(x) all fail to move the
constrained score of the predicted structure, or of the reference when realisable, by more
than REL_TOL * max(1, |score|), REL_TOL = 1e-3 unchanged. Direction 0 is the worker's
(stream [20261006, x, crc32(name)], after V2's random_in_G); directions r = 1, 2 come from
the fixed streams [20261006, x, crc32(name), r]. By the rule's symmetry a pair whose
direction 0 hit cannot miss, so directions 1 and 2 are drawn only where direction 0 missed:
the same verdict as drawing all three everywhere.

For one stage: the models with recorded V3 misses are read from the stage's shards; for
each, direction 0 is recomputed on every sequence with the worker's code and stream, the
misses are checked against the recorded count, and directions 1 and 2 are drawn for those
pairs. Also recorded: proof that the evaluator's worker() is unchanged since the shards
(the git version of scripts/mxfold2_gauge_eval.py whose sha256 equals the shards'
evaluator_sha256, its worker() source hash, and today's).

Writes runs/<date>_mxfold2_v3_redraw_<stage>/metrics.json, which the stage's reduce() reads.

Run (CPU only):
  CUDA_VISIBLE_DEVICES= .venv-mxfold2/bin/python scripts/mxfold2_v3_redraw.py --stage 2a
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import zlib

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402

EVAL = "scripts/mxfold2_gauge_eval.py"
EXTRA = (1, 2)


def out_dir(stage):
    return R / "runs" / f"{time.strftime('%Y-%m-%d')}_mxfold2_v3_redraw_{stage}"


def write(path, out):
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path / "metrics.json")


def worker_source_sha(text):
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
    return hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()


def provenance_of_shards(shard_sha):
    """The git version of the evaluator the shards were computed with, and worker() then/now."""
    commits = subprocess.run(["git", "log", "--format=%H", "--", EVAL], cwd=R,
                             capture_output=True, text=True, check=True).stdout.split()
    then = None
    for c in commits:
        text = subprocess.run(["git", "show", f"{c}:{EVAL}"], cwd=R, capture_output=True,
                              text=True, check=True).stdout
        if hashlib.sha256(text.encode()).hexdigest() == shard_sha:
            then = (c, text)
            break
    now = (R / EVAL).read_text()
    rec = {"shards_evaluator_sha256": shard_sha,
           "evaluator_sha256_now": hashlib.sha256(now.encode()).hexdigest(),
           "worker_source_sha256_now": worker_source_sha(now)}
    if then is None:
        return rec | {"shard_version_found": False, "worker_unchanged": False}
    return rec | {"shard_version_found": True, "shard_version_commit": then[0],
                  "worker_source_sha256_at_shards": worker_source_sha(then[1]),
                  "worker_unchanged": worker_source_sha(then[1]) == worker_source_sha(now)}


def direction(rng, g, th):
    z = {kk: rng.standard_normal(len(xx)) for kk, xx in th.items()}
    z["Sin"] *= g.e_in
    z["Sout"] *= g.e_out
    pz = g.project(z)
    return {kk: z[kk] - pz[kk] for kk in z}


def d3_of(constrained, net, seq, param, pp, g, targets, base, z):
    """Largest relative move of the targets' constrained scores along direction z."""
    pz_ = [{"turner": param[0]["turner"], "positional": g.apply(pp, z)}]
    new = [constrained(net, seq, pz_, t)[0] for t in targets]
    return max(abs(a - b) / max(1.0, abs(a)) for a, b in zip(base, new, strict=True))


def shard(stage, names, w, n_workers, dst):
    import torch

    import scripts.mxfold2_gauge_eval as ev
    from scripts.mxfold2_decompose import build_model, constrained, nested, predict, read_bpseq
    from scripts.mxfold2_gauge import Geometry
    from scripts.mxfold2_seed_study import recorded_seeds

    torch.set_num_threads(2)
    models = [m for m in ev.model_list(stage, recorded_seeds()) if m["name"] in names]
    assert sorted(m["name"] for m in models) == sorted(names)
    nets = []
    for mm in models:
        net = build_model(ev.LIVE / "arch.conf")
        net.load_state_dict(torch.load(mm["path"], map_location="cpu", weights_only=True)
                            ["model_state_dict"])
        net.eval()
        nets.append(net)
    entries = json.loads((ev.LIVE / "metrics.json").read_text())["evaluation_set"]["entries"]
    rows = []
    for xi, e in enumerate(entries):
        if xi % n_workers != w:
            continue
        seq, partner = read_bpseq(ev.DATA / e["file"])
        g = Geometry(seq)
        realisable = nested(partner)
        for mm, net in zip(models, nets, strict=True):
            crc = zlib.crc32(mm["name"].encode())
            with torch.no_grad():
                param = net.make_param([seq])
            pp = {kk: v.detach().cpu() for kk, v in param[0]["positional"].items()}
            th = g.theta(pp)
            _, _, bp_pred = predict(net, seq, param)
            targets = [bp_pred] + ([partner] if realisable else [])
            base = [constrained(net, seq, param, t)[0] for t in targets]
            args = (constrained, net, seq, param, pp, g, targets, base)

            rng0 = np.random.default_rng(np.random.SeedSequence([ev.BOOT_SEED, xi, crc]))
            g.random_in_G(rng0)  # V2's draw, as in the worker
            d0 = d3_of(*args, direction(rng0, g, th))
            if d0 > ev.REL_TOL:
                continue
            d = {"0": d0}
            for r in EXTRA:
                rng = np.random.default_rng(np.random.SeedSequence([ev.BOOT_SEED, xi, crc, r]))
                d[str(r)] = d3_of(*args, direction(rng, g, th))
            rows.append({"x": xi, "file": e["file"], "model": mm["name"], "d3": d,
                         "hit": {k: v > ev.REL_TOL for k, v in d.items()},
                         "misses_under_amendment": all(v <= ev.REL_TOL for v in d.values())})
    pathlib.Path(dst).write_text(json.dumps(rows))
    return 0


def main():
    stage = sys.argv[sys.argv.index("--stage") + 1]
    if "--shard" in sys.argv:
        i = sys.argv.index("--shard")
        names = json.loads(sys.argv[i + 3])
        return shard(stage, names, int(sys.argv[i + 1]), int(sys.argv[i + 2]), sys.argv[i + 4])
    import scripts.mxfold2_gauge_eval as ev
    names_all, metas, _ = ev.load_shards(stage)
    recorded = {}
    for i, n in enumerate(names_all):
        missed = sum(m_["validity"][i]["V3_n"] - m_["validity"][i]["V3_detected"] for m_ in metas)
        if missed:
            recorded[n] = missed
    od = out_dir(stage)
    shard_shas = {m_["evaluator_sha256"] for m_ in metas}
    assert len(shard_shas) == 1
    out = {"run_id": od.name, "stage": stage, "status": "running",
           "amendment": "PREREG analysis.V3_amendment_2026_10_11 (amendment 5)",
           "rel_tol": ev.REL_TOL, "extra_directions": list(EXTRA),
           "recorded_direction0_misses": recorded,
           "provenance_of_shards": provenance_of_shards(shard_shas.pop()), **git_provenance(R)}
    write(od, out)
    rows = []
    if recorded:
        n = 4
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        dsts = [od / f"shard{w}.json" for w in range(n)]
        procs = [subprocess.Popen([sys.executable, __file__, "--stage", stage, "--shard", str(w),
                                   str(n), json.dumps(sorted(recorded)), str(dsts[w])], env=env)
                 for w in range(n)]
        if any(p.wait() for p in procs):
            raise SystemExit("a shard failed")
        rows = [r for d in dsts for r in json.loads(d.read_text())]
    per_model = {}
    for name, missed in recorded.items():
        mine = [r for r in rows if r["model"] == name]
        per_model[name] = {"recorded_direction0_misses": missed, "flagged": len(mine),
                           "reproduces_recorded": len(mine) == missed,
                           "amended_misses": sum(r["misses_under_amendment"] for r in mine),
                           "pairs": mine}
    out["per_model"] = per_model
    out["all_flagged_reproduced"] = all(v["reproduces_recorded"] for v in per_model.values())
    out["V3_amended_pass"] = {n: v["reproduces_recorded"] and v["amended_misses"] == 0
                              for n, v in per_model.items()}
    out["status"] = "complete"
    write(od, out)
    print(json.dumps({k: out[k] for k in ("recorded_direction0_misses", "all_flagged_reproduced",
                                          "V3_amended_pass")} | {
        "worker_unchanged": out["provenance_of_shards"].get("worker_unchanged"),
        "pairs": [{k: r[k] for k in ("file", "model", "d3")} for r in rows]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
