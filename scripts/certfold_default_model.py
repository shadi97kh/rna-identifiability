#!/usr/bin/env python3
"""CERTIFIED FOLDING: THE DEFAULT MODEL, AND A BOUND ON DECOUPLING. Two POST HOC items, as
specified before any code in docs/certfold_default_model.md (the authors' decision of
2026-10-19, the freeze lifted one last time).

  (a) Default-model transfer. TS0 folded once under ViennaRNA's default model (the pinned
      reference model with dangles = 2 and special_hp = 1, nothing perturbed); R4's registered
      statistic (registration 2) with "kept by the default-model MFE" in place of retention over
      the three whole tables, seed tag 91; absolute retention by label.
  (b) Decoupled-coverage bound. The share of TS0 MFE pairs whose nominal gap exceeds W(s*), the
      summed lambda = 1 envelope charge on s*'s weighted loops: an upper bound on what a
      certificate charging each structure in full could certify. No folding.

Nothing here defines a label, a weight, a statistic or a bootstrap: labels are registration 1's,
weights certfold.Weights, the statistic certfold_reg2._r4_spec, the bootstrap
certfold_exploratory.stratified_bootstrap, the records certfold_run.pair_records.

PHASES, each refusing unless the specification is committed unmodified and its predecessor's
record is committed:
  fold      (a) the default-model folds and checks 1 to 3
  analyse   (a) check 4, the statistic and the tables; (b)
Run in .venv-certfold. Records: runs/2026-10-19_certfold_default_model/metrics.json.
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import sys
import time

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold_reg2 as r2  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

cf, cr = r2.cf, r2.cr
ME = "scripts/certfold_default_model.py"
SPEC = "docs/certfold_default_model.md"
RUN_ID = "2026-10-19_certfold_default_model"
OUT = R / "runs" / RUN_ID
SHARDS = OUT / "shards"
TAG_DEFAULT = 91                 # the spec's seed tag: item (a)'s stratified bootstrap
N_BOOT = r2.N_BOOT               # 10,000, registration 2's

FOLDED = "default-model folds done"
ANALYSED = "analysed"


# ----------------------------------------------------------------------------------
# records and gates (the house pattern; see certfold_unread.py)
# ----------------------------------------------------------------------------------
def read_out() -> dict:
    p = OUT / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"run_id": RUN_ID}


def write(out: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def committed() -> dict:
    rel = str((OUT / "metrics.json").relative_to(R))
    if cf._git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise SystemExit(f"{rel} is not tracked: commit it first")
    if cf._git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"{rel} has uncommitted changes: commit it first")
    return json.loads(cf._git("show", f"HEAD:{rel}").stdout)


def gate(status) -> tuple[dict, dict]:
    if cf._git("ls-files", "--error-unmatch", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} is not committed")
    if cf._git("diff", "--quiet", "HEAD", "--", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} has uncommitted changes")
    cr._in_certfold_env()
    prov = r2.clean()
    out = read_out()
    if out.get("status") != status:
        raise SystemExit(f"status is {out.get('status')!r}; this phase needs {status!r}")
    if status is not None:
        committed()
    out.setdefault("specification", {
        "path": SPEC, "commit": cf._git("log", "-1", "--format=%H", "--", SPEC).stdout.strip(),
        "sha256": cf._sha(R / SPEC)})
    return out, prov


def default_md():
    """The pinned reference model with dangles = 2 and special_hp = 1, each read back."""
    import RNA

    from thermo.vienna_ref import REFERENCE_MODEL, reference_md

    md = reference_md()
    md.dangles = 2
    md.special_hp = 1
    if md.dangles != 2 or md.special_hp != 1:
        raise GuardFailure(f"the default model did not take: dangles {md.dangles}, "
                           f"special_hp {md.special_hp}")
    bare = RNA.md()
    diff = {f: {"here": getattr(md, f), "bare_RNA_md": getattr(bare, f)}
            for f in REFERENCE_MODEL if getattr(md, f) != getattr(bare, f)}
    return md, diff


# ----------------------------------------------------------------------------------
# phase: fold -- item (a)'s folds and checks 1 to 3
# ----------------------------------------------------------------------------------
def fold() -> int:
    out, prov = gate(None)
    t0 = time.time()
    import RNA

    from scripts.certfold_baselines import _energy, _ninio_correction
    from scripts.turner_nullspace import Extractor
    from scripts.turner_par import clip_d2
    from thermo.vienna_ref import assert_turner2004_params, reference_md

    T = cf.template()
    th2 = clip_d2(T["theta"])
    m, cap = T["theta"][("ninio_m",)], T["theta"][("ninio_max",)]
    ex2 = Extractor(T["special"], 2, 1, lxc=T["lxc"])
    md2, diff = default_md()
    rows = r2.ts0_rows_with_stars()
    SHARDS.mkdir(parents=True, exist_ok=True)
    recs, n_changed, n_ninio = [], 0, 0
    for r in rows:
        seq = r["seq"]
        fc0 = RNA.fold_compound(seq, reference_md())
        assert_turner2004_params(fc0)
        if fc0.mfe()[0] != r["star"]:
            raise GuardFailure(f"{r['name']}: the dangles 0 MFE is not registration 1's s*")
        fc = RNA.fold_compound(seq, md2)
        assert_turner2004_params(fc)
        s, e = fc.mfe()
        got = int(round(e * 100))
        corr = _ninio_correction(s, m, cap)
        n_ninio += corr != 0
        want = _energy(ex2.counts(seq, s), th2) + corr
        if got != want:
            raise GuardFailure(f"{r['name']}: the default-model MFE energy {got} dcal is not the "
                               f"count model's {want} (dangles 2, special hairpins on)")
        n_changed += s != r["star"]
        recs.append({"index": r["index"], "name": r["name"], "default_mfe": s,
                     "energy_dcal": got})
    path = SHARDS / "default_mfe.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in recs))
    out["fold"] = {
        "model": {"dangles": 2, "special_hp": 1,
                  "fields_differing_from_a_bare_RNA_md": diff,
                  "every_other_field": "as thermo.vienna_ref.REFERENCE_MODEL"},
        "n_sequences": len(recs),
        "n_sequences_mfe_differs_from_dangles0": n_changed,
        "checks": {
            "1_model_read_back_and_turner2004_params": True,
            "2_energy_identity": "every MFE energy equals Extractor(special, 2, 1, lxc) counts . "
                                 "clip_d2(theta0) + offset + Ninio correction, integer dcal",
            "2_n_folds_checked": len(recs),
            "2_n_folds_with_a_nonzero_ninio_correction": n_ninio,
            "3_dangles0_mfe_is_registration_1s": True},
        "files_sha256": {"default_mfe.jsonl": cf._sha(path)},
        "seconds": round(time.time() - t0, 1), "provenance": prov,
        "environment": cf.env_record()}
    out["status"] = FOLDED
    write(out)
    print(json.dumps({k: v for k, v in out["fold"].items()
                      if k not in ("provenance", "environment")}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------------
# phase: analyse -- item (a) check 4, statistic and tables; item (b)
# ----------------------------------------------------------------------------------
def _hi_groups(recs: list[dict], rows: list[dict]):
    import scripts.certfold_exploratory as cx

    hi = [r for r in recs if r["kind"] == "unit"
          or (r["kind"] == "mfe" and r["p"] >= cr.P_HI)]
    return cx.strata_groups(hi, {r["index"]: r["stratum"] for r in rows})


def retention_table(recs: list[dict], key: str) -> dict:
    """Absolute retention by label among the p >= 0.99 MFE pairs, pooled and by stratum."""
    hi = [r for r in recs if r["kind"] == "mfe" and r["p"] >= cr.P_HI]

    def block(rs):
        c = [r for r in rs if r["certified_l1"]]
        u = [r for r in rs if not r["certified_l1"]]
        kc, ku = sum(r[key] for r in c), sum(r[key] for r in u)
        fc = kc / len(c) if c else None
        fu = ku / len(u) if u else None
        return {"n_certified": len(c), "n_uncertified": len(u),
                "kept_certified": fc, "kept_uncertified": fu,
                "lost_certified": len(c) - int(kc), "lost_uncertified": len(u) - int(ku),
                "difference_points": None if fc is None or fu is None else 100 * (fc - fu),
                "relative_reduction_in_loss": (None if fc is None or fu is None or fu == 1
                                               else 1 - (1 - fc) / (1 - fu))}

    by = collections.defaultdict(list)
    for r in hi:
        by[r["stratum"]].append(r)
    return {"pool": "TS0 MFE pairs (dangles 0) with p >= 0.99; certified = registration 1's "
                    "label at lambda = 1; kept = in the default-model MFE structure",
            "raw": block(hi), "by_stratum": {str(s): block(by[s]) for s in sorted(by)}}


def analyse() -> int:
    out, prov = gate(FOLDED)
    t0 = time.time()
    import scripts.certfold_exploratory as cx
    import scripts.certfold_reviewer as rv

    path = SHARDS / "default_mfe.jsonl"
    if cf._sha(path) != out["fold"]["files_sha256"]["default_mfe.jsonl"]:
        raise GuardFailure("default_mfe.jsonl changed since the fold phase")
    dflt = {d["index"]: set(cr._pairs_of(d["default_mfe"]))
            for d in (json.loads(ln) for ln in path.read_text().splitlines() if ln.strip())}

    # check 4: registration 2's own call, on its own records and seed, reproduces R4 exactly
    rows, recs_reg2 = rv.base_records()
    reg = cx.stratified_bootstrap(_hi_groups(recs_reg2, rows), n_boot=N_BOOT,
                                  seed_tag=r2.TAG_R4,
                                  specs={n: r2._r4_spec(k) for n, k in r2.R4_SPECS})
    want = r2.committed2()["analysis_c"]["R4"]["R4_primary"]
    if (reg["R4_primary"]["pooled_point"] != want["pooled_point"]
            or list(reg["R4_primary"]["pooled_interval"]) != list(want["pooled_interval"])):
        raise GuardFailure(f"the R4 code path does not reproduce registration 2's R4: "
                           f"{reg['R4_primary']['pooled_point']} "
                           f"{reg['R4_primary']['pooled_interval']} vs {want['pooled_point']} "
                           f"{want['pooled_interval']}")

    # item (a): the same records, "kept by the default-model MFE" in place of retention
    seq_recs, freqs = cr.read_shards(cf.read_out()["run"]["workers"])
    fbi = {int(k): v for w in freqs.values() for k, v in w["by_index"].items()}
    base = cr.pair_records([s for s in seq_recs if s["set"] == "TS0"], fbi)["TS0"]
    recs = [{**r, "ret_default": float(r["pair"] in dflt[r["index"]])} if r["kind"] == "mfe"
            else r for r in base if r["kind"] in ("mfe", "unit")]
    specs = {"default_model": r2._r4_spec("ret_default")}
    bs = cx.stratified_bootstrap(_hi_groups(recs, rows), n_boot=N_BOOT, seed_tag=TAG_DEFAULT,
                                 specs=specs)
    res_a = bs["default_model"]
    iv = res_a["pooled_interval"]
    mfe = [r for r in recs if r["kind"] == "mfe"]
    item_a = {
        "statistic": "R4's (certfold_reg2._r4_spec) on ret_default: mean kept by the "
                     "default-model MFE over certified p >= 0.99 pairs minus over uncertified, "
                     "within TR0 strata (0 and 1 merged), pooled by p >= 0.99 pair count",
        "pooled_point": res_a["pooled_point"], "pooled_interval": iv,
        "pooled_resamples_undefined": res_a["pooled_resamples_undefined"],
        "strata_left_out": res_a["strata_left_out"],
        "by_stratum": {k: {kk: v[kk] for kk in ("point", "interval", "n_certified",
                                                 "n_uncertified", "n_pairs") if kk in v}
                       for k, v in res_a["by_stratum"].items()},
        "bootstrap": bs["bootstrap"], "seed": [cf.SEED, TAG_DEFAULT], "n_boot": N_BOOT,
        "reading": ("interval above 0: Limitations' 'and no claim here covers it' becomes 'and "
                    "no guarantee here covers it, though certified pairs are kept more often in "
                    "its MFE (post hoc, S7)'" if iv is not None and iv[0] > 0 else
                    "otherwise: Limitations gains 'and the dangles 0 labels carry no detectable "
                    "signal into its MFE (post hoc, S7)'"),
        "absolute_retention": retention_table(recs, "ret_default"),
        "all_mfe_pairs_kept": {"n": int(sum(r["ret_default"] for r in mfe)), "of": len(mfe),
                               "fraction": sum(r["ret_default"] for r in mfe) / len(mfe)},
        "check_4_reproduced_R4": {"pooled_point": want["pooled_point"],
                                  "pooled_interval": want["pooled_interval"],
                                  "seed_tag": r2.TAG_R4}}

    # item (b): the decoupled-coverage bound from stored records
    B = cf.load_tr0_basis()
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    W1e = cf.Weights(B["u_env"], B["cols"], 1.0)
    tally = collections.defaultdict(lambda: [0, 0])
    n_neg = 0
    for r in rows:
        seq, star = r["seq"], r["star"]
        pt = cf.pair_table(star)
        sp = {(i, pt[i]) for i in range(1, len(seq) + 1) if pt[i] > i}
        if sp != {tuple(q["pair"]) for q in r["pairs"]}:
            raise GuardFailure(f"{r['name']}: the MFE pairs are not registration 1's")
        w_star = cf.eprime_offset(lc, seq, star, cf.star_interior_loops(lc, seq, star), W1e)
        if w_star < 0:
            raise GuardFailure(f"{r['name']}: W(s*) = {w_star} < 0")
        for q in r["pairs"]:
            gap = int(round(100 * float(q["nominal_gap_kcal"])))
            n_neg += gap < 0
            t = tally[r["stratum"]]
            t[0] += gap > w_star
            t[1] += 1
    if n_neg:
        raise GuardFailure(f"{n_neg} negative nominal gaps")
    coupled = cf.read_out()["analysis"]["sets"]["TS0"]["describe"]["labels"][cr.PRIMARY][
        "certified_present"]
    by = {}
    for s in sorted(tally):
        n, of = tally[s]
        c = coupled["by_stratum"][str(s)]
        if c["of"] != of:
            raise GuardFailure(f"stratum {s}: {of} MFE pairs here, {c['of']} in registration 1")
        by[str(s)] = {"n_bound": n, "of": of, "bound": n / of, "coupled": c["fraction"],
                      "informative": n / of < c["fraction"]}
    nb, ob = sum(v[0] for v in tally.values()), sum(v[1] for v in tally.values())
    item_b = {
        "bound": "share of TS0 MFE pairs with nominal gap > W(s*) (integer dcal, strict); "
                 "W(s*) = certfold.eprime_offset(s*) under Weights(u_env, cols, 1.0)",
        "pooled": {"n_bound": nb, "of": ob, "bound": nb / ob,
                   "coupled": coupled["pooled"]["fraction"],
                   "informative": nb / ob < coupled["pooled"]["fraction"]},
        "by_stratum": by,
        "reading": "where the bound is below the coupled share: 'a decoupled certificate could "
                   "certify at most X% (post hoc)'; elsewhere uninformative. Supplement only."}
    out["analysis"] = {"default_model_transfer": item_a, "decoupled_bound": item_b,
                       "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = ANALYSED
    write(out)
    print(json.dumps({"a": {k: item_a[k] for k in ("pooled_point", "pooled_interval",
                                                     "reading", "all_mfe_pairs_kept")},
                      "a_raw": item_a["absolute_retention"]["raw"],
                      "b_pooled": item_b["pooled"],
                      "b_by_stratum": {s: [round(v["bound"], 4), round(v["coupled"], 4)]
                                       for s, v in by.items()}}, indent=1, default=str))
    return 0



if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"fold": fold, "analyse": analyse}[ph]())
