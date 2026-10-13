#!/usr/bin/env python3
"""The certified-folding pilot's decision figure: what each candidate set certifies on TR0.

For the authors' choice of the primary set (2026-10-13, item 7), NOT a paper figure: the pilot
is training data and not a registered test. Six panels, every one computed from the pilot's
per-sequence records so that nothing is read off a summary that could hide a denominator:

  (a) the label fractions under each set: certified present, certified absent, whole structure
  (b) the certified-present fraction by length stratum, one line per set
  (c) rho for the ball (kcal/mol), as an ECDF over MFE pairs, the three published radii marked
  (d) rho_env for the envelope (multiples of it), lambda = 1 marked
  (e) the uncertain-present fraction against the competitor-side exposure, by quartile
  (f) runtime per sequence against length, timeouts marked at the budget

Reads runs/2026-10-13_certified_folding/metrics.json (committed) for the sets, the strata, the
budget and the shard digests, and the gitignored shards it names, each checked against its
recorded sha256. With --preview it reads whatever shards exist, unchecked, for a look while the
pilot is still running; a preview is labelled as one on the figure itself.

Writes runs/2026-10-13_certified_folding/pilot_decision.{pdf,png}.
Run: .venv/bin/python scripts/certfold_pilot_figure.py [--preview]
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "paper" / "recomb2027" / "figures"))
import style  # noqa: E402

RUN = R / "runs" / "2026-10-13_certified_folding"
ART = RUN / "metrics.json"
SHARDS = RUN / "shards"
ORDER = ("ball:r1", "ball:Langdon2018", "ball:Andronescu2007", "ball:Turner1999", "envelope")
COLOR = {"ball:r1": style.GREY, "ball:Langdon2018": style.GREEN,
         "ball:Andronescu2007": style.BLUE, "ball:Turner1999": style.SKY,
         "envelope": style.VERMILLION}


def label(k: str, sets: dict) -> str:
    kind, x = sets[k]["kind"], sets[k]["scale"]
    if kind == "envelope":
        return "envelope, $\\lambda$ = 1"
    name = k.split(":")[1]
    return "ball, r = 1" if name == "r1" else f"ball, {style.LABEL[name]} ({x:.1f})"


def load(preview: bool):
    if preview:
        files = sorted(SHARDS.glob("pilot_[0-9].jsonl"))
        meta = None
    else:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(ART.relative_to(R))],
                                 cwd=R, capture_output=True).returncode == 0
        assert tracked, "the pilot record is not committed"
        meta = json.loads(ART.read_text())["pilot"]
        files = []
        for f, h in meta["files_sha256"].items():
            got = hashlib.sha256((SHARDS / f).read_bytes()).hexdigest()
            assert got == h, f"{f} changed since the pilot ({got[:12]} != {h[:12]})"
            files.append(SHARDS / f)
    per = [json.loads(line) for f in files for line in f.open() if line.strip()]
    return per, meta


def ecdf(ax, vals, n_total, capped, tied, cap, color, lab):
    """ECDF over ALL pairs with a rho. Pairs tied at r = 0 (a degenerate MFE: rho = 0) start the
    curve as a step at the left edge, since a log axis cannot show 0; finite values follow; capped
    pairs are the mass at the cap. The curve reaches 1 exactly when every pair is accounted for."""
    v = np.sort(np.asarray(vals, dtype=float))
    if not n_total:
        return
    left = v[0] / 1.5 if len(v) else cap / 10
    y = (tied + np.arange(1, len(v) + 1)) / n_total
    ax.step(np.r_[left, v, cap], np.r_[tied / n_total, y, (tied + len(v) + capped) / n_total],
            where="post", color=color, label=lab)


def plain_log(ax, axis, ticks):
    """A log axis with plain-number ticks: the default mathtext labels collide at 7 pt."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_locator(NullLocator())
    a.set_minor_formatter(NullFormatter())


def main() -> int:
    preview = "--preview" in sys.argv
    per, meta = load(preview)
    if not per:
        print("no pilot records yet")
        return 1
    sets = {k: {"kind": v["kind"], "scale": v["scale"]} for k, v in per[0]["sets"].items()}
    style.setup()
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(style.FULL * 1.35, style.FULL * 0.78),
                            constrained_layout=True)
    a, b, c, d, e, f = axs.ravel()

    # (a) label fractions
    w = 0.26
    x = np.arange(len(ORDER))
    for off, key, name, col in ((-w, "present", "certified present", style.BLUE),
                                (0.0, "absent", "certified absent", style.ORANGE),
                                (w, "whole", "whole structure", style.BLACK)):
        ys = []
        for k in ORDER:
            if key == "present":
                lab_n = sum(p["sets"][k]["n_present_labelled"] for p in per)
                ys.append(sum(p["sets"][k]["n_certified_present"] for p in per) / max(1, lab_n))
            elif key == "absent":
                done = [p for p in per if p["sets"][k]["absent"] is not None]
                ys.append(sum(p["sets"][k]["absent"]["n_certified_absent"] for p in done)
                          / max(1, sum(p["n_admissible_outside_star"] for p in done)))
            else:
                det = [p["sets"][k]["whole_structure_certified"] for p in per
                       if p["sets"][k]["whole_structure_certified"] is not None]
                ys.append(sum(det) / max(1, len(det)))
        a.bar(x + off, ys, w, color=col, label=name)
    a.set_xticks(x, ["r = 1", "Langdon", "Andronescu", "Turner", "envelope"], rotation=20)
    a.set_ylim(0, 1.05)
    a.set_ylabel("fraction")
    a.legend(loc="upper center", bbox_to_anchor=(0.6, 0.98), fontsize=5.5, handlelength=1.0)
    style.panel(a, "a")

    # (b) certified present by length stratum
    strata = sorted({p["stratum"] for p in per})
    mid = [np.median([p["n_nt"] for p in per if p["stratum"] == s]) for s in strata]
    for k in ORDER:
        ys = []
        for s in strata:
            sel = [p for p in per if p["stratum"] == s]
            n = sum(p["sets"][k]["n_present_labelled"] for p in sel)
            ys.append(sum(p["sets"][k]["n_certified_present"] for p in sel) / n if n else np.nan)
        b.plot(mid, ys, marker="o", color=COLOR[k], label=label(k, sets))
    b.set_xscale("log")
    plain_log(b, "x", [40, 60, 100, 200, 400])
    b.set_xlabel("sequence length (stratum median, nt)")
    b.set_ylabel("certified present")
    b.set_ylim(-0.02, 1.02)
    b.legend(fontsize=5.5, loc="center left")
    style.panel(b, "b")

    # (c) rho, ball
    rb = [v for p in per for v in p["rho"]["ball"]["finite_values"]]
    nb = sum(p["rho"]["ball"]["n_done"] for p in per)
    capb = sum(p["rho"]["ball"]["capped"] for p in per)
    rcap = meta["rho_ball"]["cap"] if meta else 96.529
    tb = sum(p["rho"]["ball"]["tied_at_zero"] for p in per)
    ecdf(c, rb, nb, capb, tb, rcap, style.BLUE, "ball")
    for k in ("ball:Langdon2018", "ball:Andronescu2007", "ball:Turner1999"):
        c.axvline(sets[k]["scale"], color=COLOR[k], lw=0.7, ls="--")
    c.set_xscale("log")
    plain_log(c, "x", [0.5, 1, 2, 5, 10, 20, 50, 100])
    c.set_xlabel(r"$\rho(p)$, ball (kcal/mol)")
    c.set_ylabel("fraction of MFE pairs")
    c.set_ylim(0, 1.02)
    c.text(0.03, 0.97, f"{nb} pairs\n{tb} tied at 0\n{capb} at the cap",
           transform=c.transAxes, ha="left", va="top", fontsize=6)
    style.panel(c, "c")

    # (d) rho_env
    re_ = [v for p in per for v in p["rho"]["envelope"]["finite_values"]]
    ne = sum(p["rho"]["envelope"]["n_done"] for p in per)
    cape = sum(p["rho"]["envelope"]["capped"] for p in per)
    te = sum(p["rho"]["envelope"]["tied_at_zero"] for p in per)
    ecdf(d, re_, ne, cape, te, meta["rho_env"]["cap"] if meta else 4.0, style.VERMILLION,
         "envelope")
    d.axvline(1.0, color=style.VERMILLION, lw=0.7, ls="--")
    d.set_xscale("log")
    plain_log(d, "x", [0.1, 0.25, 0.5, 1, 2, 4])
    d.set_xlabel(r"$\rho_{\mathrm{env}}(p)$ (multiples of the envelope)")
    d.set_ylabel("fraction of MFE pairs")
    d.set_ylim(0, 1.02)
    d.text(0.03, 0.97, f"{ne} pairs\n{te} tied at 0\n{cape} at the cap",
           transform=d.transAxes, ha="left", va="top", fontsize=6)
    style.panel(d, "d")

    # (e) uncertain present against exposure
    expo = np.array([p["exposure"]["n_reachable_admissible_interior_loops"] for p in per])
    qs = np.quantile(expo, [0.25, 0.5, 0.75])
    bins = np.digitize(expo, qs)
    for k in ("ball:Langdon2018", "ball:Andronescu2007", "envelope"):
        xs, ys = [], []
        for qb in range(4):
            sel = [p for p, bb in zip(per, bins, strict=True) if bb == qb]
            n = sum(p["sets"][k]["n_present_labelled"] for p in sel)
            if n:
                xs.append(np.median(expo[bins == qb]))
                ys.append(1 - sum(p["sets"][k]["n_certified_present"] for p in sel) / n)
        e.plot(xs, ys, marker="o", color=COLOR[k], label=label(k, sets))
    e.set_xscale("log")
    plain_log(e, "x", [100, 200, 500, 1000, 2000, 5000, 10000])
    e.set_xlabel("exposure: reachable admissible\ninterior loops (quartile median)")
    e.set_ylabel("uncertain present")
    e.set_ylim(-0.02, 1.02)
    e.legend(fontsize=5.5)
    style.panel(e, "e")

    # (f) runtime
    lens = np.array([p["n_nt"] for p in per])
    secs = np.array([p["seconds"] for p in per])
    to = np.array([p["timed_out"] for p in per])
    f.scatter(lens[~to], secs[~to], s=4, color=style.BLACK, label="completed")
    f.scatter(lens[to], secs[to], s=10, marker="x", color=style.VERMILLION,
              label=f"timed out ({int(to.sum())})")
    budget = meta["budget_seconds_per_sequence"] if meta else 1800
    f.axhline(budget, color=style.GREY, lw=0.7, ls="--")
    f.set_xscale("log")
    f.set_yscale("log")
    plain_log(f, "x", [40, 60, 100, 200, 400])
    f.set_xlabel("sequence length (nt)")
    f.set_ylabel("seconds per sequence")
    f.legend(fontsize=6, loc="upper left")
    style.panel(f, "f")

    tag = (f"PREVIEW: {len(per)} of 300 sequences, shards unchecked" if preview else
           f"TR0 pilot, {len(per)} sequences (not a registered test)")
    fig.suptitle(f"Certified folding on TR0: the candidate sets. {tag}", fontsize=8)
    out = RUN / ("pilot_decision_preview" if preview else "pilot_decision")
    fig.savefig(out.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out.with_suffix(".png"))
    print(out.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
