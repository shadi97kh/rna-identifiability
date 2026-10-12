#!/usr/bin/env python3
"""The vienna_ref Boltzmann-factor guard leaves every reference value unchanged.

Asked by the user 2026-10-12. thermo/vienna_ref.py fingerprinted only the ENERGY table before
calling pf(); ViennaRNA 2.7.2 caches the Boltzmann-factor table separately
(runs/2026-10-12_turner_par_writer V3), so a stale Boltzmann table under a correct energy
table passed the guard. The new guard (assert_turner2004_boltzmann) must change no value the
reference returns.

  before   run at the commit BEFORE the guard: ensemble_free_energy on the golden sequences
           (tests/test_vienna_reference.py) and on every TR0+TS0 sequence of length 21-120
           (the gate's range; registered loader, scripts/turner_nullspace_tr0.tr0_ts0_rows)
  after    the same calls with the guard in place; every value must be bitwise equal

Writes runs/2026-10-12_vienna_ref_boltzmann_guard/metrics.json (atomically).
Run: .venv/bin/python scripts/vienna_ref_guard_check.py --phase {before,after}
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402

RUN_ID = "2026-10-12_vienna_ref_boltzmann_guard"
OUT = R / "runs" / RUN_ID


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def sequences():
    from scripts.turner_nullspace_tr0 import tr0_ts0_rows
    from tests.test_vienna_reference import GOLDEN

    gate = sorted({r[2] for r in tr0_ts0_rows()[0] if 21 <= len(r[2]) <= 120})
    return [s for s, _ in GOLDEN.values()] + gate


def values():
    import thermo.vienna_ref as vr

    seqs = sequences()
    t0 = time.time()
    vals = [vr.ensemble_free_energy(s) for s in seqs]
    return seqs, vals, round(time.time() - t0, 1), hasattr(vr, "assert_turner2004_boltzmann")


GUARD_COMMIT = "d5bedbe"  # the commit that added assert_turner2004_boltzmann
PLANT = """
import RNA
from thermo.vienna_ref import ModelNotPinned, ensemble_free_energy, reference_md
seq = {seq!r}
RNA.params_load_RNA_Andronescu2007()
fc = RNA.fold_compound(seq, reference_md()); fc.mfe(); fc.pf()
RNA.params_load_RNA_Turner2004()
md = reference_md(); md.temperature = 25.0
RNA.fold_compound("GGGGAAACCCC", md).mfe()
try:
    print("RETURNED", repr(ensemble_free_energy(seq)))
except ModelNotPinned as e:
    print("CAUGHT", str(e).split(" -- ")[0])
"""


def planted():
    """The planted stale-Boltzmann state (Andronescu 2007's Boltzmann table under Turner
    2004's energy table) run against the guard before and after GUARD_COMMIT, each in a fresh
    process with only that version of thermo/vienna_ref.py importable."""
    import subprocess
    import tempfile

    from tests.test_vienna_reference import GOLDEN

    seq, golden = GOLDEN["trna_73"]
    res = {"sequence": "trna_73", "golden_kcal": golden}
    for tag, rev in (("before_guard", f"{GUARD_COMMIT}^"), ("with_guard", GUARD_COMMIT)):
        src = subprocess.run(["git", "show", f"{rev}:thermo/vienna_ref.py"], cwd=R,
                             capture_output=True, text=True, check=True).stdout
        with tempfile.TemporaryDirectory() as t:
            pkg = pathlib.Path(t) / "thermo"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "vienna_ref.py").write_text(src)
            run = subprocess.run([sys.executable, "-c", PLANT.format(seq=seq)], cwd=t,
                                 capture_output=True, text=True, timeout=300,
                                 env=dict(os.environ, PYTHONPATH=t))
        line = next((ln for ln in run.stdout.splitlines()
                     if ln.startswith(("RETURNED", "CAUGHT"))), run.stderr[-300:])
        res[tag] = {"vienna_ref_at": rev, "outcome": line,
                    "vienna_ref_sha256": hashlib.sha256(src.encode()).hexdigest()}
    out = json.loads((OUT / "metrics.json").read_text())
    out["planted_stale_boltzmann"] = res | {"provenance": git_provenance(R)}
    write(out)
    print(json.dumps(res, indent=1))
    return 0


def main():
    phase = sys.argv[sys.argv.index("--phase") + 1]
    if phase == "planted":
        return planted()
    seqs, vals, secs, guarded = values()
    digest = hashlib.sha256("\n".join(seqs).encode()).hexdigest()
    rec = {"n_sequences": len(seqs), "sequences_sha256": digest, "seconds": secs,
           "guard_present": guarded, "values": [repr(v) for v in vals], **git_provenance(R)}
    if phase == "before":
        if (OUT / "metrics.json").exists():
            raise SystemExit("the before record exists")
        assert not guarded, "run 'before' at the commit before the guard"
        write({"run_id": RUN_ID, "status": "before recorded", "before": rec})
    elif phase == "after":
        out = json.loads((OUT / "metrics.json").read_text())
        assert guarded, "the guard is not in thermo/vienna_ref.py"
        b = out["before"]
        assert b["sequences_sha256"] == digest, "the sequence set changed"
        diff = [i for i, (x, y) in enumerate(zip(b["values"], rec["values"], strict=True))
                if x != y]
        out["after"] = rec
        out["comparison"] = {"n": len(seqs), "n_bitwise_different": len(diff),
                             "first_different": [(seqs[i], b["values"][i], rec["values"][i])
                                                 for i in diff[:5]],
                             "unchanged": not diff,
                             "slowdown": rec["seconds"] / b["seconds"] if b["seconds"] else None}
        out["status"] = "complete"
        write(out)
        print(json.dumps(out["comparison"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
