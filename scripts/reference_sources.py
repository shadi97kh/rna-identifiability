#!/usr/bin/env python3
"""Where the TR0 and TS0 reference structures come from, by bpRNA source database. DESCRIPTIVE:
a count of the data behind two sentences of the Limitations (most TR0+TS0 references, and most
of TS0's, are Rfam consensus structures projected onto single sequences); nothing is folded.

bpRNA-1m names each structure after the database it was taken from (bpRNA_CRW_*, bpRNA_RFAM_*,
bpRNA_PDB_*, ...); its RFAM entries come from Rfam 12.2 seed alignments, each consensus
structure projected onto one seed sequence (Danaee et al. 2018). The rows are the registered
loader's, scripts/turner_nullspace_tr0.tr0_ts0_rows, imported unchanged, and its counts must equal
the registered record runs/2026-10-08_turner_nullspace_tr0 (11,970 rows, 11,955 of them
pair-bearing); TS0's rows must be the 1,288 certified folding evaluates
(runs/2026-10-13_certified_folding, /gate/G5/by_test_set/TS0/n_sequences). Every file name must
parse as bpRNA_<source>_<id>.bpseq, or the run stops.

Records runs/2026-10-30_reference_sources/metrics.json.
Run: .venv/bin/python scripts/reference_sources.py
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import re
import sys

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_nullspace_tr0 import tr0_ts0_rows  # noqa: E402

OUT = R / "runs" / "2026-10-30_reference_sources"
LOADER = R / "runs" / "2026-10-08_turner_nullspace_tr0" / "metrics.json"
CERTFOLD = R / "runs" / "2026-10-13_certified_folding" / "metrics.json"
NAME = re.compile(r"^bpRNA_([A-Za-z]+)_\d+\.bpseq$")


class CheckFailed(RuntimeError):
    pass


def source(name: str) -> str:
    m = NAME.match(name)
    if not m:
        raise CheckFailed(f"file name does not parse as bpRNA_<source>_<id>.bpseq: {name}")
    return m.group(1)


def tally(rows) -> dict:
    by = collections.Counter(source(name) for _tag, name, _seq, _db in rows)
    return {"n": len(rows), "by_source": dict(sorted(by.items())), "rfam": by.get("RFAM", 0),
            "rfam_fraction": by.get("RFAM", 0) / len(rows)}


def main() -> int:
    prov = git_provenance(R)
    rows, stats = tr0_ts0_rows()
    reg = json.loads(LOADER.read_text())["tr0_ts0"]
    for k in ("raw", "dropped_non_ACGU", "by_list"):
        if stats[k] != reg["loader"][k]:
            raise CheckFailed(f"loader {k}: {stats[k]} here, {reg['loader'][k]} registered")
    if len(rows) != reg["counts"]["distinct"] or len({(s, d) for *_x, s, d in rows}) != len(rows):
        raise CheckFailed(f"{len(rows)} rows, registered {reg['counts']['distinct']} distinct")
    pair_bearing = [r for r in rows if "(" in r[3]]
    if len(pair_bearing) != reg["counts"]["distinct_pair_bearing"]:
        raise CheckFailed(f"{len(pair_bearing)} pair-bearing rows, registered "
                          f"{reg['counts']['distinct_pair_bearing']}")
    ts0 = [r for r in rows if r[0] == "TS0"]
    n_ts0 = json.loads(CERTFOLD.read_text())["gate"]["G5"]["by_test_set"]["TS0"]["n_sequences"]
    if len(ts0) != n_ts0:
        raise CheckFailed(f"{len(ts0)} TS0 rows, certified folding evaluates {n_ts0}")

    out = {"run_id": OUT.name,
           "label": "DESCRIPTIVE: bpRNA source database of each TR0+TS0 reference, by file name",
           "status": "done", "provenance": prov, "script": "scripts/reference_sources.py",
           "checks": {"loader_equals": "runs/2026-10-08_turner_nullspace_tr0 /tr0_ts0",
                      "ts0_equals": "runs/2026-10-13_certified_folding "
                                    "/gate/G5/by_test_set/TS0/n_sequences"},
           "all": tally(rows),
           "pair_bearing": tally(pair_bearing),
           "TR0": tally([r for r in rows if r[0] == "TR0"]),
           "TS0": tally(ts0),
           "TS0_pair_bearing": tally([r for r in ts0 if "(" in r[3]])}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")
    for k in ("all", "pair_bearing", "TR0", "TS0", "TS0_pair_bearing"):
        print(f"{k}: {out[k]['rfam']} of {out[k]['n']} RFAM; {out[k]['by_source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
