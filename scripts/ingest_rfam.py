#!/usr/bin/env python3
"""Fetch and register the Rfam 15.1 seed alignments (CC0) as a structure source.

WHY RFAM, AND WHAT IT IS NOT

ArchiveII is BLOCKED (docs/BLOCKED.md item 27). Rfam is the replacement the user
chose: a canonical, versioned distribution with an explicit licence. The release
COPYING file states: "Rfam is freely available under the Creative Commons Zero
("CC0") licence." Its seed alignments carry a hand-curated consensus secondary
structure (#=GC SS_cons). A structure obtained for one seed sequence by projecting
that consensus is COMPARATIVE, not experimentally determined for that molecule,
and any use of it must say so. It is not ArchiveII and is not to be called that.

The file is pinned to the versioned path (15.1), not CURRENT, so a later release
cannot silently change what the digest refers to. The FTP publishes no checksum
for it; the sha256 recorded here is our own, taken at ingestion.

Run: .venv/bin/python scripts/ingest_rfam.py   (idempotent; no network if present)
"""

from __future__ import annotations

import datetime
import pathlib
import subprocess
import sys
import urllib.request

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "SOURCES.yaml"
KEY = "rfam_15_1_seed"
URL = "https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/Rfam.seed.gz"
COPYING = "https://ftp.ebi.ac.uk/pub/databases/Rfam/15.1/COPYING"
DEST = ROOT / "data" / "raw" / "structure" / "Rfam_15.1.seed.gz"
EXPECTED_BYTES = 5_928_432  # Content-Length reported by the FTP on 2026-10-07

ENTRY = {
    "description": "Rfam release 15.1 seed alignments in Stockholm format, with the hand-curated "
                   "consensus secondary structure of each family (#=GC SS_cons).",
    "landing_page": "https://rfam.org/",
    "download_url": URL,
    "licence": "CC0 1.0. Release COPYING: 'Rfam is freely available under the Creative Commons "
               "Zero (\"CC0\") licence.'",
    "licence_source": COPYING,
    "citation": "Ontiveros-Palacios N, et al. Rfam 15: RNA families database in 2025. Nucleic "
                "Acids Res 2025;53:D258-D267. doi:10.1093/nar/gkae1023",
    "role": "STRUCTURE GENERATING SET for the data-restricted Turner null space "
            "(scripts/turner_nullspace_rfam.py). Structures are consensus projections onto seed "
            "sequences -- comparative, not per-molecule experimental -- and are never a training "
            "label or a measurement. Substitutes for ArchiveII (BLOCKED item 27); "
            "it is not ArchiveII.",
    "files": [],
}


def main() -> int:
    if not DEST.exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        tmp = DEST.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=300) as r, tmp.open("wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        if tmp.stat().st_size != EXPECTED_BYTES:
            tmp.unlink()
            print(f"size mismatch: expected {EXPECTED_BYTES}", file=sys.stderr)
            return 1
        tmp.rename(DEST)
        print(f"downloaded {DEST.relative_to(ROOT)}")
    reg = yaml.safe_load(REGISTRY.read_text()) or {"sources": {}}
    if KEY not in reg["sources"]:
        reg["sources"][KEY] = dict(ENTRY, retrieved=datetime.date.today().isoformat())
        REGISTRY.write_text(yaml.safe_dump(reg, sort_keys=False, width=100))
        print(f"added registry entry {KEY}")
    return subprocess.call([
        sys.executable, str(ROOT / "scripts" / "register_source.py"), KEY,
        str(DEST.relative_to(ROOT)), "--route", "machine-readable", "--origin", URL,
        "--note", "Distributed artifact, no transcription. Size checked against the FTP "
                  "Content-Length; sha256 taken at ingestion (the FTP publishes no checksum).",
    ], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
