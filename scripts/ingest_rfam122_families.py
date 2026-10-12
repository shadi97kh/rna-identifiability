#!/usr/bin/env python3
"""Fetch and register the Rfam 12.2 family list (CC0): which families bpRNA-1m could contain.

E2 (scripts/typeb_heldout.py) holds out Rfam 15.1 families that are ABSENT from Rfam 12.2,
the release bpRNA-1m drew from (Danaee et al. 2018, Nucleic Acids Res 46:5381, open full
text PMC6009582: "... and all families from RFAM 12.2"; "There are 2495 families in RFAM V12.2
and 43 273 individual sequences."). bpRNA-1m also includes six other databases (CRW, tmRNA,
tRNAdb, SRP, RNase P, PDB), so family novelty does not imply sequence novelty. E2 also removes
sequences at >= 80% identity to TR0+TS0.

The file is pinned to the versioned path (12.2). The release COPYING says: "Rfam is freely
available under the Creative Commons Zero ("CC0") licence." The FTP publishes no checksum; the
sha256 recorded here is our own, taken at ingestion.

Run: .venv/bin/python scripts/ingest_rfam122_families.py   (idempotent; no network if present)
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
KEY = "rfam_12_2_families"
URL = "https://ftp.ebi.ac.uk/pub/databases/Rfam/12.2/database_files/family.txt.gz"
COPYING = "https://ftp.ebi.ac.uk/pub/databases/Rfam/12.2/COPYING"
DEST = ROOT / "data" / "raw" / "structure" / "Rfam_12.2.family.txt.gz"
EXPECTED_BYTES = 233_455  # Content-Length reported by the FTP on 2026-10-12

ENTRY = {
    "description": "Rfam release 12.2 family table (database_files/family.txt.gz): one row per "
                   "family, accession first.",
    "landing_page": "https://rfam.org/",
    "download_url": URL,
    "licence": "CC0 1.0. Release COPYING: 'Rfam is freely available under the Creative Commons "
               "Zero (\"CC0\") licence.'",
    "licence_source": COPYING,
    "citation": "Nawrocki EP, et al. Rfam 12.0: updates to the RNA families database. Nucleic "
                "Acids Res 2015;43:D130-D137 (release 12.2: rfam.org).",
    "role": "FAMILY LIST ONLY: which Rfam families bpRNA-1m could contain (bpRNA-1m drew 'all "
            "families from RFAM 12.2', Danaee 2018, PMC6009582). Used by scripts/typeb_heldout.py "
            "to select Rfam 15.1 families absent from 12.2. No sequence or structure is read.",
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
