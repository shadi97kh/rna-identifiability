#!/usr/bin/env python3
"""Fetch and register the MXfold2 datasets (CC BY 4.0), and unpack them for training.

The datasets of Sato, Akiyama & Sakakibara (Nat Commun 2021), distributed on Zenodo
as one tarball (doi:10.5281/zenodo.4430150, CC BY 4.0) with an md5 in the record.
Used by the MXfold2 seed study (scripts/mxfold2_seed_study.py): TR0 for training,
TS0 for the fixed evaluation set (bpRNA-1m splits as MXfold2 distributes them).

The tarball also contains an `archiveII/` directory. It is NOT used: ArchiveII's
upstream licence is unstated and the decision on downstream copies is the user's
(docs/BLOCKED.md item 27). Registering the tarball records what is on disk; it does
not adopt that directory.

Run: .venv/bin/python scripts/ingest_mxfold2_data.py          (fetch + register)
     .venv/bin/python scripts/ingest_mxfold2_data.py --unpack (into .cache/, gitignored)
"""

from __future__ import annotations

import datetime
import hashlib
import pathlib
import subprocess
import sys
import tarfile
import urllib.request

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "SOURCES.yaml"
KEY = "mxfold2_datasets"
URL = "https://zenodo.org/records/4430150/files/mxfold2-data.tar.gz?download=1"
MD5 = "45a4d51f01517dd782216920ccc510b7"  # from the Zenodo record
DEST = ROOT / "data" / "raw" / "structure" / "mxfold2-data.tar.gz"
UNPACK = ROOT / ".cache" / "mxfold2-data"

ENTRY = {
    "description": "The datasets of MXfold2 (TrainSetA/B, TestSetA/B, bpRNA-1m TR0/VL0/TS0 "
                   "canonical-pair versions, bpRNA-new, RNAStrAlign) as BPSEQ files with list "
                   "files, one tarball.",
    "landing_page": "https://doi.org/10.5281/zenodo.4430150",
    "download_url": URL,
    "licence": "CC BY 4.0 (Zenodo record metadata)",
    "citation": "Sato K, Akiyama M, Sakakibara Y. RNA secondary structure prediction using deep "
                "learning with thermodynamic integration. Nat Commun 2021;12:941. "
                "doi:10.1038/s41467-021-21194-4",
    "role": "TRAINING AND EVALUATION DATA for the MXfold2 seed study only: TR0 trains, TS0 "
            "is the fixed evaluation set. The bundled archiveII/ directory is NOT used "
            "(BLOCKED item 27).",
    "files": [],
}


def md5(path):
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack() -> pathlib.Path:
    """Extract into .cache/ with the 'data' filter (no absolute paths, no escapes)."""
    if not (UNPACK / "data" / "TR0-canonicals.lst").exists():
        UNPACK.mkdir(parents=True, exist_ok=True)
        with tarfile.open(DEST) as tf:
            tf.extractall(UNPACK, filter="data")
    return UNPACK


def main() -> int:
    if "--unpack" in sys.argv:
        print(unpack())
        return 0
    if not DEST.exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        tmp = DEST.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=600) as r, tmp.open("wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        if md5(tmp) != MD5:
            tmp.unlink()
            print("md5 mismatch against the Zenodo record", file=sys.stderr)
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
        "--note", f"Distributed artifact, no transcription. md5 {MD5} matches the Zenodo record.",
    ], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
