#!/usr/bin/env python3
"""CERTIFIED FOLDING, THE TEST-SET LOADERS. Which rows the run phase folds, and nothing else.

Every set the run phase certifies is read here, by one rule per set, so that the rows a label
was computed on can be named, counted and re-derived from a digest. Nothing in this module folds
a sequence, and nothing compares a prediction with a reference structure: the references are
carried as given, because accuracy on TS0, the held-out set and bpRNA-new belongs after the
registration, not in a loader.

THE SETS
  TS0         the bpRNA-1m test split as MXfold2 distributes it, read by the registered loader
              (scripts/turner_nullspace_tr0.tr0_ts0_rows) and filtered exactly as
              scripts/certfold.tr0_rows filters TR0: distinct (sequence, structure) over TR0+TS0,
              then the tag, then pair-bearing. 1,288 rows.
  held-out    Experiment 2's held-out families (scripts/typeb_heldout.heldout_rows, the only
              sanctioned read path; it checks heldout.tsv's sha256), ONE sequence per Rfam family
              by a seeded draw, so that no family weighs more than another. 1,641 families.
  bpRNA-new   every entry of data/bpRNAnew.nr500.canonicals.lst in the MXfold2 tarball, under
              TS0's rules, minus any sequence identical to a TR0 or TS0 sequence. The files carry
              no family labels, so this set can only be weighted per sequence.

THE LOADER RULES are tr0_ts0_rows's, which inlines them rather than factoring them out, so
bpseq_row replicates them and tarball_rows applies them to the tarball itself; the tests hold
tarball_rows to tr0_ts0_rows row for row and count for count on TR0 and TS0. The rules:
uppercase, T -> U, a sequence with any other letter dropped; a non-canonical or crossing pair
fails the read; a pair closing a hairpin of fewer than 3 unpaired bases removed, repeated to a
fixpoint. Format guards tr0_ts0_rows lacks (three fields per line, the index column 1..n, a
symmetric partner column) are added: they fail loudly on a malformed file rather than parse it
some other way, and they change nothing on a well-formed one.

PROVENANCE. tr0_ts0_rows reads the unpacked copy in .cache/mxfold2-data and checks no digest.
Here the tarball's sha256 is checked against its registry entry (mxfold2_datasets) through
scripts.typeb_heldout._registered, and the bytes that are parsed are hashed again, so what is
parsed is what was registered. ts0_rows also re-derives TR0 and TS0 from the tarball and refuses
if the unpacked copy the basis loader reads has drifted from it.

SEEDS. Every draw uses numpy.random.default_rng(SeedSequence([SEED, tag])), SEED = 20261013 as
in scripts/certfold.py. Tags in use elsewhere when this was written: certfold.py 1, 2, 3, 10, 11,
12 (11 and 12 with a per-sequence index), 50 (G5); certfold_baselines.py 30, 31;
certfold_metrics.py 40.
Tags here: 20 = the held-out one-per-family draw, 21 = the bpRNA-new subsample (defaults; a
different tag gives a different, equally valid draw, recorded in the draw's record). A tag must
be a non-negative int and not a bool: SeedSequence takes True as 1, which would silently replay
gate_sequences's tag-1 stream.

TR0_STRATA are the pilot's ten equal-count TR0 length strata, recomputed on first use through
scripts/certfold.pilot_sequences on scripts/certfold.tr0_rows, never typed in; length_stratum
maps any length onto them.

Guards raise GuardFailure and are never repaired.
"""

from __future__ import annotations

import bisect
import collections
import functools
import hashlib
import io
import json
import pathlib
import sys
import tarfile
from collections.abc import Sequence

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.turner_nullspace import PTYPE  # noqa: E402
from scripts.turner_nullspace_tr0 import distinct_rows, tr0_ts0_rows  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

SEED = 20261013                  # scripts/certfold.py's SEED; a test holds the two equal
SEED_TAG_HELDOUT = 20            # the held-out one-per-family draw
SEED_TAG_BPRNA_NEW = 21          # the bpRNA-new subsample
TARBALL = "data/raw/structure/mxfold2-data.tar.gz"   # registered as mxfold2_datasets
# tag -> (list file in the tarball, the directory every one of its entries must lie in)
LISTS = {
    "TR0": ("data/TR0-canonicals.lst", "data/bpRNA_dataset-canonicals/TR0/"),
    "TS0": ("data/TS0-canonicals.lst", "data/bpRNA_dataset-canonicals/TS0/"),
    "bpRNA-new": ("data/bpRNAnew.nr500.canonicals.lst",
                  "data/bpRNAnew_dataset/bpRNAnew.nr500.canonicals/"),
}
TS0_ROWS = 1288                  # distinct pair-bearing TS0 rows the basis loader yields
HELDOUT_FAMILIES = 1641          # Experiment 2's held-out families
BPRNA_NEW_LISTED = 5401          # entries of bpRNAnew.nr500.canonicals.lst
MIN_HAIRPIN = 3                  # the loader's rule: fewer unpaired bases closes no hairpin
CANONICAL_JSON = ("json.dumps(obj, separators=(',', ':'), ensure_ascii=True, sort_keys=True), "
                  "UTF-8")

Row = tuple[str, str, str, str]


def _rng(seed_tag: int) -> np.random.Generator:
    """default_rng(SeedSequence([SEED, seed_tag])), refusing a tag that is not a plain int >= 0.

    SeedSequence accepts True as 1 and a numpy integer as itself, so without this a bool would
    replay another draw's stream and a numpy tag would reach a record json cannot write.
    """
    if isinstance(seed_tag, bool) or not isinstance(seed_tag, int) or seed_tag < 0:
        raise GuardFailure(f"seed_tag {seed_tag!r} is not a non-negative int")
    return np.random.default_rng(np.random.SeedSequence([SEED, seed_tag]))


def canonical_sha256(obj) -> str:
    """sha256 of `obj` as canonical JSON (CANONICAL_JSON), so a list of names has one digest.

    Lists, not tuples, are what JSON writes; a tuple serialises as the same array, so a list of
    (family, name) pairs and a list of [family, name] lists share a digest by construction.
    """
    s = json.dumps(obj, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------
# the loader rules, replicated from tr0_ts0_rows, and the tarball they read
# ----------------------------------------------------------------------------------
def bpseq_row(text: str, where: str) -> tuple[str, str, int] | None:
    """(seq, db, pairs_removed) for one BPSEQ file under the registered loader rules, or None.

    tr0_ts0_rows's per-file body, line for line: None when the sequence keeps a non-ACGU letter
    after uppercasing and T -> U; a non-canonical or crossing pair fails the read; pairs closing
    a hairpin of fewer than MIN_HAIRPIN unpaired bases are removed to a fixpoint, and the number
    removed is returned so the caller can keep tr0_ts0_rows's counts. Text is read with
    universal newlines, as Path.read_text reads it. `where` names the file in a failure.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    rows = [r.split() for r in text.split("\n") if r.strip() and not r.startswith("#")]
    for k, r in enumerate(rows, start=1):
        if len(r) != 3 or not (r[0].isdigit() and int(r[0]) == k):
            raise GuardFailure(f"{where}: line {k} is not '<{k}> <base> <partner>': {r}")
    raw = "".join(r[1] for r in rows)
    partner = [0] + [int(r[2]) for r in rows]
    n = len(raw)
    for i in range(1, n + 1):
        j = partner[i]
        # j == i would otherwise pass the symmetry test and be read as unpaired, a silent repair
        if not 0 <= j <= n or j == i or (j and partner[j] != i):
            raise GuardFailure(f"{where}: partner column not symmetric at {i} -> {j}")
    seq = raw.upper().replace("T", "U")
    if any(c not in "ACGU" for c in seq):
        return None
    pt = [-1] * (n + 1)
    pairs = [(i, partner[i]) for i in range(1, n + 1) if partner[i] > i]
    for i, j in pairs:
        if seq[i - 1] + seq[j - 1] not in PTYPE:
            raise GuardFailure(f"{where}: non-canonical pair ({i}, {j})")
        pt[i], pt[j] = j, i
    for i, j in pairs:
        for k in range(i + 1, j):
            if pt[k] > j or 0 < pt[k] < i:
                raise GuardFailure(f"{where}: crossing pair ({i}, {j}) / ({k}, {pt[k]})")
    removed, changed = 0, True
    while changed:
        changed = False
        for a in range(1, n + 1):
            b = pt[a]
            if b > a and b - a - 1 < MIN_HAIRPIN and all(pt[x] == -1 for x in range(a + 1, b)):
                pt[a] = pt[b] = -1
                removed += 1
                changed = True
    db = "".join("(" if pt[k] > k else ")" if pt[k] > 0 else "." for k in range(1, n + 1))
    return seq, db, removed


def _tarball_bytes() -> tuple[bytes, str]:
    """The registered tarball's bytes and their sha256, refused unless they match the registry.

    _registered checks the file on disk against its registry entry; the bytes read here are
    hashed again, so a file replaced between the two reads is caught rather than parsed.
    _registered guards with assert, so its refusal is re-raised as the GuardFailure every other
    guard here raises.
    """
    from scripts.typeb_heldout import _registered

    try:
        want = _registered(TARBALL)
    except AssertionError as e:
        raise GuardFailure(f"{TARBALL}: {e}") from e
    data = (R / TARBALL).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise GuardFailure(f"{TARBALL}: sha256 {got} read, {want} registered")
    return data, got


def tarball_rows(tags: Sequence[str]) -> tuple[list[Row], dict]:
    """(tag, name, seq, db) rows for each tag's list, read from the registered tarball.

    The rows are tr0_ts0_rows's for the same lists, in list order and list after list, and the
    stats carry its keys (raw, dropped_non_ACGU, pairs_removed_short_hairpin,
    structures_touched_by_short_hairpin_rule, by_list), summed over the lists as it sums them,
    plus the same four counts per list under per_list, and the tarball's path and sha256. One
    streaming pass over the archive, keeping only the list files and the members under the
    directories the requested lists live in, so memory stays at those files' size. Every list
    entry must lie in its directory and be present; a missing one fails the read.
    """
    unknown = [t for t in tags if t not in LISTS]
    if unknown or not tags:
        raise GuardFailure(f"unknown or no list tags {list(tags)}; known: {sorted(LISTS)}")
    data, digest = _tarball_bytes()
    want_lists = {LISTS[t][0] for t in tags}
    prefixes = tuple(LISTS[t][1] for t in tags)
    lists, members = {}, {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for ti in tf:
            if not ti.isfile():
                continue
            if ti.name in want_lists:
                lists[ti.name] = tf.extractfile(ti).read().decode("utf-8")
            elif ti.name.startswith(prefixes) and "/._" not in ti.name:
                members[ti.name] = tf.extractfile(ti).read()
    stats = {"raw": 0, "dropped_non_ACGU": 0, "pairs_removed_short_hairpin": 0,
             "structures_touched_by_short_hairpin_rule": 0, "by_list": {}, "per_list": {},
             "tarball": TARBALL, "tarball_sha256": digest}
    keys = ("raw", "dropped_non_ACGU", "pairs_removed_short_hairpin",
            "structures_touched_by_short_hairpin_rule")
    rows = []
    for tag in tags:
        lst, prefix = LISTS[tag]
        if lst not in lists:
            raise GuardFailure(f"{TARBALL}: no {lst}")
        before = {k: stats[k] for k in keys}
        n_list = 0
        for ln in lists[lst].split("\n"):
            if not ln.strip():
                continue
            entry = ln.split()[0]
            if not entry.startswith(prefix) or entry not in members:
                raise GuardFailure(f"{lst}: entry {entry} is not a member under {prefix}")
            stats["raw"] += 1
            n_list += 1
            name = pathlib.PurePosixPath(entry).name
            got = bpseq_row(members[entry].decode("utf-8"), entry)
            if got is None:
                stats["dropped_non_ACGU"] += 1
                continue
            seq, db, removed = got
            stats["pairs_removed_short_hairpin"] += removed
            stats["structures_touched_by_short_hairpin_rule"] += int(removed > 0)
            rows.append((tag, name, seq, db))
        stats["by_list"][tag] = n_list
        stats["per_list"][tag] = {k: stats[k] - before[k] for k in keys}
    return rows, stats


# ----------------------------------------------------------------------------------
# TS0
# ----------------------------------------------------------------------------------
def ts0_rows() -> list[Row]:
    """The TS0 rows exactly as the certfold basis loader sees them: 1,288 (tag, name, seq, db).

    certfold.tr0_rows takes distinct_rows over TR0+TS0 and then the tag, so a TS0 row identical
    to an earlier TR0 row would be dropped here too; that order is kept rather than
    distinct-within-TS0 (the two give the same 1,288 today, and only this one is the basis
    loader's view). Pair-bearing, as tr0_rows is. Because tr0_ts0_rows reads the unpacked
    cache unchecked, the same rows are re-derived from the registered tarball and any
    difference fails the read: the set the run phase certifies is the registered one.
    """
    rows, _ = tr0_ts0_rows()
    from_tar, _ = tarball_rows(("TR0", "TS0"))
    if rows != from_tar:
        diff = sum(a != b for a, b in zip(rows, from_tar, strict=False))
        raise GuardFailure(f"the unpacked cache differs from the registered tarball: "
                           f"{len(rows)} vs {len(from_tar)} rows, {diff} differing in the overlap")
    ts0 = [r for r in distinct_rows(rows) if r[0] == "TS0" and "(" in r[3]]
    if len(ts0) != TS0_ROWS:
        raise GuardFailure(f"TS0 yields {len(ts0)} distinct pair-bearing rows, not {TS0_ROWS}")
    return ts0


# ----------------------------------------------------------------------------------
# the held-out families, one sequence each
# ----------------------------------------------------------------------------------
def heldout_one_per_family(seed_tag: int = SEED_TAG_HELDOUT) -> tuple[list[Row], dict]:
    """One sequence per Experiment 2 held-out family, drawn by a recorded seed.

    Rows are (family, name, seq, db), sorted by family. Families are visited in sorted order and
    each family's members in (name, seq) order, so the draw depends on the seed alone and not
    on heldout.tsv's row order; one generator, default_rng(SeedSequence([SEED, seed_tag])),
    draws one index per family in that order. One per family because a family with hundreds of
    seed sequences would otherwise dominate every fraction, and the families, not the
    sequences, are what Experiment 2 holds out. The record names the pool, the seed and the
    sha256 of the canonical JSON of the [(family, name)] list, which identifies the draw.
    """
    from scripts.typeb_heldout import heldout_rows
    from scripts.typeb_heldout import read_out as heldout_record

    rng = _rng(seed_tag)
    pool = heldout_rows()
    bad = [r for r in pool if len(r) != 4]
    if bad:
        raise GuardFailure(f"held-out rows are not (family, name, seq, db): {bad[:3]}")
    fam = collections.defaultdict(list)
    for r in pool:
        fam[r[0]].append(r)
    if len(fam) != HELDOUT_FAMILIES:
        raise GuardFailure(f"held-out pool has {len(fam)} families, not {HELDOUT_FAMILIES}")
    rows = []
    for f in sorted(fam):
        members = sorted(fam[f], key=lambda r: (r[1], r[2]))
        rows.append(members[int(rng.integers(len(members)))])
    ids = [[r[0], r[1]] for r in rows]
    record = {
        "n_families": len(rows), "n_sequences_in_pool": len(pool),
        "seed": [SEED, seed_tag], "seed_tag": seed_tag,
        "rule": "families in sorted order, members in (name, seq) order, one "
                "rng.integers(len(members)) per family from default_rng(SeedSequence(seed))",
        "family_name_sha256": canonical_sha256(ids),
        "canonical_json": CANONICAL_JSON + ", of [[family, name], ...] in family order",
        "heldout_tsv_sha256": heldout_record()["build"]["heldout_tsv_sha256"],
        "source": "scripts/typeb_heldout.heldout_rows (Experiment 2; Rfam consensus "
                  "projections, not structures determined per molecule)",
    }
    return rows, record


# ----------------------------------------------------------------------------------
# bpRNA-new
# ----------------------------------------------------------------------------------
def bprna_new_rows() -> tuple[list[Row], dict]:
    """Every bpRNA-new entry under TS0's rules, minus sequences identical to TR0 or TS0.

    Read from the registered tarball by the same tarball_rows pass that re-derives TR0 and TS0,
    so the duplicate check runs against the registered training and test sequences, every TR0
    and TS0 row the loader keeps, not only the distinct pair-bearing ones (a dropped non-ACGU
    sequence cannot equal an ACGU one). Then TS0's filter: distinct (sequence, structure),
    pair-bearing. Identity is on the sequence alone. Rows are ("bpRNA-new", name, seq, db) in
    list order. The record counts every step, and states what the set cannot support: no
    per-family weighting, and no claim of family disjointness from Experiment 2.
    """
    rows, st = tarball_rows(("TR0", "TS0", "bpRNA-new"))
    if st["by_list"]["bpRNA-new"] != BPRNA_NEW_LISTED:
        raise GuardFailure(f"bpRNA-new lists {st['by_list']['bpRNA-new']} entries, "
                           f"not {BPRNA_NEW_LISTED}")
    seen = {t: {r[2] for r in rows if r[0] == t} for t in ("TR0", "TS0")}
    new = [r for r in rows if r[0] == "bpRNA-new"]
    own = st["per_list"]["bpRNA-new"]
    steps = [{"step": "listed", "sequences": BPRNA_NEW_LISTED},
             {"step": "ACGU only after uppercasing and T -> U", "sequences": len(new)}]
    dist = distinct_rows(new)
    steps.append({"step": "distinct (sequence, structure)", "sequences": len(dist)})
    pb = [r for r in dist if "(" in r[3]]
    steps.append({"step": "pair-bearing", "sequences": len(pb)})
    in_tr0 = sum(r[2] in seen["TR0"] for r in pb)
    in_ts0 = sum(r[2] in seen["TS0"] for r in pb)
    in_both = sum(r[2] in seen["TR0"] and r[2] in seen["TS0"] for r in pb)
    out = [r for r in pb if r[2] not in seen["TR0"] and r[2] not in seen["TS0"]]
    steps.append({"step": "sequence identical to no TR0 or TS0 sequence", "sequences": len(out)})
    names = [r[1] for r in out]
    if len(set(names)) != len(names):
        raise GuardFailure("bpRNA-new names are not unique; a name cannot identify a row")
    by_seq = collections.Counter(r[2] for r in out)
    lens = np.array([len(r[2]) for r in out])
    record = {
        "source": "MXfold2 datasets (mxfold2_datasets, CC BY 4.0): " + LISTS["bpRNA-new"][0],
        "tarball": TARBALL, "tarball_sha256": st["tarball_sha256"],
        "n_listed": BPRNA_NEW_LISTED,
        "dropped_non_ACGU": own["dropped_non_ACGU"],
        "pairs_removed_short_hairpin": own["pairs_removed_short_hairpin"],
        "structures_touched_by_short_hairpin_rule":
            own["structures_touched_by_short_hairpin_rule"],
        "loader": "scripts/turner_nullspace_tr0.tr0_ts0_rows's rules via bpseq_row; then TS0's "
                  "filter (distinct (sequence, structure), pair-bearing)",
        "steps": steps,
        "dropped_identical_to_TR0_or_TS0": len(pb) - len(out),
        "identical_to_TR0": in_tr0, "identical_to_TS0": in_ts0, "identical_to_both": in_both,
        "sequences_with_more_than_one_structure": sum(c > 1 for c in by_seq.values()),
        "n_rows": len(out),
        "length_min_median_max": [int(lens.min()), float(np.median(lens)), int(lens.max())],
        "rows_sha256": canonical_sha256([[r[1], r[2], r[3]] for r in out]),
        "canonical_json": CANONICAL_JSON + ", of [[name, seq, db], ...] in list order",
        "weighting": "PER SEQUENCE ONLY. The BPSEQ files carry no Rfam family labels, so this "
                     "set cannot be weighted per family or drawn one per family; every "
                     "fraction over it is a per-sequence fraction.",
        "family_overlap_caveat": "bpRNA-new's families are Rfam families absent from bpRNA-1m, "
                                 "and Experiment 2's held-out families are Rfam families absent "
                                 "from Rfam 12.2, which bpRNA-1m drew from; the two sets may "
                                 "therefore share families, and with no family labels here the "
                                 "overlap is not measured, so the two cannot be read as "
                                 "independent replicates of one another.",
    }
    return out, record


def bprna_new_subsample(n: int = 2500,
                        seed_tag: int = SEED_TAG_BPRNA_NEW) -> tuple[list[Row], dict]:
    """A seeded subsample of bpRNA-new without replacement, sorted by name.

    The draw indexes the name-sorted pool, so it depends on the seed alone and not on the list
    file's order: rng.choice(len(pool), size=n, replace=False) from
    default_rng(SeedSequence([SEED, seed_tag])). The record carries the seed, n, the pool size,
    the sha256 of the canonical JSON of the sorted name list, and the pool's own record.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise GuardFailure(f"n = {n!r} is not a positive integer")
    rng = _rng(seed_tag)
    pool, prec = bprna_new_rows()
    if n > len(pool):
        raise GuardFailure(f"n = {n} exceeds the bpRNA-new pool of {len(pool)}")
    pool = sorted(pool, key=lambda r: r[1])
    pick = rng.choice(len(pool), size=n, replace=False)
    rows = sorted((pool[int(i)] for i in pick), key=lambda r: r[1])
    names = [r[1] for r in rows]
    record = {
        "n": n, "n_pool": len(pool), "seed": [SEED, seed_tag], "seed_tag": seed_tag,
        "rule": "pool sorted by name; rng.choice(len(pool), size=n, replace=False) from "
                "default_rng(SeedSequence(seed)); rows sorted by name",
        "names_sha256": canonical_sha256(names),
        "canonical_json": CANONICAL_JSON + ", of the sorted name list",
        "pool": prec,
    }
    return rows, record


# ----------------------------------------------------------------------------------
# the pilot's TR0 length strata
# ----------------------------------------------------------------------------------
@functools.cache
def _tr0_strata() -> tuple[tuple[int, int], ...]:
    """(min_len, max_len) per pilot stratum, recomputed by certfold.pilot_sequences itself.

    Imported here rather than at the top so that scripts/certfold.py can import this module
    without a cycle. Cached: the strata are a function of the TR0 rows alone, and TR0 takes a
    couple of seconds to load.
    """
    from scripts.certfold import PILOT_STRATA, pilot_sequences, tr0_rows

    rows, _ = tr0_rows()
    _, bounds = pilot_sequences(rows)
    if len(bounds) != PILOT_STRATA or [b["stratum"] for b in bounds] != list(range(PILOT_STRATA)):
        raise GuardFailure(f"pilot_sequences gave {len(bounds)} strata, not {PILOT_STRATA}")
    return tuple((int(b["min_len"]), int(b["max_len"])) for b in bounds)


def __getattr__(name: str):
    """TR0_STRATA, computed on first use: a fresh list each access, so no caller can edit it."""
    if name == "TR0_STRATA":
        return list(_tr0_strata())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def length_stratum(n_nt: int, bounds: Sequence[tuple[int, int]]) -> int:
    """The stratum a length falls in: the first whose max_len reaches n_nt.

    Equal-count bins over a sorted pool can cut through a run of equal lengths, so a length can
    be both one stratum's max_len and the next one's min_len (most of TR0's boundaries are
    such lengths); it goes to the LOWER stratum. A length in a gap between two
    strata goes to the upper one, below the first stratum to the first, beyond the last to the
    last. The map is monotone and reaches every stratum, which is what the bounds are refused
    for otherwise: unsorted, overlapping by more than one shared length, or two strata with the
    same max_len (the second could never be returned).
    """
    if not bounds:
        raise GuardFailure("no strata")
    for s, (lo, hi) in enumerate(bounds):
        if lo > hi:
            raise GuardFailure(f"stratum {s}: min_len {lo} > max_len {hi}")
        if s and (hi <= bounds[s - 1][1] or lo < bounds[s - 1][1]):
            raise GuardFailure(f"stratum {s} ({lo}, {hi}) does not follow {bounds[s - 1]}")
    if isinstance(n_nt, bool) or not isinstance(n_nt, int | np.integer) or n_nt < 1:
        raise GuardFailure(f"length {n_nt!r} is not a positive integer")
    return min(bisect.bisect_left([hi for _lo, hi in bounds], int(n_nt)), len(bounds) - 1)
