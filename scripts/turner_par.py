#!/usr/bin/env python3
"""Write a ViennaRNA v2.0 parameter file from a Turner theta: the inverse of
scripts/turner_nullspace.theta_from_par. The only new code the Type B experiment needs.

DESIGN

The writer edits a TEMPLATE, the Turner 2004 table exactly as ViennaRNA writes it
(RNA.params_save, via turner_nullspace.dump_params), at the token positions theta_from_par
reads, and nowhere else. The positions are not re-derived here. They come from running
theta_from_par on a PROBE copy of the template in which every value token is replaced by
its own position number, so theta_from_par(probe)[key] is the position of key. The writer is
the inverse of the parser by construction: whatever the parser's indexing is, the writer
inherits it, and a change to the parser moves the writer with it. Every token the parser
does not read (all enthalpies, the non-canonical pair type, N rows and columns, lxc,
DuplexInit) is copied from the template unchanged.

PRECISION

The format holds integers in dcal/mol (0.01 kcal/mol), which is also ViennaRNA's internal
energy unit, so no finer value can reach the model by any route. write_par refuses a value
that is not an integer; round_to_format rounds explicitly (nearest, ties to even) and
returns the residual, so no rounding is ever silent. A template INF is never replaced by a
finite value (that would make a forbidden motif allowed). Template tokens are kept verbatim
wherever the value is unchanged, so DEF stays DEF and the unperturbed write is the template
byte for byte.

TEMPERATURE

Only the 37 C free energies are written; the enthalpies stay the template's. At 37 C
ViennaRNA's rescaling dH - (dH - dG) * T/T37 returns dG exactly, so a written dG is used as
written. At any other temperature the template enthalpies would be inconsistent with a
perturbed dG. Everything here pins 37 C through thermo.vienna_ref.reference_md.

PROCESS-GLOBAL STATE, AND A CACHE THAT SURVIVES params_load

ViennaRNA's parameter table is process-global (thermo/vienna_ref.py, hazard 3). Worse,
measured here on 2.7.2 (recorded by --check as V1): a new fold compound whose model details
equal the previous compound's reuses that compound's scaled table, EVEN AFTER params_load.
A file loaded after a compound was built with model A is invisible to the next compound
built with model A; a compound with different model details forces the rebuild. So
loaded() builds a compound with a sentinel model (25 C, never used for a measurement)
after every load and after the restore, and every measurement checks the table its first
compound actually carries against the theta it is meant to carry (params_match). The
rebuild is the remedy; the per-configuration check is the guard.

The Boltzmann-factor table used by the partition function has a SECOND, separate last-model
cache (recorded by --check as V3, found 2026-10-12 by the Type B smoke test: two changed
MFEs with an exactly unchanged base-pair-probability matrix). A compound that only computes
an MFE refreshes the energy table but not the Boltzmann factors, so _rebuild also runs a
partition function on the sentinel compound. params_match reads only the energy table; a
caller that uses pf() must check the Boltzmann factors separately (Type B constrains a pf to
one structure and compares its free energy with counts . theta).

A NONLINEARITY OF THE MODEL: d2 CLIPS FOUR TABLES AT ZERO

With dangles = 2, ViennaRNA's scaled table holds min(x, 0) for every entry of
mismatch_exterior, mismatch_multi, dangle5 and dangle3 (recorded by --check as V2: with
positive values in those tables the d2 energies equal counts . clip(theta) on every row).
Turner 2004 lies entirely on the linear side (all 240 entries <= 0; dangle5 reaches 0, the
clip point itself). A perturbation that makes any of these entries positive is clipped,
so for d2 a null-space direction is linear only while those entries stay <= 0. With
dangles = 0 the four tables are not used. The hairpin and interior-loop mismatches are not
clipped (they have positive Turner 2004 entries, and W4 passes with positive labels).

CHECKS (--check; writes runs/<RUN_ID>/metrics.json)

  W1  identity: write_par(theta_2004) is the template, byte for byte.
  W2  parser round trip: theta_from_par(write_par(theta)) == theta exactly, for theta_2004
      and for the labelled theta below.
  W3  ViennaRNA's reader, unperturbed: with the written Turner 2004 file loaded,
      eval_structure on the 23,741 generating-set rows of
      runs/2026-10-07_turner_nullspace_r2 equals stock Turner 2004's energies exactly
      (integer dcal/mol), in all four configurations (dangles 0/2 x special_hp 0/1).
  W4  ViennaRNA's reader, labelled: a LABELLED theta gives every column a distinct value
      (1 + its rank by key, negated in the four d2-clipped tables so they stay on the
      linear side; the ninio pair is set to (a, 5a) with a above every rank, so
      that MIN2(ninio_max, asym * ninio_m) takes the branch the count vectors linearise,
      ninio_m for asym 1-5 and ninio_max for 6+). With it written and loaded,
      eval_structure on the same rows must equal counts . theta_labelled exactly in all
      four configurations. The generating set uses every reachable column, so any entry
      that ViennaRNA's reader drops, moves, or symmetrises fails some row, including entries
      that the r2 foundation check could not see because Turner 2004 gives them equal
      values. The labelled theta is a test device for the reader. It is not a model and is
      never folded.
  C1  must fail: the labelled theta written by a writer with two columns' positions swapped
      (planted defect) fails W4 on exactly the rows whose counts of the two columns differ.
  W5  restore: after every load, Turner 2004 is back (assert_turner2004_params).
  V1  the cache, measured: Turner 2004 compound with model A; load the labelled file;
      the next model-A compound still carries Turner 2004 (stale), a model-B compound
      carries the file (fresh). Recorded as found; loaded() and params_match handle it.
  V2  the clip, measured: the labelled theta with the four d2 tables POSITIVE; in both d2
      configurations eval_structure equals counts . clip(theta) on every row, and differs
      from counts . theta on the rows that use those tables.
  V3  the Boltzmann-factor cache, measured: after a load and an MFE-only sentinel, a
      model-A partition function still returns the Turner 2004 free energy while its MFE is
      the file's; after the full _rebuild both are the file's. Recorded as found.

Run: .venv/bin/python scripts/turner_par.py --check
"""

from __future__ import annotations

import contextlib
import functools
import json
import numbers
import os
import pathlib
import re
import sys
import tempfile
import time

import RNA

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_nullspace import (  # noqa: E402
    INF,
    Extractor,
    _reachable,
    _tok,
    dump_params,
    generating_set,
    theta_from_par,
)
from thermo.vienna_ref import assert_turner2004_params, reference_md  # noqa: E402

RUN_ID = "2026-10-12_turner_par_writer"
OUT = R / "runs" / RUN_ID
CONFIGS = [(0, 0), (0, 1), (2, 0), (2, 1)]
TOKEN = re.compile(r"\S+")


def _is_value(tok: str) -> bool:
    if tok in ("INF", "DEF"):
        return True
    try:
        float(tok)
    except ValueError:
        return False
    return True


def _value_positions(lines: list[str]) -> list[tuple[int, int]]:
    """(line, token) of every value token on a data line, in file order."""
    pos = []
    for li, raw in enumerate(lines):
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("/*"):
            continue
        for ti, tok in enumerate(s.split()):
            if _is_value(tok):
                pos.append((li, ti))
    return pos


def _replace_tokens(lines: list[str], new: dict[tuple[int, int], str]) -> list[str]:
    """Splice replacement tokens into their spans; everything else is kept verbatim."""
    out = list(lines)
    by_line: dict[int, dict[int, str]] = {}
    for (li, ti), t in new.items():
        by_line.setdefault(li, {})[ti] = t
    for li, repl in by_line.items():
        raw = lines[li]
        spans = [m.span() for m in TOKEN.finditer(raw)]
        parts, last = [], 0
        for ti, (a, b) in enumerate(spans):
            parts.append(raw[last:a])
            parts.append(repl[ti].rjust(b - a) if ti in repl else raw[a:b])
            last = b
        parts.append(raw[last:])
        out[li] = "".join(parts)
    return out


@functools.lru_cache(maxsize=1)
def template() -> dict:
    """The Turner 2004 template, its parsed theta, and the key -> (line, token) locator."""
    path = dump_params()
    text = path.read_text()
    lines = text.split("\n")
    pos = _value_positions(lines)
    probe_lines = _replace_tokens(lines, {p: str(i + 1) for i, p in enumerate(pos)})
    with tempfile.TemporaryDirectory() as t:
        probe = pathlib.Path(t) / "probe.par"
        probe.write_text("\n".join(probe_lines))
        ids = theta_from_par(probe)["theta"]
    P = theta_from_par(path)
    theta = P["theta"]
    assert set(ids) == set(theta), "probe and template parse to different keys"
    loc = {k: pos[ids[k] - 1] for k in theta}
    assert len(set(loc.values())) == len(loc), "two keys share one file position"
    for k, (li, ti) in loc.items():  # the located token is the parsed value
        assert _tok(lines[li].split()[ti]) == theta[k], k
    return {"text": text, "lines": lines, "theta": theta, "loc": loc,
            "special": P["special"], "lxc": P["lxc"]}


def round_to_format(theta: dict) -> tuple[dict, float]:
    """Round to the format's integer dcal/mol; return the rounded theta and the largest
    absolute rounding residual in dcal/mol."""
    out = {k: int(round(v)) for k, v in theta.items()}
    return out, max((abs(out[k] - v) for k, v in theta.items()), default=0.0)


def write_par_text(theta: dict, *, _loc=None) -> str:
    """The parameter file for `theta` (keys as theta_from_par; missing keys keep Turner
    2004). Values must be integers in dcal/mol. `_loc` exists only for the planted-defect
    control."""
    T = template()
    loc = T["loc"] if _loc is None else _loc
    unknown = set(theta) - set(loc)
    if unknown:
        raise KeyError(f"not a parameter-file column: {sorted(unknown, key=repr)[:5]}")
    new = {}
    for k, v in theta.items():
        if isinstance(v, numbers.Integral):
            v = int(v)
        elif isinstance(v, numbers.Real) and float(v).is_integer():
            v = int(v)
        elif isinstance(v, numbers.Real):
            raise ValueError(f"{k}: {v} is not an integer dcal/mol; use round_to_format")
        else:
            raise TypeError(f"{k}: {type(v).__name__}")
        li, ti = loc[k]
        old = T["lines"][li].split()[ti]
        if v == _tok(old):
            continue
        if old == "INF" or v >= INF:
            raise ValueError(f"{k}: refusing to replace {old} by {v}")
        new[(li, ti)] = str(v)
    return "\n".join(_replace_tokens(T["lines"], new))


def write_par(theta: dict, path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.write_text(write_par_text(theta))
    return path


D2_CLIPPED = ("mismatchExt", "mismatchM", "dangle5", "dangle3")


def _rebuild():
    """Defeat ViennaRNA's two last-model caches (energies and Boltzmann factors): a compound
    with a sentinel model (25 C, never used for a measurement), with an MFE and a partition
    function computed on it, makes the next measured compound rebuild both tables."""
    md = reference_md()
    md.temperature = 25.0
    fc = RNA.fold_compound("GGGGAAACCCC", md)
    fc.mfe()
    fc.pf()


def params_match(fc, theta: dict, dangles: int, special_hp: int) -> dict:
    """Mismatches between the table `fc` carries and `theta` (empty if none): twelve
    entries spanning the 1-D and 2-D tables, plus the model details that select the
    tables. (SWIG does not expose the 3-D tables for indexing; W4 covers those.)"""
    P = fc.params
    got = {("stack", 2, 2): P.stack[2][2], ("stack", 1, 2): P.stack[1][2],
           ("stack", 5, 6): P.stack[5][6], ("hairpin", 3): P.hairpin[3],
           ("hairpin", 4): P.hairpin[4], ("bulge", 1): P.bulge[1],
           ("internal_loop", 4): P.internal_loop[4], ("TerminalAU",): P.TerminalAU,
           ("ML_cc",): P.MLclosing, ("ML_cu",): P.MLbase, ("ML_ci",): P.MLintern[1],
           ("ninio_m",): P.ninio[2]}
    bad = {repr(k): (theta[k], v) for k, v in got.items() if theta[k] != v}
    md = P.model_details
    for f, want in (("dangles", dangles), ("special_hp", special_hp), ("temperature", 37.0)):
        if getattr(md, f) != want:
            bad[f] = (want, getattr(md, f))
    return bad


@contextlib.contextmanager
def loaded(path: pathlib.Path):
    """ViennaRNA's global table is the file at `path` inside the block; Turner 2004 after.
    Callers still check each configuration's compound with params_match."""
    if not RNA.params_load(str(path)):
        raise RuntimeError(f"ViennaRNA refused {path}")
    _rebuild()
    try:
        yield
    finally:
        RNA.params_load_RNA_Turner2004()
        _rebuild()
        assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))


def labelled_theta(clipped_sign: int = -1) -> dict:
    """Every column a distinct value: 1 + its rank, times `clipped_sign` in the four
    d2-clipped tables (-1 keeps them on the linear side; +1 is V2's probe of the clip). The
    ninio pair keeps its linearised branches."""
    keys = sorted(template()["theta"], key=repr)
    lab = {k: (clipped_sign if k[0] in D2_CLIPPED else 1) * (i + 1)
           for i, k in enumerate(keys)}
    a = len(keys) + 1
    lab[("ninio_m",)], lab[("ninio_max",)] = a, 5 * a
    return lab


def clip_d2(theta: dict) -> dict:
    return {k: min(v, 0) if k[0] in D2_CLIPPED else v for k, v in theta.items()}


# ----------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------
def _md(dangles, special_hp):
    md = reference_md()
    md.dangles, md.special_hp = dangles, special_hp
    assert md.dangles == dangles and md.special_hp == special_hp
    return md


def _energies(rows, dangles, special_hp, theta) -> list[int]:
    """eval_structure on every row, refusing if the compounds do not carry `theta`."""
    md = _md(dangles, special_hp)
    bad = params_match(RNA.fold_compound(rows[0][1], md), theta, dangles, special_hp)
    if bad:
        raise RuntimeError(f"d{dangles} special_hp{special_hp}: compound table is not the "
                           f"intended theta: {bad}")
    return [int(round(RNA.fold_compound(seq, md).eval_structure(db) * 100))
            for _, seq, db in rows]


def cache_probe(path: pathlib.Path) -> dict:
    """V1: is a file loaded after a model-A compound visible to the next model-A compound?"""
    RNA.params_load_RNA_Turner2004()
    _rebuild()
    a, b = _md(0, 0), _md(0, 1)
    before = RNA.fold_compound("GGGGAAACCCC", a).params.stack[2][2]
    RNA.params_load(str(path))
    same = RNA.fold_compound("ACGUACGUAAAAACGU", a).params.stack[2][2]
    other = RNA.fold_compound("GGGGAAACCCC", b).params.stack[2][2]
    RNA.params_load_RNA_Turner2004()
    _rebuild()
    assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))
    return {"turner_stack22_model_A": before, "after_load_model_A_again": same,
            "after_load_model_B": other, "file_stack22": theta_from_par(path)["theta"][
                ("stack", 2, 2)],
            "stale_on_same_model": same == before, "fresh_on_other_model": other != before}


def pf_cache_probe(path: pathlib.Path) -> dict:
    """V3: after a load, does a model-A partition function see the file when only an MFE
    compound with other model details was built in between (the old _rebuild), and when the
    sentinel also ran a partition function (the current _rebuild)?"""
    seq = "GGGAAACCCAGCUUCGGCUGGGAAACCC"
    a = _md(0, 0)
    sent = reference_md()
    sent.temperature = 25.0

    def g():
        fc = RNA.fold_compound(seq, a)
        _, e = fc.mfe()
        fc.exp_params_rescale(e)
        return round(e, 2), round(fc.pf()[1], 2)

    RNA.params_load_RNA_Turner2004()
    _rebuild()
    turner = g()
    RNA.params_load(str(path))
    RNA.fold_compound("GGGGAAACCCC", sent)  # MFE-only sentinel: the old _rebuild
    old = g()
    _rebuild()
    new = g()
    RNA.params_load_RNA_Turner2004()
    _rebuild()
    assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))
    return {"sequence": seq, "turner_mfe_and_G": turner,
            "after_load_mfe_only_rebuild": old, "after_load_full_rebuild": new,
            "boltzmann_stale_after_mfe_only_rebuild": old[1] == turner[1] and old[0] != turner[0],
            "fresh_after_full_rebuild": new[1] != turner[1]}


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def check() -> int:
    t0 = time.time()
    out = {"run_id": RUN_ID, "status": "running", "viennarna_version": RNA.__version__,
           **git_provenance(R)}
    write(out)
    T = template()
    theta0, lab = T["theta"], labelled_theta()
    out["template"] = {"n_lines": len(T["lines"]), "n_columns": len(theta0),
                       "n_value_tokens": len(_value_positions(T["lines"]))}

    w1 = write_par_text(theta0) == T["text"]
    rt0 = theta_from_par_text(write_par_text(theta0))
    rtl = theta_from_par_text(write_par_text(lab))
    out["W1_identity"] = w1
    out["W2_round_trip"] = {"turner2004": rt0 == theta0, "labelled": rtl == lab,
                            "labelled_values_distinct": len(set(lab.values())) == len(lab)}

    rows = generating_set(T["special"])
    out["rows"] = {"n": len(rows), "source": "turner_nullspace.generating_set (the r2 rows)"}
    _rebuild()
    stock = {c: _energies(rows, *c, theta0) for c in CONFIGS}
    assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))
    counts = {c: [Extractor(T["special"], *c).counts(seq, db) for _, seq, db in rows]
              for c in CONFIGS}

    def dot(th, c):
        return [sum(th[k] * v for k, v in cnt.items()) for cnt in counts[c]]

    def compare(e, want):
        return [i for i, (x, y) in enumerate(zip(e, want, strict=True)) if x != y]

    with tempfile.TemporaryDirectory() as t:
        tdir = pathlib.Path(t)
        w3 = {}
        with loaded(write_par(theta0, tdir / "turner2004_written.par")):
            for c in CONFIGS:
                bad = compare(_energies(rows, *c, theta0), stock[c])
                w3[f"d{c[0]}_special_hp{c[1]}"] = {"n_rows": len(rows), "n_mismatch": len(bad),
                                                   "passes": not bad}
        out["W3_written_turner2004_reproduces_stock"] = w3
        write(out)

        w4 = {}
        lab_path = write_par(lab, tdir / "labelled.par")
        with loaded(lab_path):
            for c in CONFIGS:
                e, want = _energies(rows, *c, lab), dot(lab, c)
                bad = compare(e, want)
                cols = {k for cnt in counts[c] for k in cnt}
                reach = {k for k in theta0 if _reachable(k, *c)}
                w4[f"d{c[0]}_special_hp{c[1]}"] = {
                    "n_rows": len(e), "n_mismatch": len(bad), "passes": not bad,
                    "n_reachable": len(reach), "n_reachable_used": len(reach & cols),
                    "first_mismatches": [{"seq": rows[i][1], "db": rows[i][2],
                                          "vienna": e[i], "counts_dot_theta": want[i]}
                                         for i in bad[:5]]}
        out["W4_labelled_reader_agrees_with_parser"] = w4
        write(out)

        # V2: the four d2 tables positive; ViennaRNA must equal counts . clip(theta)
        pos = labelled_theta(clipped_sign=+1)
        v2 = {}
        with loaded(write_par(pos, tdir / "labelled_positive.par")):
            for c in [cc for cc in CONFIGS if cc[0] == 2]:
                e = _energies(rows, *c, pos)
                uses = sum(any(k[0] in D2_CLIPPED for k in cnt) for cnt in counts[c])
                v2[f"d{c[0]}_special_hp{c[1]}"] = {
                    "n_rows": len(e), "n_rows_using_clipped_tables": uses,
                    "mismatch_vs_clip_model": len(compare(e, dot(clip_d2(pos), c))),
                    "mismatch_vs_linear_model": len(compare(e, dot(pos, c)))}
        out["V2_d2_clips_four_tables_at_zero"] = v2 | {
            "tables": list(D2_CLIPPED),
            "turner2004_max_in_those_tables": max(v for k, v in theta0.items()
                                                  if k[0] in D2_CLIPPED),
            "holds": all(v["mismatch_vs_clip_model"] == 0
                         and v["mismatch_vs_linear_model"] == v["n_rows_using_clipped_tables"]
                         > 0 for v in v2.values())}
        out["V1_table_cache_survives_params_load"] = cache_probe(lab_path)
        out["V3_boltzmann_cache_separate"] = pf_cache_probe(lab_path)
        write(out)

        # C1: a writer with two positions swapped must fail W4 on exactly the rows whose
        # counts of the two columns differ (d0, special_hp 0; two int22 entries)
        a, b = ("int22", 1, 1, 1, 1, 1, 2), ("int22", 1, 1, 1, 1, 2, 1)
        bad_loc = dict(T["loc"])
        bad_loc[a], bad_loc[b] = bad_loc[b], bad_loc[a]
        p = tdir / "planted.par"
        p.write_text(write_par_text(lab, _loc=bad_loc))
        c = (0, 0)
        with loaded(p):
            e = _energies(rows, *c, lab)
        fail = set(compare(e, dot(lab, c)))
        expect = {i for i, cnt in enumerate(counts[c]) if cnt.get(a, 0) != cnt.get(b, 0)}
        out["C1_planted_swap_detected"] = {"columns": [repr(a), repr(b)],
                                           "turner2004_values": [theta0[a], theta0[b]],
                                           "rows_expected_to_fail": len(expect),
                                           "rows_failing": len(fail),
                                           "detected": fail == expect and len(fail) > 0}
    assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))
    out["W5_turner2004_restored"] = True
    passes = (w1 and all(out["W2_round_trip"].values())
              and all(v["passes"] for v in w3.values())
              and all(v["passes"] for v in w4.values())
              and out["V2_d2_clips_four_tables_at_zero"]["holds"]
              and out["C1_planted_swap_detected"]["detected"])
    out["all_pass"] = passes
    out["status"] = "complete"
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    write(out)
    print(json.dumps({k: out[k] for k in ("W1_identity", "W2_round_trip",
                                          "W3_written_turner2004_reproduces_stock",
                                          "V1_table_cache_survives_params_load",
                                          "V3_boltzmann_cache_separate",
                                          "V2_d2_clips_four_tables_at_zero",
                                          "C1_planted_swap_detected", "all_pass")}
                     | {"W4": {k: {kk: v[kk] for kk in ("n_mismatch", "n_reachable",
                                                        "n_reachable_used")}
                               for k, v in w4.items()}}, indent=1))
    return 0 if passes else 1


def theta_from_par_text(text: str) -> dict:
    with tempfile.TemporaryDirectory() as t:
        p = pathlib.Path(t) / "x.par"
        p.write_text(text)
        return theta_from_par(p)["theta"]


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        sys.exit(check())
    raise SystemExit("usage: turner_par.py --check")
