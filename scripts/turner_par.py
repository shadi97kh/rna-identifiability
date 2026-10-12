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

PROCESS-GLOBAL STATE

ViennaRNA's parameter table is process-global (thermo/vienna_ref.py, hazard 3). loaded()
loads a written file for the duration of a with-block, then reloads Turner 2004 and checks,
on a fresh fold compound, that the Turner 2004 fingerprint is back.

CHECKS (--check; writes runs/<RUN_ID>/metrics.json)

  W1  identity: write_par(theta_2004) is the template, byte for byte.
  W2  parser round trip: theta_from_par(write_par(theta)) == theta exactly, for theta_2004
      and for the labelled theta below.
  W3  ViennaRNA's reader, unperturbed: with the written Turner 2004 file loaded,
      eval_structure on the 23,741 generating-set rows of
      runs/2026-10-07_turner_nullspace_r2 equals stock Turner 2004's energies exactly
      (integer dcal/mol), in all four configurations (dangles 0/2 x special_hp 0/1).
  W4  ViennaRNA's reader, labelled: a LABELLED theta gives every column a distinct value
      (1 + its rank by key; the ninio pair is set to (a, 5a) with a above every rank, so
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


@contextlib.contextmanager
def loaded(path: pathlib.Path):
    """ViennaRNA's global table is the file at `path` inside the block; Turner 2004 after."""
    if not RNA.params_load(str(path)):
        raise RuntimeError(f"ViennaRNA refused {path}")
    try:
        yield
    finally:
        RNA.params_load_RNA_Turner2004()
        assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))


def labelled_theta() -> dict:
    """Every column a distinct value; the ninio pair keeps its linearised branches."""
    keys = sorted(template()["theta"], key=repr)
    lab = {k: i + 1 for i, k in enumerate(keys)}
    a = len(keys) + 1
    lab[("ninio_m",)], lab[("ninio_max",)] = a, 5 * a
    return lab


# ----------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------
def _md(dangles, special_hp):
    md = reference_md()
    md.dangles, md.special_hp = dangles, special_hp
    assert md.dangles == dangles and md.special_hp == special_hp
    return md


def _energies(rows, dangles, special_hp) -> list[int]:
    md = _md(dangles, special_hp)
    return [int(round(RNA.fold_compound(seq, md).eval_structure(db) * 100))
            for _, seq, db in rows]


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
    stock = {c: _energies(rows, *c) for c in CONFIGS}
    assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))

    with tempfile.TemporaryDirectory() as t:
        tdir = pathlib.Path(t)
        w3 = {}
        with loaded(write_par(theta0, tdir / "turner2004_written.par")):
            for c in CONFIGS:
                e = _energies(rows, *c)
                bad = [i for i, (a, b) in enumerate(zip(e, stock[c], strict=True)) if a != b]
                w3[f"d{c[0]}_special_hp{c[1]}"] = {"n_rows": len(e), "n_mismatch": len(bad),
                                                   "passes": not bad}
        out["W3_written_turner2004_reproduces_stock"] = w3
        write(out)

        counts = {c: [Extractor(T["special"], *c).counts(seq, db) for _, seq, db in rows]
                  for c in CONFIGS}
        w4 = {}
        with loaded(write_par(lab, tdir / "labelled.par")):
            for c in CONFIGS:
                e = _energies(rows, *c)
                want = [sum(lab[k] * v for k, v in cnt.items()) for cnt in counts[c]]
                bad = [i for i, (a, b) in enumerate(zip(e, want, strict=True)) if a != b]
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

        # C1: a writer with two positions swapped must fail W4 on exactly the rows whose
        # counts of the two columns differ (d0, special_hp 0; columns from int22, the table
        # with the most equal Turner 2004 values)
        a, b = ("int22", 1, 1, 1, 1, 1, 2), ("int22", 1, 1, 1, 1, 2, 1)
        bad_loc = dict(T["loc"])
        bad_loc[a], bad_loc[b] = bad_loc[b], bad_loc[a]
        p = tdir / "planted.par"
        p.write_text(write_par_text(lab, _loc=bad_loc))
        c = (0, 0)
        with loaded(p):
            e = _energies(rows, *c)
        want = [sum(lab[k] * v for k, v in cnt.items()) for cnt in counts[c]]
        fail = {i for i, (x, y) in enumerate(zip(e, want, strict=True)) if x != y}
        expect = {i for i, cnt in enumerate(counts[c]) if cnt.get(a, 0) != cnt.get(b, 0)}
        out["C1_planted_swap_detected"] = {"columns": [repr(a), repr(b)],
                                           "turner2004_values": [theta0[a], theta0[b]],
                                           "rows_expected_to_fail": len(expect),
                                           "rows_failing": len(fail),
                                           "detected": fail == expect and len(fail) > 0}
    assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))
    out["W5_turner2004_restored"] = True
    passes = (w1 and all(out["W2_round_trip"].values())
              and all(v["passes"] for v in out["W3_written_turner2004_reproduces_stock"].values())
              and all(v["passes"] for v in out["W4_labelled_reader_agrees_with_parser"].values())
              and out["C1_planted_swap_detected"]["detected"])
    out["all_pass"] = passes
    out["status"] = "complete"
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    write(out)
    print(json.dumps({k: out[k] for k in ("W1_identity", "W2_round_trip",
                                          "W3_written_turner2004_reproduces_stock",
                                          "C1_planted_swap_detected", "all_pass")}
                     | {"W4": {k: {kk: v[kk] for kk in ("n_mismatch", "n_reachable",
                                                        "n_reachable_used")}
                               for k, v in out["W4_labelled_reader_agrees_with_parser"].items()}},
                     indent=1))
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
