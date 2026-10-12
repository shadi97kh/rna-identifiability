"""scripts/turner_par.py: the ViennaRNA parameter-file writer is the inverse of
theta_from_par, and ViennaRNA reads what it writes.

Fast tests check the format level (identity, round trip, refusals, rounding). The slow
test is the reader check of `turner_par.py --check` without writing an artifact: on the
23,741 generating-set rows, the written Turner 2004 file reproduces stock energies and a
labelled file reproduces counts . theta, in all four configurations.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.turner_nullspace import Extractor, generating_set  # noqa: E402
from scripts.turner_par import (  # noqa: E402
    CONFIGS,
    _energies,
    labelled_theta,
    loaded,
    round_to_format,
    template,
    theta_from_par_text,
    write_par,
    write_par_text,
)


def test_unperturbed_write_is_the_template():
    T = template()
    assert write_par_text(T["theta"]) == T["text"]


def test_round_trip_labelled():
    lab = labelled_theta()
    assert len(set(lab.values())) == len(lab)
    assert theta_from_par_text(write_par_text(lab)) == lab


def test_locator_is_injective_and_complete():
    T = template()
    assert set(T["loc"]) == set(T["theta"])
    assert len(set(T["loc"].values())) == len(T["loc"])


def test_one_entry_changes_one_token():
    T = template()
    th = dict(T["theta"])
    th[("int21", 2, 3, 1, 4, 2)] += 7
    old, new = T["text"].split("\n"), write_par_text(th).split("\n")
    changed = [i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b]
    assert len(changed) == 1
    diff = [(a, b) for a, b in zip(old[changed[0]].split(), new[changed[0]].split(),
                                   strict=True) if a != b]
    assert len(diff) == 1 and int(diff[0][1]) - int(diff[0][0]) == 7


@pytest.mark.parametrize("bad", [{("stack", 1, 1): -240.5}, {("hairpin", 3): 10**7},
                                 {("no_such_table", 1): 0}, {("stack", 1, 1): "x"}])
def test_refusals(bad):
    with pytest.raises((ValueError, KeyError, TypeError)):
        write_par_text(bad)


def test_round_to_format_reports_residual():
    r, res = round_to_format({("stack", 1, 1): -240.4, ("stack", 1, 2): -329.5})
    assert r == {("stack", 1, 1): -240, ("stack", 1, 2): -330} and res == 0.5
    assert theta_from_par_text(write_par_text(r))[("stack", 1, 2)] == -330


@pytest.mark.slow
def test_viennarna_reads_what_is_written():
    T = template()
    theta0, lab = T["theta"], labelled_theta()
    rows = generating_set(T["special"])
    stock = {c: _energies(rows, *c, theta0) for c in CONFIGS}
    with tempfile.TemporaryDirectory() as t:
        tdir = pathlib.Path(t)
        with loaded(write_par(theta0, tdir / "t.par")):
            for c in CONFIGS:
                assert _energies(rows, *c, theta0) == stock[c], c
        with loaded(write_par(lab, tdir / "lab.par")):
            for c in CONFIGS:
                ex = Extractor(T["special"], *c)
                want = [sum(lab[k] * v for k, v in ex.counts(s, d).items()) for _, s, d in rows]
                assert _energies(rows, *c, lab) == want, c
