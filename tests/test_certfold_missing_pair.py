"""An admissible pair missing from the Zuker set is an error, not a certified-absent label.

Every admissible pair is in some structure (the pair alone, closing a hairpin), so a pair the
Zuker set misses can only mean the routine failed to backtrack. Until 2026-10-21 certify()
labeled such a pair certified absent without recording it; it now raises. certify_sequence, which
labels the test sets, did the same until 2026-10-27; it now takes its absent margins through
absent_margins(), which raises. These tests plant the case with a stand-in for the constrained
compound, so they need no folding.
"""

import inspect
import pathlib
import sys

import pytest

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold as cf  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

SEQ = "GGGGAAACCCCAAAAGGGAAACCC"


class _Folder:
    """What certify() reads from a Folder: E'(s*), the forbidden-pair fold and the Zuker set."""

    def __init__(self, zuker_structures):
        self.eprime_star = -500
        self.folds = 0
        self.W = type("W", (), {"r": 1.0})()
        self._zuker = zuker_structures

    def without(self, i, j):
        return "." * len(SEQ), -400, -400

    def zuker(self):
        return list(self._zuker)


def _single_pair(q):
    db = ["."] * len(SEQ)
    db[q[0] - 1], db[q[1] - 1] = "(", ")"
    return "".join(db)


def test_a_complete_zuker_set_labels_every_admissible_pair():
    adm = cf.admissible_pairs(SEQ)
    c = cf.certify(_Folder([(_single_pair(q), -100) for q in adm]), [adm[0]], adm)
    assert c["n_absent_because_unreachable"] == 0
    assert len(c["margins_absent_dcal"]) == len(adm) - 1


def test_a_missing_admissible_pair_is_an_error():
    adm = cf.admissible_pairs(SEQ)
    planted = [(_single_pair(q), -100) for q in adm[:-1]]          # the last pair goes missing
    with pytest.raises(GuardFailure, match="missing from the Zuker set"):
        cf.certify(_Folder(planted), [adm[0]], adm)


def test_absent_margins_cover_every_admissible_pair():
    adm = cf.admissible_pairs(SEQ)
    zmap = {q: -100 for q in adm}
    assert cf.absent_margins(zmap, adm[1:], -500) == {q: 400 for q in adm[1:]}


def test_absent_margins_refuse_a_missing_pair():
    adm = cf.admissible_pairs(SEQ)
    zmap = {q: -100 for q in adm[:-1]}                              # the last pair goes missing
    with pytest.raises(GuardFailure, match="missing from the Zuker set"):
        cf.absent_margins(zmap, adm, -500)


def test_certify_sequence_reads_the_zuker_set_through_the_guard():
    """certify_sequence labels the test sets, so its absent margins must come from
    absent_margins(), which raises on a missing pair, and never from the Zuker map directly."""
    src = inspect.getsource(cf.certify_sequence)
    assert "absent_margins(zmap, outside" in src
    assert "zmap.get(q)" not in src and "m is None" not in src
