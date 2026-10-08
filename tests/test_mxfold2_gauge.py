"""Unit tests for scripts/mxfold2_gauge.py on SYNTHETIC inputs only.

Real-data checks belong to Phase 0 (scripts/mxfold2_head_checks.py), after the
pre-registration. These run under stdlib unittest because .venv-mxfold2 has no pytest:
  .venv-mxfold2/bin/python -m unittest tests.test_mxfold2_gauge -v   (make test-mxfold2)
They skip cleanly where torch is unavailable.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import torch

    from scripts.mxfold2_counting import MX
    from scripts.mxfold2_gauge import (
        MAX_HELIX,
        Geometry,
        Masks,
        allowed_matrix,
        canonicalise,
        has_long_run,
        null_norm2,
    )
    HAVE = True
except ImportError:  # pragma: no cover
    HAVE = False


def synthetic_param(n, seed):
    """A positional param dict shaped like MXfold2 type C, filled with seeded noise."""
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(n + 1, n + 1, generator=g)
    M = torch.randn(n + 1, n + 1, generator=g)
    U = torch.randn(n + 1, n + 1, generator=g)
    p = {"score_helix_stacking": S, "score_helix_length": torch.randn(31, generator=g)}
    for k in ("score_mismatch_external", "score_mismatch_hairpin", "score_mismatch_internal",
              "score_mismatch_multi"):
        p[k] = M
    for k in ("score_base_hairpin", "score_base_internal", "score_base_multi",
              "score_base_external"):
        p[k] = U
    return p


def structure_score(p, seq, db):
    """Positional score of one structure by the counting rule (U and the length tables
    other than helix_length are untouched by the head, so they are left out)."""
    s = 0.0
    for k, c in MX("C").counts(seq, db).items():
        if k[0] == "S":
            s += c * float(p["score_helix_stacking"][k[1], k[2]])
        elif k[0] == "M":
            s += c * float(p["score_mismatch_hairpin"][k[1], k[2]])
        elif k[0] == "helix_length":
            s += c * float(p["score_helix_length"][k[1]])
    return s


SEQS = ["GGGGAAACCCC", "gcgcAAAAgcgc", "GCNGAAAYCGC", "ACGUACGUAGCUAGCUAGGAUCCGAUCGAUCGAUAAUGCGC",
        "GGGAAACUCCGAUAGCUAAGGCUUCGGCCUUAGC", "AU", "GGGGGAAAACCCCC"]
def random_structures(seq, k, seed):
    """k random VALID structures: nested, canonical pairs MXfold2 allows (j - i > 3)."""
    al = allowed_matrix(seq)
    cand = [(int(i), int(j)) for i, j in zip(*np.nonzero(al), strict=True)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        chosen = []
        for idx in rng.permutation(len(cand)):
            i, j = cand[idx]
            if all(i not in (a, b) and j not in (a, b) and not (a < i < b < j or i < a < j < b)
                   for a, b in chosen):
                chosen.append((i, j))
            if len(chosen) >= rng.integers(1, len(seq) // 3 + 2):
                break
        db = ["."] * len(seq)
        for i, j in chosen:
            db[i - 1], db[j - 1] = "(", ")"
        out.append("".join(db))
    out.append("." * len(seq))
    return out


STRUCT_SEQS = ["GGGAAACUCCGAUAGCUAAGGCUUCGGCCUUAGC", "GGGGAAACCCC",
               "ACGUACGUAGCUAGCUAGGAUCCGAUCGAUCGAUAAUGCGC"]


@unittest.skipUnless(HAVE, "torch / scripts not importable")
class TestGauge(unittest.TestCase):
    def test_masks_equal_geometry(self):
        for seq in SEQS:
            g, mk = Geometry(seq.upper()), Masks(seq)
            pairs = set(zip(g.I.tolist(), g.J.tolist(), strict=True))
            self.assertEqual(pairs, set(zip(*np.nonzero(mk.A.numpy()), strict=True)), seq)
            self.assertTrue(np.array_equal(g.e_in, mk.Ein.numpy()[g.I, g.J]))
            self.assertTrue(np.array_equal(g.e_out, mk.Eout.numpy()[g.I, g.J]))
            self.assertEqual(g.h_in_G, mk.h_in_G)

    def test_case_and_iupac(self):
        self.assertTrue(np.array_equal(allowed_matrix("gcgcAAAAgcgc"),
                                       allowed_matrix("GCGCAAAAGCGC")))
        al = allowed_matrix("GCNGAAAYCGC")
        self.assertFalse(al[3].any() or al[:, 3].any())  # N pairs with nothing

    def test_long_run_boundary(self):
        for L, want in ((MAX_HELIX, False), (MAX_HELIX + 1, True)):
            s = "G" * L + "AAAA" + "C" * L
            self.assertEqual(has_long_run(allowed_matrix(s)), want)
            self.assertEqual(Geometry(s).longest_helix > MAX_HELIX, want)

    def test_null_norm_matches_geometry(self):
        for n_, seq in enumerate(SEQS):
            p = synthetic_param(len(seq), n_)
            g, mk = Geometry(seq.upper()), Masks(seq)
            a = g.proj_norm2(g.theta(p))
            b = float(null_norm2(p, mk))
            self.assertAlmostEqual(a, b, delta=1e-10 * max(1.0, a))

    def test_canonicalise_is_idempotent_and_null_free(self):
        for n_, seq in enumerate(SEQS):
            p = synthetic_param(len(seq), 100 + n_)
            mk = Masks(seq)
            q = canonicalise(p, mk)
            self.assertLess(float(null_norm2(q, mk)), 1e-10 * max(1.0, float(null_norm2(p, mk))))
            q2 = canonicalise(q, mk)
            for k in ("score_helix_stacking", "score_mismatch_hairpin", "score_helix_length"):
                self.assertLess(float((q2[k] - q[k]).abs().max()), 1e-5)
            self.assertIs(q["score_mismatch_multi"], q["score_mismatch_hairpin"])
            self.assertIs(q["score_base_hairpin"], p["score_base_hairpin"])
            for k, v in q.items():
                self.assertEqual(v.dtype, torch.float32, k)

    def test_canonicalise_leaves_structure_scores(self):
        for n_, seq in enumerate(STRUCT_SEQS):
            p = synthetic_param(len(seq), 7 + n_)
            q = canonicalise(p, Masks(seq))
            for db in random_structures(seq, 25, n_):
                self.assertAlmostEqual(structure_score(p, seq, db), structure_score(q, seq, db),
                                       delta=1e-4)

    def test_without_h_leaves_a_null_residual(self):
        seq = "GGGAAACUCCGAUAGCUAAGGCUUCGGCCUUAGC"
        p = synthetic_param(len(seq), 11)
        mk = Masks(seq)
        full = float(null_norm2(canonicalise(p, mk), mk))
        no_h = float(null_norm2(canonicalise(p, mk, include_h=False), mk))
        self.assertGreater(no_h, 100 * max(full, 1e-12))


class TestLiveLauncherImports(unittest.TestCase):
    """The gauge study may import only pure functions from the live seed-study launcher, whose
    write/schedule/start_run/arch_conf are bound to the live study's metrics.json."""

    ALLOWED = {"recorded_seeds", "gpus", "compare_checkpoints"}

    def test_only_pure_imports(self):
        import ast
        root = pathlib.Path(__file__).resolve().parents[1] / "scripts"
        for name in ("mxfold2_gauge_study.py", "mxfold2_head_checks.py",
                     "mxfold2_lambda_pilot.py", "mxfold2_gauge_hooks.py",
                     "mxfold2_gauge_train.py", "mxfold2_gauge.py"):
            tree = ast.parse((root / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "scripts.mxfold2_seed_study":
                    got = {a.name for a in node.names}
                    self.assertTrue(got <= self.ALLOWED, f"{name} imports {got - self.ALLOWED}")
                if isinstance(node, ast.Import):
                    for a in node.names:
                        self.assertNotEqual(a.name, "scripts.mxfold2_seed_study", name)


if __name__ == "__main__":
    unittest.main()
