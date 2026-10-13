"""Unit tests for scripts/mxfold2_vproject.py (A3's parameter projection) on SYNTHETIC inputs.

stdlib unittest, as tests/test_mxfold2_gauge.py, because .venv-mxfold2 has no pytest:
  CUDA_VISIBLE_DEVICES= .venv-mxfold2/bin/python -m unittest tests.test_mxfold2_vproject -v
(make test-mxfold2). Under .venv's pytest the synthetic tests run and the MixC-model test skips
(mxfold2 is installed only in .venv-mxfold2).

What is checked, over 10^4 seeded float32 draws of the two tensors (bias (3,), weight (1, 31),
the shapes of MixC's zuker.net.fc_paired.fc.4.bias and score_helix_length's Linear weight),
magnitudes 1e-3..1e3 in four families (one scale per draw, a scale per entry, w nearly along
v, w exactly along v):
  * after project_v_, |v.w| <= v_bound(b, W, c) (the rigorous float32 bound);
  * only bias[0] and weight[0, 2:31] change, and each changes by -c / +2c up to its own
    float32 rounding;
  * projecting a second time changes nothing beyond the first bound: every entry moves by at
    most the shift the bound allows (|c2| <= bound/117) plus its own rounding, and the second
    result is again within its bound;
  * the bound is not vacuous: without a projection (c = 0) it rejects nearly every draw;
  * vw64 equals scripts/mxfold2_gauge.v_dot bit for bit on a CPU MixC model, and projecting
    that model's two parameters leaves every other tensor of its state dict untouched.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import torch

    from scripts.mxfold2_gauge import BIAS_KEY, HELIX_W_KEY, v_dot
    from scripts.mxfold2_vproject import K_HI, K_LO, V_NORM2, project_v_, v_bound, vw64
    HAVE = True
except ImportError:  # pragma: no cover
    HAVE = False

try:
    import mxfold2.train  # noqa: F401
    HAVE_MX = True
except ImportError:
    HAVE_MX = False

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_DRAWS = 10_000
SEED = 20261012
U32 = 2.0 ** -24


def draws(n=N_DRAWS, seed=SEED):
    """n (bias, weight, family) float32 pairs; four families in turn, magnitudes 1e-3..1e3."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    v_b = np.zeros(3)
    v_b[0] = 1.0
    v_w = np.zeros((1, 31))
    v_w[0, K_LO:K_HI] = -2.0
    out = []
    for i in range(n):
        fam = i % 4
        if fam == 0:      # one scale per draw
            s = 10.0 ** rng.uniform(-3, 3)
            b, w = s * rng.standard_normal(3), s * rng.standard_normal((1, 31))
        elif fam == 1:    # a scale per entry: mixed magnitudes, cancellation in the sum
            b = rng.choice([-1, 1], 3) * 10.0 ** rng.uniform(-3, 3, 3)
            w = rng.choice([-1, 1], (1, 31)) * 10.0 ** rng.uniform(-3, 3, (1, 31))
        elif fam == 2:    # nearly along v: |c| large against the post-projection magnitudes
            a = rng.choice([-1, 1]) * 10.0 ** rng.uniform(-3, 3)
            e = a * 10.0 ** rng.uniform(-8, -1)
            b = a * v_b + e * rng.standard_normal(3)
            w = a * v_w + e * rng.standard_normal((1, 31))
        else:             # exactly along v (the float32 values of a * v)
            a = rng.choice([-1, 1]) * 10.0 ** rng.uniform(-3, 3)
            b, w = a * v_b, a * v_w
        out.append((torch.tensor(b, dtype=torch.float32), torch.tensor(w, dtype=torch.float32),
                    fam))
    return out


def ulp_rel(t):
    return U32 * (1.0 + 2.0 ** -23) * t.double().abs() + 2.0 ** -149


@unittest.skipUnless(HAVE, "torch / scripts not importable")
class TestProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = draws()

    def test_bound_holds_and_only_30_entries_move(self):
        mask_w = torch.zeros(1, 31, dtype=torch.bool)
        mask_w[0, K_LO:K_HI] = True
        worst = 0.0
        for b0, w0, fam in self.D:
            b, w = b0.clone(), w0.clone()
            r = project_v_(b, w)
            self.assertTrue(r["written"])
            x, bd = vw64(b, w), v_bound(b, w, r["c"])
            self.assertLessEqual(abs(x), bd, (fam, r, x, bd))
            worst = max(worst, abs(x) / bd)
            # untouched entries, bit for bit
            self.assertTrue(torch.equal(b[1:], b0[1:]))
            self.assertTrue(torch.equal(w[~mask_w], w0[~mask_w]))
            # the 30 entries moved by -c / +2c, each up to its own rounding
            c = r["c"]
            db = (b0[0].double() - c) - b[0].double()
            dw = (w0[0, K_LO:K_HI].double() + 2.0 * c) - w[0, K_LO:K_HI].double()
            self.assertLessEqual(abs(float(db)), float(ulp_rel(b[0])) + 2.0 ** -52 * abs(c)
                                 + 2.0 ** -52 * abs(float(b0[0])))
            tol_w = ulp_rel(w[0, K_LO:K_HI]) + 2.0 ** -51 * abs(c) \
                + 2.0 ** -52 * w0[0, K_LO:K_HI].double().abs()
            self.assertTrue(bool((dw.abs() <= tol_w).all()), fam)
            self.assertEqual(b.dtype, torch.float32)
            self.assertEqual(w.dtype, torch.float32)
        self.assertLess(worst, 1.0)

    def test_second_projection_changes_nothing_beyond_the_bound(self):
        n_exact = 0
        for b0, w0, fam in self.D:
            b, w = b0.clone(), w0.clone()
            r1 = project_v_(b, w)
            bd1 = v_bound(b, w, r1["c"])
            b1, w1 = b.clone(), w.clone()
            r2 = project_v_(b, w)
            self.assertLessEqual(abs(r2["c"]), bd1 / V_NORM2 * (1 + 1e-12), fam)
            self.assertLessEqual(abs(vw64(b, w)), v_bound(b, w, r2["c"]), fam)
            lim_b = bd1 / V_NORM2 * (1 + 1e-12) + ulp_rel(b[0]) + 2.0 ** -52 * b1[0].double().abs()
            lim_w = 2 * bd1 / V_NORM2 * (1 + 1e-12) + ulp_rel(w[0, K_LO:K_HI]) \
                + 2.0 ** -52 * w1[0, K_LO:K_HI].double().abs()
            self.assertLessEqual(float((b[0].double() - b1[0].double()).abs()), float(lim_b))
            self.assertTrue(bool(((w[0, K_LO:K_HI].double() - w1[0, K_LO:K_HI].double()).abs()
                                  <= lim_w).all()), fam)
            self.assertTrue(torch.equal(b[1:], b1[1:]))
            self.assertTrue(torch.equal(w[:, :K_LO], w1[:, :K_LO]))
            self.assertTrue(torch.equal(w[:, K_HI:], w1[:, K_HI:]))
            n_exact += int(torch.equal(b, b1) and torch.equal(w, w1))
        # reported by -v: how often the second projection is exactly a no-op
        print(f"\n  second projection bitwise no-op on {n_exact} of {len(self.D)} draws")

    def test_bound_is_not_vacuous(self):
        """Without a projection (c = 0) the bound rejects the unprojected weights."""
        caught = sum(int(abs(vw64(b, w)) > v_bound(b, w, 0.0)) for b, w, fam in self.D
                     if fam in (0, 1))
        self.assertGreaterEqual(caught, 0.99 * N_DRAWS / 2)

    def test_planted_step_is_caught(self):
        """An Adam-sized step along v after a projection (relative 1e-3) exceeds the bound."""
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 4]))
        for b0, w0, fam in self.D[:2000]:
            if fam not in (0, 1):
                continue
            b, w = b0.clone(), w0.clone()
            project_v_(b, w)
            step = 1e-3 * max(float(w.abs().max()), float(b.abs().max()))
            k = int(rng.integers(K_LO, K_HI))
            w[0, k] += step
            self.assertGreater(abs(vw64(b, w)), v_bound(b, w, 0.0), (fam, step))

    def test_exact_zero_and_non_finite(self):
        b, w = torch.zeros(3), torch.zeros(1, 31)
        r = project_v_(b, w)
        self.assertEqual((r["pre"], r["c"]), (0.0, 0.0))
        self.assertTrue(torch.equal(b, torch.zeros(3)) and torch.equal(w, torch.zeros(1, 31)))
        b = torch.tensor([float("nan"), 1.0, 2.0])
        w = torch.ones(1, 31)
        r = project_v_(b, w)
        self.assertFalse(r["written"])
        self.assertTrue(torch.equal(w, torch.ones(1, 31)))  # nothing written
        self.assertFalse(v_bound(b, w, r["c"]) >= abs(vw64(b, w)))  # NaN never passes

    def test_rejects_float64(self):
        with self.assertRaises(AssertionError):
            project_v_(torch.zeros(3, dtype=torch.float64), torch.zeros(1, 31, dtype=torch.float64))

    def test_vw64_is_v_dot_on_tensors(self):
        for b, w, _ in self.D[:500]:
            self.assertEqual(vw64(b, w), v_dot({BIAS_KEY: b, HELIX_W_KEY: w}))


def arch_file(dst: pathlib.Path) -> pathlib.Path:
    """MXfold2's TrainSetAB.conf without its --param line (the studies' arch.conf)."""
    conf = ROOT / "vendor" / "mxfold2" / "mxfold2" / "models" / "TrainSetAB.conf"
    keep, skip = [], False
    for ln in conf.read_text().split("\n"):
        if skip:
            skip = False
            continue
        if ln.strip() == "--param":
            skip = True
            continue
        if ln.strip():
            keep.append(ln)
    dst.write_text("\n".join(keep) + "\n")
    return dst


@unittest.skipUnless(HAVE and HAVE_MX, "mxfold2 not importable (it lives in .venv-mxfold2)")
class TestOnMixC(unittest.TestCase):
    def test_model_v_dot_and_untouched_tensors(self):
        import random

        from scripts.mxfold2_decompose import build_model
        with tempfile.TemporaryDirectory() as td:
            arch = arch_file(pathlib.Path(td) / "arch.conf")
            for seed in (20261006, 20261007, 20261008):
                torch.manual_seed(seed)
                random.seed(seed)
                m = build_model(arch)
                named = dict(m.named_parameters())
                b, w = named[BIAS_KEY], named[HELIX_W_KEY]
                self.assertEqual(vw64(b, w), v_dot(m.state_dict()))
                before = {k: v.clone() for k, v in m.state_dict().items()}
                r = project_v_(b, w)
                self.assertEqual(r["pre"], v_dot(before))
                after = m.state_dict()
                self.assertLessEqual(abs(v_dot(after)), v_bound(b, w, r["c"]))
                self.assertEqual(vw64(b, w), v_dot(after))
                for k, v in after.items():
                    if k not in (BIAS_KEY, HELIX_W_KEY):
                        self.assertTrue(torch.equal(v, before[k]), k)
                self.assertTrue(torch.equal(after[BIAS_KEY][1:], before[BIAS_KEY][1:]))
                self.assertGreater(abs(r["pre"]), 1e-3)  # the projection did something


if __name__ == "__main__":
    unittest.main()
