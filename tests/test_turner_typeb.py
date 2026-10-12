"""scripts/turner_typeb.py: the orthogonal split and the controls of Experiment 1.

Toy bases built here (orthonormal blocks on a few labelled columns), never the registered
ones: these test the arithmetic, not the data.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.turner_typeb import decompose  # noqa: E402


def toy(seed=0):
    """12 columns: 9 used (int22 x5, int21 x4), 3 unused; intrinsic 1, nontrivial 2."""
    rng = np.random.default_rng(seed)
    cols = [("int22", i) for i in range(5)] + [("int21", i) for i in range(4)] + [
        ("hairpin", i) for i in range(3)]
    used, unused = np.arange(9), np.arange(9, 12)
    Q, _ = np.linalg.qr(rng.standard_normal((9, 9)))
    NI, NT, ROW = Q[:, :1], Q[:, 1:3], Q[:, 3:]
    A = sp.csr_matrix(ROW.T)  # rows span the identified complement, so A @ (NI, NT) = 0
    return {"cols": cols, "used": used, "unused": unused, "I": NI, "NT": NT, "A": A}, ROW


def test_planted_split_is_exact():
    B, ROW = toy()
    a, b, c, d = 2.0, 3.0, 0.5, 1.5
    delta = np.zeros(12)
    delta[B["used"]] = a * B["NT"][:, 0] + c * B["I"][:, 0] + d * ROW[:, 0]
    delta[9] = b
    r = decompose(delta, B, np.random.default_rng(1))
    nk = r["norm_kcal"]
    assert np.isclose(nk["nontrivial"], a) and np.isclose(nk["coverage"], b)
    assert np.isclose(nk["intrinsic"], c) and np.isclose(nk["identified"], d)
    assert np.isclose(sum(r["squared_fraction"].values()), 1.0)
    assert np.isclose(r["full_null_fraction"], np.sqrt(a * a + b * b + c * c)
                      / np.linalg.norm(delta))
    # metric-free: A sees only the identified part
    assert np.isclose(r["metric_free_max_abs_reference_energy_change_kcal"],
                      np.abs(B["A"] @ (d * ROW[:, 0])).max())


def test_null_only_delta_is_invisible_to_A():
    B, _ = toy(2)
    delta = np.zeros(12)
    delta[B["used"]] = B["NT"] @ np.array([1.0, -2.0])
    r = decompose(delta, B, np.random.default_rng(3))
    assert r["metric_free_max_abs_reference_energy_change_kcal"] < 1e-12
    assert np.isclose(r["fraction"]["nontrivial"], 1.0)


def test_beta_quantile_matches_sampled_random_subspaces():
    """Control (a)'s exact quantile: the squared projection of a fixed unit vector onto a
    uniformly random k-dim subspace of R^n is Beta(k/2, (n-k)/2)."""
    rng = np.random.default_rng(4)
    n, k, m = 40, 3, 20000
    x = np.zeros(n)
    x[0] = 1.0
    s = np.empty(m)
    for i in range(m):
        Q, _ = np.linalg.qr(rng.standard_normal((n, k)))
        s[i] = float(((Q.T @ x) ** 2).sum())
    for q in (0.5, 0.9, 0.99):
        assert abs(np.quantile(s, q) - stats.beta.ppf(q, k / 2, (n - k) / 2)) < 0.01


def test_table_matched_control_keeps_table_norms():
    """Control (c) on a delta that lives in one table, a basis that lives in the other: every
    draw projects to zero, and the true fraction is zero too."""
    B, _ = toy(5)
    NT = np.zeros((9, 2))
    NT[5, 0], NT[6, 1] = 1.0, 1.0  # int21 only
    B["NT"] = NT
    B["I"] = np.zeros((9, 1))
    B["I"][7, 0] = 1.0
    delta = np.zeros(12)
    delta[:5] = [1, 2, 3, 4, 5]  # int22 only
    r = decompose(delta, B, np.random.default_rng(6))
    c = r["control_c_table_matched"]
    assert c["max_fraction"] == 0.0 and r["fraction"]["nontrivial"] == 0.0


def test_fold_driver_status_is_read_as_running_by_the_standard_waiter(tmp_path):
    """The 2026-10-12 driver wrote 'fold RUNNING', which scripts/predecessor.py reads as a
    fault; the status must start with RUNNING_PREFIX."""
    import json

    from scripts.predecessor import RUNNING_PREFIX, state
    from scripts.turner_typeb import FOLD_RUNNING

    assert FOLD_RUNNING.startswith(RUNNING_PREFIX)
    art = tmp_path / "metrics.json"
    art.write_text(json.dumps({"status": FOLD_RUNNING}))
    done = {"fold complete (workers)"}
    assert state(art, done, lambda: [12345])["state"] == "running"
    assert state(art, done, lambda: [])["state"] == "fault"
    art.write_text(json.dumps({"status": "fold complete (workers)"}))
    assert state(art, done, lambda: [])["state"] == "done"
