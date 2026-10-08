"""The exact gauge of MXfold2's positional outputs: one implementation for every consumer.

`Geometry` (numpy) is moved here VERBATIM from scripts/mxfold2_decompose.py (2026-10-08),
which now imports it, so the evaluator's numbers are unchanged by construction. The
torch head below computes the same quantities densely and differentiably, for the
post-hoc canonicaliser and the null-component regulariser.

G(x), for sequence x of length N (corrections c1-c3 of the seed study included):
  g_p, one per allowed pair p=(i,j) (canonical, j - i > 3):
      +1 on S_in(i,j) [if (i+1,j-1) allowed] and M_in(i,j);
      -1 on S_out(j,i) [if (i-1,j+1) allowed] and M_out(j,i);
  h:  +1 on every S entry, -2(m-1) on helix_length[m], m = 1..30, in G only if no
      structure of x can contain a helix longer than 30 pairs (c2);
  the projection orthogonalises h against the g_p (c1).
S, M are the stacking and (shared) mismatch channels, (N+1)x(N+1), upper triangle =
pair seen from inside, lower = from outside; helix_length is per sequence in the
param dict. The unpaired channel U is not in G and is never modified here.

Float32 rule: MXfold2's C++ DP reads every positional tensor as float32 through a
pointer into a temporary when the input is not float32 (src/param/util.h), so a
float64 tensor silently yields garbage. Everything handed back to MXfold2 is float32
and contiguous, and `assert_f32` enforces it.

The weight-space gauge v: +1 on the S-channel output bias, -2 on helix_length's
Linear weights k = 2..30. Moving the weights along v moves the outputs along h
exactly, so v.w is a gauge coordinate of the WEIGHTS (`v_dot`, `v_dot_grad`).

Note: Geometry (numpy, moved verbatim) is case-SENSITIVE and expects an uppercase
sequence (the evaluator passes read_bpseq's uppercased one); Masks is case-insensitive,
as MXfold2's C++ allow_paired is, so training-time sequences can be passed as they are.
"""

from __future__ import annotations

import numpy as np
import torch

MAX_HELIX = 30
CANON = {"AU", "UA", "GC", "CG", "GU", "UG"}
BIAS_KEY = "zuker.net.fc_paired.fc.4.bias"          # channel 0 = S
HELIX_W_KEY = "zuker.fc_length.score_helix_length.net.0.weight"  # (1, 31)

_CODE = np.full(256, 4, dtype=np.int8)
for _c, _k in zip("ACGU", range(4), strict=True):
    _CODE[ord(_c)] = _k
    _CODE[ord(_c.lower())] = _k
_PAIR = np.zeros((5, 5), dtype=bool)
for _p in CANON:
    _PAIR["ACGU".index(_p[0]), "ACGU".index(_p[1])] = True


class Geometry:
    """Index structures of theta(x) and G(x) for one sequence."""

    def __init__(self, seq):
        n = len(seq)
        self.n = n
        al = np.zeros((n + 2, n + 2), bool)
        for i in range(1, n + 1):
            for j in range(i + 4, n + 1):
                al[i, j] = seq[i - 1] + seq[j - 1] in CANON
        self.I, self.J = np.nonzero(al)
        self.e_in = al[self.I + 1, self.J - 1]
        self.e_out = al[self.I - 1, self.J + 1]
        self.n_p = 2 + self.e_in.astype(int) + self.e_out.astype(int)
        self.hg = self.e_in.astype(int) - self.e_out.astype(int)
        self.ua, self.ub = np.triu_indices(n + 1, 0)
        keep = self.ua >= 1
        self.ua, self.ub = self.ua[keep], self.ub[keep]
        self.la = np.arange(2, n + 1)  # empty-interval entries U(a, a-1), correction c3
        self.m = np.arange(1, MAX_HELIX + 1)
        self.hw = -2.0 * (self.m - 1)
        # longest run of stacked allowed pairs, along anti-diagonals
        longest = 0
        for s in range(2, 2 * n + 1):
            i = np.arange(max(1, s - n), min(n, s - 1) + 1)
            v = al[i, s - i]
            if v.any():
                d = np.diff(np.concatenate([[0], v.astype(int), [0]]))
                longest = max(longest, int((np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]).max()))
        self.longest_helix = longest
        self.h_in_G = longest <= MAX_HELIX
        self.hh = int(self.e_in.sum() + self.e_out.sum()) + float((self.hw ** 2).sum())

    def theta(self, p):
        S = p["score_helix_stacking"].numpy().astype(np.float64)
        M = p["score_mismatch_hairpin"].numpy().astype(np.float64)
        U = p["score_base_hairpin"].numpy().astype(np.float64)
        hl = p["score_helix_length"].numpy().astype(np.float64)
        assert S.shape == (self.n + 1, self.n + 1), S.shape
        return {"Sin": np.where(self.e_in, S[self.I, self.J], 0.0),
                "Sout": np.where(self.e_out, S[self.J, self.I], 0.0),
                "Min": M[self.I, self.J], "Mout": M[self.J, self.I],
                "U": U[self.ua, self.ub], "Ulow": U[self.la, self.la - 1],
                "HL": hl[1:MAX_HELIX + 1]}

    def norm2(self, d):
        return float(sum((v ** 2).sum() for v in d.values()))

    def proj_norm2(self, d):
        dg = d["Sin"] + d["Min"] - d["Sout"] - d["Mout"]
        val = float((dg ** 2 / self.n_p).sum())
        if self.h_in_G:
            dh = d["Sin"].sum() + d["Sout"].sum() + float((self.hw * d["HL"]).sum())
            dhp = dh - float((self.hg * dg / self.n_p).sum())
            hhp = self.hh - float((self.hg ** 2 / self.n_p).sum())
            val += dhp ** 2 / hhp
        return val

    def random_in_G(self, rng):
        c = rng.standard_normal(len(self.I))
        g = {"Sin": c * self.e_in, "Min": c.copy(), "Sout": -c * self.e_out, "Mout": -c.copy(),
             "U": np.zeros(len(self.ua)), "Ulow": np.zeros(len(self.la)),
             "HL": np.zeros(MAX_HELIX)}
        if self.h_in_G:
            ch = rng.standard_normal()
            g["Sin"] = g["Sin"] + ch * self.e_in
            g["Sout"] = g["Sout"] + ch * self.e_out
            g["HL"] = ch * self.hw
        return g

    def project(self, d):
        """P_G d, as a theta-shaped dict."""
        dg = d["Sin"] + d["Min"] - d["Sout"] - d["Mout"]
        c = dg / self.n_p
        out = {"Sin": c * self.e_in, "Min": c.copy(), "Sout": -c * self.e_out, "Mout": -c.copy(),
               "U": np.zeros(len(self.ua)), "Ulow": np.zeros(len(self.la)),
               "HL": np.zeros(MAX_HELIX)}
        if self.h_in_G:
            dh = d["Sin"].sum() + d["Sout"].sum() + float((self.hw * d["HL"]).sum())
            dhp = dh - float((self.hg * c).sum())
            hhp = self.hh - float((self.hg ** 2 / self.n_p).sum())
            a = dhp / hhp
            # h' = h - sum_p (h.g_p / n_p) g_p
            ap = -a * self.hg / self.n_p
            out["Sin"] += (a + ap) * self.e_in
            out["Sout"] += (a - ap) * self.e_out
            out["Min"] += ap
            out["Mout"] -= ap
            out["HL"] = a * self.hw
        return out

    def apply(self, p, d):
        """Positional parameter dict with theta-direction d added (S, M, U, helix_length)."""
        q = dict(p)
        S = p["score_helix_stacking"].clone()
        M = p["score_mismatch_hairpin"].clone()
        U = p["score_base_hairpin"].clone()
        hl = p["score_helix_length"].clone()
        S[self.I, self.J] += torch.tensor(d["Sin"] * self.e_in, dtype=S.dtype)
        S[self.J, self.I] += torch.tensor(d["Sout"] * self.e_out, dtype=S.dtype)
        M[self.I, self.J] += torch.tensor(d["Min"], dtype=M.dtype)
        M[self.J, self.I] += torch.tensor(d["Mout"], dtype=M.dtype)
        U[self.ua, self.ub] += torch.tensor(d["U"], dtype=U.dtype)
        U[self.la, self.la - 1] += torch.tensor(d["Ulow"], dtype=U.dtype)
        hl[1:MAX_HELIX + 1] += torch.tensor(d["HL"], dtype=hl.dtype)
        q["score_helix_stacking"] = S
        for k in ("score_mismatch_external", "score_mismatch_hairpin",
                  "score_mismatch_internal", "score_mismatch_multi"):
            q[k] = M
        for k in ("score_base_hairpin", "score_base_internal", "score_base_multi",
                  "score_base_external"):
            q[k] = U
        q["score_helix_length"] = hl
        return q


# ----------------------------------------------------------------------------------
# vectorised masks (numpy) and the dense torch head
# ----------------------------------------------------------------------------------
def allowed_matrix(seq: str) -> np.ndarray:
    """(n+2)x(n+2) bool: allowed[i, j] for 1 <= i < j <= n, canonical and j - i > 3.

    Case-insensitive; T and IUPAC codes do not pair. Equal to Geometry's `al`."""
    n = len(seq)
    code = _CODE[np.frombuffer(seq.encode("latin-1"), dtype=np.uint8)]
    al = np.zeros((n + 2, n + 2), dtype=bool)
    pm = _PAIR[code[:, None], code[None, :]]
    i, j = np.indices((n, n))
    al[1:n + 1, 1:n + 1] = pm & (j - i > 3)
    return al


def has_long_run(al: np.ndarray) -> bool:
    """True if the allowed pairs admit 31 or more consecutive stacked pairs."""
    run = al.copy()
    m = al.shape[0]
    for k in range(1, MAX_HELIX + 1):
        nxt = np.zeros_like(run)
        nxt[: m - k, k:] = al[k:, : m - k]  # al[i + k, j - k] placed at [i, j]
        run &= nxt
        if not run.any():
            return False
    return bool(run.any())


class Masks:
    """Per-sequence masks of G(x) as dense tensors on `device` (float64 where numeric)."""

    def __init__(self, seq: str, device="cpu"):
        al = allowed_matrix(seq)
        n = len(seq)
        ein = np.zeros_like(al)
        eout = np.zeros_like(al)
        ein[: n + 1, 1:] = al[: n + 1, 1:] & al[1:, : n + 1]          # al[i+1, j-1]
        eout[1:, : n + 1] = al[1:, : n + 1] & al[: n + 1, 1:]          # al[i-1, j+1]
        A = al[: n + 1, : n + 1]
        ein, eout = ein[: n + 1, : n + 1] & A, eout[: n + 1, : n + 1] & A
        t = lambda x, dt=torch.float64: torch.as_tensor(x, dtype=dt, device=device)  # noqa: E731
        self.n = n
        self.A = t(A, torch.bool)
        self.Ein = t(ein, torch.bool)
        self.Eout = t(eout, torch.bool)
        np_ = 2.0 + ein + eout.astype(float)
        np_[~A] = 1.0
        hg = (ein.astype(float) - eout) * A
        self.NP = t(np_)
        self.HG = t(hg)
        self.hw = t(-2.0 * (np.arange(1, MAX_HELIX + 1) - 1))
        self.h_in_G = not has_long_run(al)
        self.n_pairs = int(A.sum())
        self.dimG = self.n_pairs + int(self.h_in_G)
        self.hhp = float(ein.sum() + eout.sum() + (self.hw.cpu().numpy() ** 2).sum()
                         - (hg[A] ** 2 / np_[A]).sum())


def _parts(p: dict, mk: Masks):
    S = p["score_helix_stacking"].double()
    M = p["score_mismatch_hairpin"].double()
    hl = p["score_helix_length"].double()
    zero = S.new_zeros(())
    Sin = torch.where(mk.Ein, S, zero)
    Sout = torch.where(mk.Eout, S.transpose(0, 1), zero)
    D = torch.where(mk.A, Sin + M - Sout - M.transpose(0, 1), zero)
    dh = Sin.sum() + Sout.sum() + (mk.hw * hl[1:MAX_HELIX + 1]).sum()
    dhp = dh - (mk.HG * D / mk.NP).sum()
    return S, M, hl, D, dhp


def null_norm2(p: dict, mk: Masks, include_h: bool = True) -> torch.Tensor:
    """||P_G theta||^2 (float64, differentiable). Equals Geometry.proj_norm2."""
    _, _, _, D, dhp = _parts(p, mk)
    val = (D * D / mk.NP).sum()
    if include_h and mk.h_in_G:
        val = val + dhp * dhp / mk.hhp
    return val


def null_ratio(p: dict, mk: Masks) -> torch.Tensor:
    """R = ||P_G theta||^2 / dim G, the mean squared gauge coordinate (pre-registered)."""
    return null_norm2(p, mk) / mk.dimG


def canonicalise(p: dict, mk: Masks, include_h: bool = True) -> dict:
    """theta - P_G theta, returned as a positional param dict of contiguous float32.

    Differentiable. The four mismatch keys alias one tensor and the four base keys
    keep the unpaired channel, as MXfold2 type C builds them. include_h=False is the
    NC2 negative control only."""
    S, M, hl, D, dhp = _parts(p, mk)
    zero = S.new_zeros(())
    C = D / mk.NP
    use_h = include_h and mk.h_in_G
    a = dhp / mk.hhp if use_h else S.new_zeros(())
    AP = torch.where(mk.A, -a * mk.HG / mk.NP, zero)
    up_S = torch.where(mk.Ein, C + a + AP, zero)
    lo_S = torch.where(mk.Eout, -C + a - AP, zero).transpose(0, 1)
    up_M = torch.where(mk.A, C + AP, zero)
    lo_M = torch.where(mk.A, -C - AP, zero).transpose(0, 1)
    S2 = (S - up_S - lo_S).float().contiguous()
    M2 = (M - up_M - lo_M).float().contiguous()
    hl2 = torch.cat([hl[:1], hl[1:MAX_HELIX + 1] - a * mk.hw, hl[MAX_HELIX + 1:]])
    q = dict(p)
    q["score_helix_stacking"] = S2
    for k in ("score_mismatch_external", "score_mismatch_hairpin",
              "score_mismatch_internal", "score_mismatch_multi"):
        q[k] = M2
    q["score_helix_length"] = hl2.float().contiguous()
    assert_f32(q)
    return q


def assert_f32(p: dict) -> None:
    for k, v in p.items():
        if torch.is_tensor(v) and k.startswith("score_"):
            assert v.dtype == torch.float32, f"{k} is {v.dtype}; MXfold2's DP reads float32"


def v_dot(sd: dict) -> float:
    """v.w = b_S - 2 * sum_{k>=2} W_k, from a state dict (float64)."""
    b = sd[BIAS_KEY].double()
    w = sd[HELIX_W_KEY].double()
    return float(b[0] - 2.0 * w[0, 2:MAX_HELIX + 1].sum())


def v_norm2() -> float:
    return 1.0 + 4.0 * (MAX_HELIX - 1)


def v_dot_grad(model) -> float | None:
    """v.grad for a MixedFold model, or None if the gradients are absent."""
    named = dict(model.named_parameters())
    b, w = named.get(BIAS_KEY), named.get(HELIX_W_KEY)
    if b is None or w is None or b.grad is None or w.grad is None:
        return None
    return float(b.grad.double()[0] - 2.0 * w.grad.double()[0, 2:MAX_HELIX + 1].sum())
