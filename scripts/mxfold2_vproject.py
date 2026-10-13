"""Projection of MXfold2's weights off the weight-space gauge direction v, with a float32 bound.

THE DIRECTION (scripts/mxfold2_gauge.py, unchanged): v = +1 on BIAS_KEY[0] (the S channel of
the paired output layer), -2 on HELIX_W_KEY[0, k] for k = 2..30, zero elsewhere; |v|^2 = 117
(mxfold2_gauge.v_norm2). Moving the weights along v moves the positional outputs along the
output gauge h, so the hinge_mix loss is flat along v except on the long-helix "breaker"
sequences (runs/2026-10-10_mxfold2_vg_breaker).

THE PROJECTION (A3, plan wild-kindling-moth.md): w <- w - ((v.w)/117) v, i.e.
    c = (b[0] - 2 * sum_{k=2..30} W[0, k]) / 117      computed in float64 on the CPU
    b[0] <- b[0] - c,   W[0, k] <- W[0, k] + 2c       targets in float64, written back float32
Only those 30 entries are written. The arithmetic is done on CPU copies, so a projection gives
bitwise the same result on any device. If c is not finite (the weights are already NaN/inf)
nothing is written and c is returned as it is: the caller reports a diverged run.

THE BOUND v_bound(b', W', c), on the COMPUTED |v.w'| after a projection that used shift c
(u = 2^-53, A' = |b'[0]| + 2 sum_k |W'[0, k]| over the WRITTEN float32 values):
  write b' = fl32(beta), beta = fl64(b - c) and W'_k = fl32(omega_k), omega_k = fl64(W_k + 2c).
  Exact v.w' = (N - 117 c) + (delta_b - 2 sum delta_k) + (eps_b - 2 sum eps_k), with
    N = b - 2 sum W_k the exact pre-projection v.w,
    delta = the float64 rounding of beta / omega,  |delta| <= u |beta| <= u (1 + 2^-24) |b'|,
    eps   = the float32 rounding of the write: each written entry carries at most half an ulp
            of its new value, |eps| <= 2^-24 |y| + 2^-150 (the second term only for subnormals).
  (i)  float32 writes:  |eps_b| + 2 sum |eps_k| <= 2^-24 A' + 59 * 2^-150.
  (ii) the float64 shift: c = fl(fl(b - 2 fl(sum W)) / 117), so
       |N - 117 c| <= (2u + u^2) |N| + 2 gamma_28 sum |W_k| (1 + u)^2 <= 30.1 u A_pre, where
       A_pre = |b| + 2 sum |W_k| <= (1 + 2^-23) A' + 117 |c| (b = b' + c - eps_b - delta_b ...).
  (iii) the float64 additions: |delta_b| + 2 sum |delta_k| <= 1.01 u A'.
  (iv) the MEASUREMENT: the asserted value is v.w' recomputed in float64 (v_dot / vw64, any
       summation order, on any device): |computed - exact| <= (u + gamma_28 (1 + u)) A'
       <= 29.1 u A'.
  Sum: |computed v.w'| <= 2^-24 A' + 62 u (A' + 117 |c|) + 2^-143.
  v_bound returns 2^-24 (1 + 2^-23) A' + 2^-45 (A' + 117 |c|) + 2^-140. The factor (1 + 2^-23)
  absorbs the float64 evaluation of A' itself (relative error <= 30 u), and 2^-45 = 256 u is
  four times the 62 u needed, which absorbs the float64 evaluation of the bound. The bound
  needs |c| because the shift's float64 error scales with the PRE-projection magnitudes, which
  can exceed the post-projection ones by 117 |c| (w nearly along v); v_bound therefore takes
  c explicitly (c = 0 when no projection was applied, e.g. the planted skip control).
  For weights of magnitude 0.1-10 the bound is ~1e-7 to 1e-5; one Adam step moves v.w by
  ~1e-3, so a skipped projection is caught on its first step.

Torch only (imported from .venv-mxfold2 and from .venv).
"""

from __future__ import annotations

import math

import torch

from scripts.mxfold2_gauge import MAX_HELIX, v_norm2

K_LO, K_HI = 2, MAX_HELIX + 1            # W[0, 2:31], the 29 helix-length weights on v
V_NORM2 = v_norm2()                      # 117.0
F32_REL = 2.0 ** -24 * (1.0 + 2.0 ** -23)  # half an ulp per written entry, with slack
F64_REL = 2.0 ** -45                      # 256 u64: the float64 shift, sums and measurement
ABS_FLOOR = 2.0 ** -140                   # subnormal writes (59 * 2^-150 < 2^-143)
assert V_NORM2 == 117.0
# the runner's exit messages (scripts/mxfold2_ablation_train.py), read by the launcher
BOUND_MSG = "MXF_PROJECT BOUND VIOLATION"
NONFINITE_MSG = "MXF_PROJECT NON-FINITE v.w"


def _check(bias: torch.Tensor, weight: torch.Tensor) -> None:
    assert bias.dtype == torch.float32 and weight.dtype == torch.float32, \
        f"the bound is stated for float32 weights, got {bias.dtype} / {weight.dtype}"
    assert bias.dim() == 1 and bias.numel() >= 1, tuple(bias.shape)
    assert weight.dim() == 2 and weight.shape[0] >= 1 and weight.shape[1] == MAX_HELIX + 1, \
        tuple(weight.shape)


def vw64(bias: torch.Tensor, weight: torch.Tensor) -> float:
    """v.w = b[0] - 2 sum_{k=2..30} W[0, k] in float64, on the tensors' own device.

    The expression of scripts/mxfold2_gauge.v_dot, so on the same tensors it returns the same
    float bit for bit (tests/test_mxfold2_vproject.py checks this on a MixC model)."""
    b = bias.detach().double()
    w = weight.detach().double()
    return float(b[0] - 2.0 * w[0, K_LO:K_HI].sum())


@torch.no_grad()
def project_v_(bias: torch.Tensor, weight: torch.Tensor) -> dict:
    """w <- w - ((v.w)/117) v in place on the two tensors; returns {"pre": v.w before, "c"}.

    "pre" is the float64 v.w the shift was computed from (CPU, before the projection)."""
    _check(bias, weight)
    b = bias.detach()[:1].double().cpu()
    w = weight.detach()[0, K_LO:K_HI].double().cpu()
    pre = float(b[0] - 2.0 * w.sum())
    c = pre / V_NORM2
    if not math.isfinite(c):
        return {"pre": pre, "c": c, "written": False}
    nb = (b - c).to(torch.float32)
    nw = (w + 2.0 * c).to(torch.float32)
    bias[:1].copy_(nb)
    weight[0, K_LO:K_HI].copy_(nw)
    return {"pre": pre, "c": c, "written": True}


def v_bound(bias: torch.Tensor, weight: torch.Tensor, c: float) -> float:
    """Rigorous upper bound on the computed |v.w| of float32 weights just written by a
    projection that used float64 shift c (c = 0.0 if none was applied). See the module
    docstring for the derivation. NaN if c or a weight is not finite."""
    _check(bias, weight)
    b = bias.detach()[:1].double().cpu()
    w = weight.detach()[0, K_LO:K_HI].double().cpu()
    a = float(b.abs().sum() + 2.0 * w.abs().sum())
    return F32_REL * a + F64_REL * (a + V_NORM2 * abs(float(c))) + ABS_FLOOR
