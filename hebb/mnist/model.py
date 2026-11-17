"""Torch implementation of the SPEC §1 model with minibatch (stochastic) or exact
(mean-field) Hebbian updates.  Mirrors hebb.core.step exactly (same clip order),
checked by hebb/mnist/check_core.py.

State: Wii (N,N) symmetric, zero diagonal; Wib (N,B).  p = 1 (every I-I and I-B
synapse is potential), so masks are implicit: Mii = 1 - I, Mib = 1.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import torch


@dataclass
class Cfg:
    name: str = "run"
    lift: str = "pixels"        # pixels | rff | relu
    B: int = 196
    bw_mult: float = 1.0        # rff bandwidth multiplier on the median heuristic
    lift_seed: int = 0
    N: int = 512
    theta: float = 0.3
    wmin: float = 1e-3
    gamma: float = 1.0
    lam: float = 1.0
    eta: float = 1.0
    dt: float = 0.1
    steps: int = 3000
    batch: int = 256
    mode: str = "sgd"           # sgd (minibatch covariance) | mf (exact Sigma_g)
    seed: int = 0
    ii_scale: float = 0.2       # Wii ~ U(0, ii_scale)  (mean 0.1)
    ib_scale: float = 0.1       # Wib ~ Exp(mean ib_scale)  (ib_shape = 1)
    ib_shape: float = 1.0       # Wib ~ Gamma(ib_shape, ib_scale/ib_shape): <1 = heavy-tailed, still dense
    n_snap: int = 36
    n_sub: int = 512            # neurons whose H is stored at every snapshot
    avg_frac: float = 0.2       # time-average weights over the last avg_frac of steps
    readout_every_snap: bool = True
    cg_iters: int = 400
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def init_weights(cfg: Cfg, device, dtype):
    """Generic dense random init, identical to hebb.core.init_state(p=1)."""
    rng = np.random.default_rng(cfg.seed)
    N, B = cfg.N, cfg.B
    # core.init_state draws the mask first; with p=1 the draws are irrelevant but we keep the
    # same RNG stream so that the numpy reference reproduces this init exactly.
    _ = np.triu(rng.random((N, N)) < 1.0, 1)
    _ = rng.random((N, B))
    Wii = rng.uniform(0, 1, (N, N)) * cfg.ii_scale
    Wii = np.triu(Wii, 1); Wii = Wii + Wii.T
    if cfg.ib_shape == 1.0:
        Wib = rng.exponential(cfg.ib_scale, (N, B))
    else:
        Wib = rng.gamma(cfg.ib_shape, cfg.ib_scale / cfg.ib_shape, (N, B))
    return (torch.as_tensor(Wii, device=device, dtype=dtype),
            torch.as_tensor(Wib, device=device, dtype=dtype))


def harmonic(Wii, Wib):
    """H = L_II^{-1} W_IB via Cholesky (L_II is SPD: strictly diagonally dominant)."""
    deg = Wii.sum(1) + Wib.sum(1)
    L = torch.diag(deg) - Wii
    Lc, info = torch.linalg.cholesky_ex(L)
    if int(info) != 0:
        return torch.linalg.solve(L, Wib)
    return torch.cholesky_solve(Wib, Lc)


def clip(Wii, Wib, wmin):
    Wii = Wii.clamp_min(wmin); Wii.fill_diagonal_(0.0)
    return Wii, Wib.clamp_min(wmin)


def covs_minibatch(H, G):
    """Unbiased minibatch covariances: Cii = Cov(u,u) (N,N), Cib = Cov(u,g) (N,B)."""
    Gc = G - G.mean(0, keepdim=True)
    U = Gc @ H.T
    m = G.shape[0]
    return (U.T @ U) / (m - 1), (U.T @ Gc) / (m - 1)


def covs_exact(H, Sigma):
    HS = H @ Sigma
    return HS @ H.T, HS


def step(Wii, Wib, cfg: Cfg, Sigma=None, G=None):
    """One Euler step, same order as core.step: floor-clip, H, update, floor-clip."""
    Wii, Wib = clip(Wii, Wib, cfg.wmin)
    H = harmonic(Wii, Wib)
    Cii, Cib = covs_exact(H, Sigma) if G is None else covs_minibatch(H, G)
    w0 = -cfg.eta * cfg.theta
    Wii = Wii + cfg.dt * (w0 - cfg.lam * Wii + cfg.gamma * cfg.eta * Cii)
    Wib = Wib + cfg.dt * (w0 - cfg.lam * Wib + cfg.eta * Cib)
    Wii, Wib = clip(Wii, Wib, cfg.wmin)
    return Wii, Wib, H


def drift(Wii, Wib, Sigma, cfg: Cfg):
    """Mean-field drift dw/dt at the given (clipped) weights, and the projected
    residual (edges at the floor with negative drift count as 0)."""
    H = harmonic(Wii, Wib)
    Cii, Cib = covs_exact(H, Sigma)
    w0 = -cfg.eta * cfg.theta
    dii = w0 - cfg.lam * Wii + cfg.gamma * cfg.eta * Cii
    dib = w0 - cfg.lam * Wib + cfg.eta * Cib
    tol = cfg.wmin * (1 + 1e-6)
    dii = torch.where((Wii <= tol) & (dii < 0), torch.zeros_like(dii), dii)
    dii.fill_diagonal_(0.0)
    dib = torch.where((Wib <= tol) & (dib < 0), torch.zeros_like(dib), dib)
    return dii, dib, H


def residual_stats(Wii, Wib, Sigma, cfg: Cfg):
    dii, dib, H = drift(Wii, Wib, Sigma, cfg)
    a = torch.cat([dii.abs().flatten(), dib.abs().flatten()])
    act = torch.cat([(Wii > 2 * cfg.wmin).flatten(), (Wib > 2 * cfg.wmin).flatten()])
    return dict(res_max=float(a.max()), res_mean=float(a.mean()),
                res_active_mean=float(a[act].mean()) if bool(act.any()) else 0.0,
                res_rel=float(a.max() / max(float(Wib.max()), float(Wii.max()), 1e-12)))
