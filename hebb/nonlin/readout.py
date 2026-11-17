"""Weight-space readout for the rectified rule under the Gaussian closure (docs/nonlin.md §2.3).

Fixed point on active edges (w > 2 w_min, the sibling's convention), rates r = c f(u - b):

    I-I:  gamma c^2 s_i s_j k_tau(rho_ij) = theta + (lambda/eta) w_ij
    I-B:        c^2 s_i s_b k_tau(rho_ib) = theta + (lambda/eta) w_ib

with s_i^2 = h_i^T C h_i, s_b^2 = C_bb, rho_ij = h_i^T C h_j / (s_i s_j), rho_ib = (HC)_ib/(s_i s_b).
Every equation is homogeneous of degree 1 in C, so with (theta, lambda/eta, gamma, c, tau)
known the scale of C is fixed; but the variances enter EVERY equation (not only the
unobserved diagonal), so they have to be estimated jointly with the correlations.

Estimator `gauss_readout`: nonlinear least squares over symmetric C,
    sum_fam  mean_{active edges}( s s k_tau(rho) - Q )^2 / mean(Q^2)  +  alpha ||C||_F^2,
solved by L-BFGS from the "chord" initialisation C0 = LS readout of Q / k_tau(1)
(k_tau(rho) <= rho k_tau(1) on [0,1] because k_tau is convex with k_tau(0) = 0; C0 is also
what you get by ignoring the nonlinearity, reported as the `chord` baseline).  k_tau is
evaluated from a dense table with its exact derivative (Price: k' = P(both active)) in the
backward pass and extended linearly outside [-1, 1] so the loss stays smooth for
non-PSD iterates.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from hebb.mnist import readout as R
from hebb.nonlin import dual


class _Tab:
    def __init__(self, tau, n=8001, device="cpu", dtype=torch.float64):
        rho = np.linspace(-1, 1, n)
        self.n = n
        self.k = torch.as_tensor(dual.k(rho, tau), device=device, dtype=dtype)
        self.dk = torch.as_tensor(dual.dk(rho, tau), device=device, dtype=dtype)
        self.k1, self.dk1 = float(self.k[-1]), float(self.dk[-1])
        self.km1, self.dkm1 = float(self.k[0]), float(self.dk[0])


class _KFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho, tab):
        x = (rho.clamp(-1, 1) + 1) * 0.5 * (tab.n - 1)
        i0 = x.floor().clamp(0, tab.n - 2).long(); fr = x - i0
        k = tab.k[i0] * (1 - fr) + tab.k[i0 + 1] * fr
        dk = tab.dk[i0] * (1 - fr) + tab.dk[i0 + 1] * fr
        k = torch.where(rho > 1, tab.k1 + tab.dk1 * (rho - 1), k)
        k = torch.where(rho < -1, tab.km1 + tab.dkm1 * (rho + 1), k)
        ctx.save_for_backward(dk)
        return k

    @staticmethod
    def backward(ctx, g):
        (dk,) = ctx.saved_tensors
        return g * dk, None


def rect_equations(Wii, Wib, theta, lam=1.0, eta=1.0, gamma=1.0, wmin=1e-3, c=1.0):
    """Masks and targets Q (= s s k(rho) in g units) on active edges."""
    Mii = (Wii > 2 * wmin).to(Wii.dtype); Mii.fill_diagonal_(0.0)
    Mib = (Wib > 2 * wmin).to(Wib.dtype)
    Qii = (theta + lam / eta * Wii) / (gamma * c * c)
    Qib = (theta + lam / eta * Wib) / (c * c)
    return Mii, Qii, Mib, Qib


def predict(H, C, tau_tab, var_floor=1e-12):
    """Gaussian-closure rate covariances (unit gain) for embedding H and input covariance C."""
    HC = H @ C
    M = HC @ H.T
    s = M.diagonal().clamp_min(var_floor).sqrt()
    sb = C.diagonal().clamp_min(var_floor).sqrt()
    rho_ii = M / (s[:, None] * s[None, :])
    rho_ib = HC / (s[:, None] * sb[None, :])
    Pii = s[:, None] * s[None, :] * _KFn.apply(rho_ii, tau_tab)
    Pib = s[:, None] * sb[None, :] * _KFn.apply(rho_ib, tau_tab)
    return Pii, Pib


def gauss_readout(H, Mii, Qii, Mib, Qib, tau=0.0, family="both", C0=None, iters=300,
                  ridge_rel=1e-4, fit_theta=False, a_ii=1.0, a_ib=1.0, verbose=False):
    """Nonlinear LS inversion of the dual activation.  Tensors float64 (any device).

    fit_theta: theta unknown -> Q = Q_base + theta_hat * a_fam, with Q_base built at theta = 0
    (pass Qii/Qib computed with theta = 0 and a_ii = 1/(gamma c^2), a_ib = 1/c^2)."""
    dev, dt = H.device, torch.float64
    H = H.to(dt); Mii, Qii, Mib, Qib = (t.to(dt) for t in (Mii, Qii, Mib, Qib))
    B = H.shape[1]
    tab = _Tab(tau, device=dev, dtype=dt)
    use_ii = family in ("both", "ii") and float(Mii.sum()) > 0
    use_ib = family in ("both", "ib") and float(Mib.sum()) > 0
    nii, nib = max(float(Mii.sum()), 1.0), max(float(Mib.sum()), 1.0)
    # chord initialisation (also the "ignore the nonlinearity" baseline)
    if C0 is None:
        C0, _ = R.ls_readout(H, Mii, Qii / tab.k1, Mib, Qib / tab.k1, family, iters=400)
    C0 = C0.to(dt)
    sii = float((Mii * Qii).pow(2).sum() / nii) if use_ii else 1.0
    sib = float((Mib * Qib).pow(2).sum() / nib) if use_ib else 1.0
    # ridge scale: linearised operator norm (power iteration), times dk(1)^2
    Ht = H.T
    X = torch.randn(B, B, device=dev, dtype=dt, generator=torch.Generator(dev).manual_seed(0)); X = 0.5 * (X + X.T)
    lam_max = 0.0
    for _ in range(20):
        Y = torch.zeros_like(X); HX = H @ X
        if use_ii: Y += (Ht @ (Mii * (HX @ Ht)) @ H) / (nii * sii)
        if use_ib: Y += 0.5 * ((Ht @ (Mib * HX)) + (Ht @ (Mib * HX)).T) / (nib * sib)
        lam_max = float(Y.norm()); X = Y / max(lam_max, 1e-300)
    alpha = ridge_rel * lam_max * tab.dk1 ** 2
    A = C0.clone().requires_grad_(True)
    th = torch.zeros((), device=dev, dtype=dt, requires_grad=True)
    params = [A, th] if fit_theta else [A]
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=30, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-14, tolerance_change=1e-16)
    state = {}

    def closure():
        opt.zero_grad()
        C = 0.5 * (A + A.T)
        Pii, Pib = predict(H, C, tab)
        loss = alpha * (C * C).sum()
        tii = Qii + th * a_ii if fit_theta else Qii
        tib = Qib + th * a_ib if fit_theta else Qib
        if use_ii:
            state["ii"] = (Mii * (Pii - tii)).pow(2).sum() / nii
            loss = loss + state["ii"] / sii
        if use_ib:
            state["ib"] = (Mib * (Pib - tib)).pow(2).sum() / nib
            loss = loss + state["ib"] / sib
        loss.backward()
        state["loss"] = float(loss.detach())
        return loss

    opt.step(closure)
    C = (0.5 * (A + A.T)).detach()
    info = dict(loss=state.get("loss"), alpha=alpha, n_ii=float(Mii.sum()) if use_ii else 0.0,
                n_ib=float(Mib.sum()) if use_ib else 0.0, theta_fit=float(th.detach()) if fit_theta else None,
                rms_ii=float(state["ii"].detach()) ** 0.5 if "ii" in state else None,
                rms_ib=float(state["ib"].detach()) ** 0.5 if "ib" in state else None,
                rel_rms_ii=(float(state["ii"].detach()) / sii) ** 0.5 if "ii" in state else None,
                rel_rms_ib=(float(state["ib"].detach()) / sib) ** 0.5 if "ib" in state else None)
    return C, C0, info
