"""Mean-field rectified Hebb rule for GAUSSIAN g (numpy; docs/nonlin.md §2).

Mirrors hebb.core.step (same clip order) with Cov(u_i,u_j) replaced by the exact
Gaussian rate covariance  c^2 s_i s_j k_tau(rho_ij)  (I-I) and  c^2 s_i s_b k_tau(rho_ib)
(I-B), s_i^2 = h_i^T Sigma h_i, s_b^2 = Sigma_bb.  Only valid for Gaussian g: for any
other distribution the expectation needs the samples (stochastic mode, hebb.nonlin.model).
"""
from __future__ import annotations

import numpy as np

from hebb import core
from hebb.nonlin import dual


def rate_covs(H, Sigma, tau=0.0, c=1.0):
    """(Cii (N,N), Cib (N,B)) Gaussian-closure rate covariances, gain c."""
    HS = H @ Sigma
    Cuu = HS @ H.T
    Cii, Cib = dual.gaussian_rate_cov(Cuu, tau, HS, np.diag(Sigma))
    return c * c * Cii, c * c * Cib


def step(Wii, Wib, Mii, Mib, Sigma, P: core.Params, tau=0.0, c=1.0):
    Wii = np.maximum(Wii, P.wmin * Mii); Wib = np.maximum(Wib, P.wmin * Mib)
    H = core.harmonic(Wii, Wib)
    Cii, Cib = rate_covs(H, Sigma, tau, c)
    w0 = -P.eta * P.theta
    Wii = Wii + P.dt * (w0 - P.lam * Wii + P.gamma * P.eta * Cii)
    Wib = Wib + P.dt * (w0 - P.lam * Wib + P.eta * Cib)
    Wii = np.maximum(Wii, P.wmin) * Mii; np.fill_diagonal(Wii, 0.0)
    Wib = np.maximum(Wib, P.wmin) * Mib
    return Wii, Wib, H


def drift(Wii, Wib, Mii, Mib, Sigma, P: core.Params, tau=0.0, c=1.0):
    Wii = np.maximum(Wii, P.wmin * Mii); Wib = np.maximum(Wib, P.wmin * Mib)
    H = core.harmonic(Wii, Wib)
    Cii, Cib = rate_covs(H, Sigma, tau, c)
    w0 = -P.eta * P.theta
    fii = (w0 - P.lam * Wii + P.gamma * P.eta * Cii) * Mii
    fib = (w0 - P.lam * Wib + P.eta * Cib) * Mib
    np.fill_diagonal(fii, 0.0)
    return fii, fib, H, Wii, Wib


def stationarity(Wii, Wib, Mii, Mib, Sigma, P: core.Params, tau=0.0, c=1.0):
    fii, fib, _, Wc, Wbc = drift(Wii, Wib, Mii, Mib, Sigma, P, tau, c)
    pii = np.where((Wc <= P.wmin) & (fii < 0), 0.0, np.abs(fii)) * Mii
    pib = np.where((Wbc <= P.wmin) & (fib < 0), 0.0, np.abs(fib)) * Mib
    return float(max(np.triu(pii, 1).max(), pib.max()))


def run(Sigma, N, P: core.Params, tau=0.0, c=1.0, T=4000, seed=0, p=1.0, ii_scale=0.0, ib_scale=0.2,
        linear=False):
    """linear=True runs the base rule (core.step) from the same init, for paired comparisons."""
    rng = np.random.default_rng(seed)
    Wii, Wib, Mii, Mib = core.init_state(N, Sigma.shape[0], rng, p, ii_scale, ib_scale)
    for _ in range(T):
        if linear:
            Wii, Wib, _ = core.step(Wii, Wib, Mii, Mib, Sigma, P)
        else:
            Wii, Wib, _ = step(Wii, Wib, Mii, Mib, Sigma, P, tau, c)
    res = (core.stationarity(Wii, Wib, Mii, Mib, Sigma, P)['r'] if linear
           else stationarity(Wii, Wib, Mii, Mib, Sigma, P, tau, c))
    return dict(Wii=Wii, Wib=Wib, Mii=Mii, Mib=Mib, H=core.harmonic(Wii, Wib), residual=res)
