"""Dual activation of the thresholded ReLU for jointly Gaussian inputs (docs/nonlin.md §2).

Setting.  Two voltages (u_i, u_j) jointly Gaussian with standard deviations s_i, s_j and
correlation rho.  Rates r = relu(u - E u - tau * sd(u)) (homeostatic threshold, the same
tau for every neuron).  Then

    Cov(r_i, r_j) = s_i s_j k_tau(rho),

with ONE scalar function k_tau of the correlation (this is why the threshold is set in
units of each neuron's own sd).  With X, Y standard bivariate normal, corr rho,

    k_tau(rho) = E[(X-tau)+ (Y-tau)+] - m_tau^2,           m_tau = E[(X-tau)+]
    k_tau'(rho) = P(X > tau, Y > tau) = Phi2(-tau, -tau; rho)      (Price's theorem)
    k_tau''(rho) = phi2(tau, tau; rho) > 0                         (convex)
    k_tau(0) = 0,  k_tau(1) = Var((X-tau)+),  k_tau(-1) = -m_tau^2 for tau >= 0.

Integrating Price twice from rho = 0 (substituting s = sin t to remove the endpoint
singularity):

    k_tau(rho) = rho Phi(-tau)^2 + (1/2pi) int_0^{arcsin rho} (rho - sin t) exp(-tau^2/(1+sin t)) dt.

tau = 0 is the degree-1 arc-cosine kernel minus the product of means:
    k_0(rho) = (sin phi + (pi - phi) cos phi - 1) / (2 pi),  phi = arccos rho
             = rho/4 + rho^2/(4 pi) + rho^4/(48 pi) + ...
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr  # standard normal CDF

_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)
_GL_X = 0.5 * (_GL_X + 1.0); _GL_W = 0.5 * _GL_W          # nodes/weights on [0, 1]
SQ2PI = math.sqrt(2 * math.pi)


def npdf(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


def relu_mean(tau):
    """m_tau = E[(X - tau)+], X ~ N(0,1)."""
    return npdf(tau) - tau * ndtr(-tau)


def relu_second(tau):
    """E[(X - tau)+^2]."""
    return (1 + tau * tau) * ndtr(-tau) - tau * npdf(tau)


def relu_var(tau):
    """Var((X - tau)+) = k_tau(1)."""
    return relu_second(tau) - relu_mean(tau) ** 2


def k(rho, tau=0.0):
    """k_tau(rho), vectorised over rho (any shape). Exact (48-point Gauss-Legendre in t)."""
    rho = np.clip(np.asarray(rho, float), -1.0, 1.0)
    if tau == 0.0:
        phi = np.arccos(rho)
        return (np.sin(phi) + (np.pi - phi) * rho - 1.0) / (2 * np.pi)
    a = np.arcsin(rho)[..., None]
    t = a * _GL_X
    st = np.sin(t)
    integ = (a[..., 0] * ((rho[..., None] - st) * np.exp(-tau * tau / (1.0 + st + 1e-300)) * _GL_W).sum(-1))
    return rho * ndtr(-tau) ** 2 + integ / (2 * np.pi)


def dk(rho, tau=0.0):
    """k_tau'(rho) = Phi2(-tau, -tau; rho) = P(both neurons active)."""
    rho = np.clip(np.asarray(rho, float), -1.0, 1.0)
    if tau == 0.0:
        return (np.pi - np.arccos(rho)) / (2 * np.pi)
    a = np.arcsin(rho)[..., None]
    st = np.sin(a * _GL_X)
    return ndtr(-tau) ** 2 + a[..., 0] * (np.exp(-tau * tau / (1.0 + st + 1e-300)) * _GL_W).sum(-1) / (2 * np.pi)


class KTable:
    """Monotone lookup table for k_tau and its inverse (rho in [-1, 1])."""

    def __init__(self, tau=0.0, n=8001):
        self.tau = float(tau)
        self.rho = np.linspace(-1.0, 1.0, n)
        self.k = k(self.rho, tau)
        self.kmin, self.kmax = float(self.k[0]), float(self.k[-1])
        # strictly increasing for the inverse (flat only at rho = -1 when tau > 0)
        kk = np.maximum.accumulate(self.k + 1e-15 * np.arange(n))
        self._k_inc = kk

    def __call__(self, rho):
        return np.interp(rho, self.rho, self.k)

    def inv(self, kv):
        """k_tau^{-1}; values outside (k(-1), k(1)) clip to rho = -1 / 1."""
        return np.interp(kv, self._k_inc, self.rho)


# ---------------------------------------------------------------- torch versions
def k_torch(rho, tau=0.0):
    """Differentiable k_tau(rho) (torch), same formula; rho is clamped to (-1, 1)."""
    import torch
    rho = rho.clamp(-1 + 1e-7, 1 - 1e-7)
    if tau == 0.0:
        phi = torch.arccos(rho)
        return (torch.sin(phi) + (math.pi - phi) * rho - 1.0) / (2 * math.pi)
    x = torch.as_tensor(_GL_X, dtype=rho.dtype, device=rho.device)
    w = torch.as_tensor(_GL_W, dtype=rho.dtype, device=rho.device)
    a = torch.arcsin(rho)[..., None]
    st = torch.sin(a * x)
    integ = a[..., 0] * ((rho[..., None] - st) * torch.exp(-tau * tau / (1.0 + st + 1e-30)) * w).sum(-1)
    return rho * float(ndtr(-tau)) ** 2 + integ / (2 * math.pi)


def gaussian_rate_cov(Cuu, tau=0.0, Cuv=None, var_v=None):
    """Gaussian-closure prediction of rate covariances (unit gain).

    Cuu: (n, n) voltage covariance among n neurons -> (n, n) predicted Cov(r_i, r_j)
    (diagonal = k_tau(1) * var).  If Cuv (n, m) and var_v (m,) are given, also returns the
    cross block Cov(r_i, s_b) for a second population (e.g. boundary neurons)."""
    s = np.sqrt(np.maximum(np.diag(Cuu), 0))
    den = np.outer(s, s)
    rho = np.divide(Cuu, den, out=np.zeros_like(Cuu), where=den > 0)
    out = den * k(rho, tau)
    if Cuv is None:
        return out
    sv = np.sqrt(np.maximum(var_v, 0))
    den2 = np.outer(s, sv)
    rho2 = np.divide(Cuv, den2, out=np.zeros_like(Cuv), where=den2 > 0)
    return out, den2 * k(rho2, tau)


def threshold_bias(Hmu, var_u, tau):
    """Homeostatic threshold b_i = E u_i + tau sd(u_i)."""
    return Hmu + tau * np.sqrt(np.maximum(var_u, 0))
