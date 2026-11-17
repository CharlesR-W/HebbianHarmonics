"""Moment-based sample synthesis under explicit generative assumptions (export item 6).

Second moments do not determine p(x).  Each generator below ADDS a model:

  pixels lift  ->  Gaussian completion  x ~ N(xbar, C_x)
      xbar      = the data mean.  The Hebb rule uses Cov, so the mean is NOT in the weights;
                  this is outside information, stated as an assumption.
      C_x       = C / scale^2: C is read out in the "parameters known" setting (theta and
                  lambda/eta known), so it is in absolute lifted units; scale is the lift's
                  global feature scale (pixels -> lifted units).
      PSD repair: negative eigenvalues of C are clipped to 0 (mass reported).
      Unidentified directions: the ridge readout is minimum-norm, so directions the
                  equations do not constrain get ~0 variance (completion by zero).
      Display:  samples are clipped to [0, 1] for the PNG; clipping changes the moments,
                  so moment checks use the unclipped samples.

  rff lift  ->  feature-space Gaussian + pre-image (exploratory)
      g ~ N(mu_g, C) (mu_g = data mean of the features, an assumption), then
      x* = argmin_{x in [0,1]^196} ||phi(x) - g||^2 by projected Adam from x0 = xbar,
      identical settings/seeds for every row.  A Gaussian g is almost never on the image
      of phi, so the optimiser, init and box constraint shape the result; the achieved
      relative residual ||phi(x*) - g|| / ||g - mu_g|| is exported.

Rows (same generator, same seeds within a sheet): "real data", "true-moment generator"
(Sigma_g: a same-generator oracle reference, not a universal ceiling),
"weight-reconstructed generator" (C_hat from the trained weights), "baseline generator"
(diag(Sigma_g) and the untrained-network readout).
Metric: squared kernel MMD (Gaussian kernel, median-heuristic bandwidth) between generated
samples and held-out test digits, plus a moment-consistency check.
"""
from __future__ import annotations

import numpy as np
import torch


def psd_factor(C: np.ndarray):
    C = 0.5 * (C + C.T).astype(np.float64)
    ev, V = np.linalg.eigh(C)
    neg = float(np.abs(ev[ev < 0]).sum() / max(np.abs(ev).sum(), 1e-300))
    L = V * np.sqrt(np.clip(ev, 0, None))
    return L, neg, (L @ L.T)


def mmd2(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased squared MMD with k(a,b) = exp(-|a-b|^2 / (2 sigma^2))."""
    X = torch.as_tensor(X, dtype=torch.float64); Y = torch.as_tensor(Y, dtype=torch.float64)

    def k(A, B):
        return torch.exp(-torch.cdist(A, B).pow(2) / (2 * sigma ** 2))
    n, m = len(X), len(Y)
    kxx = (k(X, X).sum() - n) / (n * (n - 1))
    kyy = (k(Y, Y).sum() - m) / (m * (m - 1))
    return float(kxx + kyy - 2 * k(X, Y).mean())


def gaussian_samples(mean: np.ndarray, C: np.ndarray, Z: np.ndarray):
    L, neg, Cpsd = psd_factor(C)
    return mean + Z @ L.T, neg, Cpsd


def rel_fro(A, B):
    return float(np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-300))


def preimage(lift, G: np.ndarray, x0: np.ndarray, steps: int = 300, lr: float = 0.02):
    """Projected-Adam pre-image x* in [0,1]^196 of target features G (n,B)."""
    Gt = torch.as_tensor(G, dtype=torch.float32)
    x = torch.as_tensor(np.repeat(x0[None], len(G), 0), dtype=torch.float32).clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = (lift.apply_torch(x, torch) - Gt).pow(2).sum()
        loss.backward(); opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    with torch.no_grad():
        res = (lift.apply_torch(x, torch) - Gt).norm(dim=1)
    return x.detach().numpy(), res.numpy()
