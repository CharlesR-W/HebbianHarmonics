"""Canonical reference implementation of the model in SPEC.md §1 (numpy).

Existing signatures are frozen (other code imports them); add, don't change.
State: Wii (N,N) symmetric zero-diagonal, Wib (N,B). Masks Mii (N,N), Mib (N,B) in {0,1}
mark potential synapses; non-potential synapses are exactly 0 (not w_min).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    theta: float = 0.3      # = -w0/eta, in covariance units
    lam: float = 1.0        # weight decay
    eta: float = 1.0        # Hebbian gain on interior-boundary synapses
    gamma: float = 1.0      # eta_II / eta_IB
    wmin: float = 1e-3      # floor on potential synapses
    dt: float = 0.05


def init_state(N, B, rng, p=1.0, ii_scale=0.0, ib_scale=0.2):
    """Random start. Wib ~ Exp(ib_scale) spreads embeddings over the simplex;
    Wii ~ U(0, ii_scale). Masks are Bernoulli(p), symmetric for I-I; every neuron
    keeps at least one boundary synapse so L_II stays invertible."""
    Mii = np.triu(rng.random((N, N)) < p, 1); Mii = (Mii | Mii.T).astype(float)
    Mib = (rng.random((N, B)) < p).astype(float)
    for i in np.where(Mib.sum(1) == 0)[0]:
        Mib[i, rng.integers(B)] = 1.0
    Wii = rng.uniform(0, 1, (N, N)) * ii_scale; Wii = np.triu(Wii, 1); Wii = (Wii + Wii.T) * Mii
    Wib = rng.exponential(ib_scale, (N, B)) * Mib
    return Wii, Wib, Mii, Mib


def harmonic(Wii, Wib):
    """H = L_II^{-1} W_IB; rows are probability vectors over boundary neurons."""
    deg = Wii.sum(1) + Wib.sum(1)
    return np.linalg.solve(np.diag(deg) - Wii, Wib)


def step(Wii, Wib, Mii, Mib, Sigma, P: Params):
    """One Euler step of dw = w0 - lam*w + eta_ij*Cov(u_i,u_j), clipped at wmin.
    Sigma may be the exact covariance (mean-field) or a minibatch estimate."""
    Wii = np.maximum(Wii, P.wmin * Mii); Wib = np.maximum(Wib, P.wmin * Mib)
    H = harmonic(Wii, Wib)
    HS = H @ Sigma
    w0 = -P.eta * P.theta
    Wii = Wii + P.dt * (w0 - P.lam * Wii + P.gamma * P.eta * (HS @ H.T))
    Wib = Wib + P.dt * (w0 - P.lam * Wib + P.eta * HS)
    Wii = np.maximum(Wii, P.wmin) * Mii; np.fill_diagonal(Wii, 0.0)
    Wib = np.maximum(Wib, P.wmin) * Mib
    return Wii, Wib, H


def run(Sigma, N, P: Params, T=4000, seed=0, p=1.0, ii_scale=0.0, ib_scale=0.2, log_every=0):
    rng = np.random.default_rng(seed)
    Wii, Wib, Mii, Mib = init_state(N, Sigma.shape[0], rng, p, ii_scale, ib_scale)
    hist = []
    for t in range(T):
        Wii, Wib, H = step(Wii, Wib, Mii, Mib, Sigma, P)
        if log_every and t % log_every == 0:
            hist.append(H)
    return dict(Wii=Wii, Wib=Wib, Mii=Mii, Mib=Mib, H=harmonic(Wii, Wib), hist=hist)


def clusters(H, tol=0.1):
    """Greedy leader clustering of embeddings in L1 distance. Returns (centroids, labels)."""
    cents, labels = [], np.zeros(len(H), int)
    for i, h in enumerate(H):
        for k, c in enumerate(cents):
            if np.abs(h - c).sum() < tol:
                labels[i] = k; break
        else:
            labels[i] = len(cents); cents.append(h.copy())
    return np.array(cents), labels
