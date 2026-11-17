"""Stochastic (minibatch) training of the rectified-output Hebb rule on raw MNIST pixels.

Linear circuit, rectified output (docs/nonlin.md §1):
    u = H g                               (Kirchhoff, unchanged)
    r_i = c * relu(u_i - b_i),  b_i = E[u_i] + tau * sd(u_i)          interior rates
    s_b = c * relu(g_b - b_b),  b_b = E[g_b] + tau * sd(g_b)          boundary rates
    dw_ij/dt = w0 - lam w_ij + gamma eta Cov(r_i, r_j)               I-I
    dw_ib/dt = w0 - lam w_ib + eta Cov(r_i, s_b)                     I-B
Thresholds use the population moments of u_i (fast intrinsic homeostasis limit);
c is ONE global gain fixed from the pixel data so that the mean boundary-rate variance
is 1 (the pixel lift is likewise scaled to mean variance 1), so theta is on the same
numerical scale for both rules.  rule="linear" is the base model on the same inputs
(Cov(u_i,u_j), Cov(u_i,g_b)).  input="gauss" replaces the pixels by Gaussian samples
with the same mean and covariance (surrogate control, experiment B).

The weight update, init and clip order are copied from hebb/mnist/model.py (sibling) so
that rule="linear", input="pixels" reproduces its `pixels` lift exactly.
"""
from __future__ import annotations

import gzip
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

MNIST_DIR = Path(os.environ.get("MNIST_DIR", "/var/tmp/hebb-mnist/data/mnist"))


@dataclass
class Cfg:
    rule: str = "rect"          # rect | linear
    tau: float = 0.0            # threshold in units of each neuron's own sd
    input: str = "pixels"       # pixels | gauss
    res: int = 14               # 14 (B=196) | 28 (B=784)
    N: int = 980
    theta: float = 0.3
    wmin: float = 1e-3
    gamma: float = 1.0
    lam: float = 1.0
    eta: float = 1.0
    dt: float = 0.1
    steps: int = 3000
    batch: int = 256
    seed: int = 0
    ii_scale: float = 0.2
    ib_scale: float = 0.1
    n_snap: int = 24
    n_sub: int = 256
    avg_frac: float = 0.2
    probes: bool = True
    readout: bool = True
    nls_iters: int = 300
    c: float | None = None      # rate gain; None -> computed from pixels (rect only)
    extra: dict = field(default_factory=dict)

    @property
    def B(self):
        return self.res * self.res

    def to_dict(self):
        d = asdict(self); d["B"] = self.B; return d


def run_name(c: Cfg) -> str:
    r = "lin" if c.rule == "linear" else f"rect_t{c.tau:g}"
    g = "" if c.gamma == 1.0 else f"_g{c.gamma:g}"
    return f"{r}_{c.input}{c.res}_N{c.N}_th{c.theta:g}{g}_s{c.seed}"


# ---------------------------------------------------------------- data
def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load(split="train", res=14, root=None):
    """(x float32 (n, res^2) in [0,1], y int64).  res 14 = 2x2 average pool (sibling's)."""
    root = Path(root or MNIST_DIR)
    pre = "train" if split == "train" else "t10k"
    imgs = _read_idx(root / f"{pre}-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    y = _read_idx(root / f"{pre}-labels-idx1-ubyte.gz").astype(np.int64)
    if res == 14:
        x = imgs.reshape(-1, 14, 2, 14, 2).mean((2, 4)).reshape(-1, 196)
    else:
        x = imgs.reshape(-1, 784)
    return x, y


class Data:
    """Scaled pixels g = s * x (mean pixel variance 1), exact population moments, rate gain."""

    def __init__(self, res=14, device="cpu", dtype=torch.float32):
        xtr, ytr = load("train", res); xte, yte = load("test", res)
        self.scale = float(1.0 / np.sqrt(xtr.astype(np.float64).var(0).mean()))
        g = xtr.astype(np.float64) * self.scale
        self.mu = g.mean(0); self.Sigma = np.cov(g.T, bias=True)
        self.sd = np.sqrt(np.maximum(np.diag(self.Sigma), 0))
        self.ytr, self.yte = ytr, yte
        self.device, self.dtype = device, dtype
        self.Gtr = torch.as_tensor(xtr * self.scale, device=device, dtype=dtype)
        self.Gte = torch.as_tensor(xte * self.scale, device=device, dtype=dtype)
        self.mu_t = torch.as_tensor(self.mu, device=device, dtype=dtype)
        self.Sigma_t = torch.as_tensor(self.Sigma, device=device, dtype=dtype)
        self.sd_t = torch.as_tensor(self.sd, device=device, dtype=dtype)
        ev, V = np.linalg.eigh(self.Sigma)
        self.L_t = torch.as_tensor(V * np.sqrt(np.maximum(ev, 0)), device=device, dtype=dtype)

    def pixel_rate_gain(self, tau):
        """c such that mean_b Var(c relu(g_b - mu_b - tau sd_b)) = 1 on the training pixels."""
        s = torch.relu(self.Gtr - self.mu_t - tau * self.sd_t).double()
        return float(1.0 / np.sqrt(float(s.var(0, unbiased=False).mean())))

    def sample(self, kind, m, gen):
        if kind == "pixels":
            idx = torch.randint(0, self.Gtr.shape[0], (m,), device=self.device, generator=gen)
            return self.Gtr[idx]
        z = torch.randn(m, self.L_t.shape[1], device=self.device, dtype=self.dtype, generator=gen)
        return self.mu_t + z @ self.L_t.T


# ---------------------------------------------------------------- network
def init_weights(cfg: Cfg, device, dtype):
    """Identical to hebb/mnist/model.py init_weights (same RNG stream)."""
    rng = np.random.default_rng(cfg.seed)
    N, B = cfg.N, cfg.B
    _ = np.triu(rng.random((N, N)) < 1.0, 1)
    _ = rng.random((N, B))
    Wii = rng.uniform(0, 1, (N, N)) * cfg.ii_scale
    Wii = np.triu(Wii, 1); Wii = Wii + Wii.T
    Wib = rng.exponential(cfg.ib_scale, (N, B))
    return (torch.as_tensor(Wii, device=device, dtype=dtype),
            torch.as_tensor(Wib, device=device, dtype=dtype))


def harmonic(Wii, Wib):
    deg = Wii.sum(1) + Wib.sum(1)
    L = torch.diag(deg) - Wii
    Lc, info = torch.linalg.cholesky_ex(L)
    if int(info) != 0:
        return torch.linalg.solve(L, Wib)
    return torch.cholesky_solve(Wib, Lc)


def clip(Wii, Wib, wmin):
    Wii = Wii.clamp_min(wmin); Wii.fill_diagonal_(0.0)
    return Wii, Wib.clamp_min(wmin)


def thresholds(H, D: Data, tau):
    """Interior b = H mu + tau sqrt(h^T Sigma h), boundary b = mu + tau sd."""
    var = ((H @ D.Sigma_t) * H).sum(1).clamp_min(0)
    return H @ D.mu_t + tau * var.sqrt(), D.mu_t + tau * D.sd_t


def responses(H, G, D: Data, cfg: Cfg, c):
    """Signals the rule correlates: (interior (m,N), boundary (m,B))."""
    U = G @ H.T
    if cfg.rule == "linear":
        return U, G
    bi, bb = thresholds(H, D, cfg.tau)
    return c * torch.relu(U - bi), c * torch.relu(G - bb)


def covs_batch(H, G, D, cfg, c):
    Ri, Sb = responses(H, G, D, cfg, c)
    Ri = Ri - Ri.mean(0, keepdim=True); Sb = Sb - Sb.mean(0, keepdim=True)
    m = G.shape[0]
    return (Ri.T @ Ri) / (m - 1), (Ri.T @ Sb) / (m - 1)


def covs_expected(H, D: Data, cfg: Cfg, c, chunk=20000):
    """Exact expected covariances driving the rule:
    linear: H Sigma H^T, H Sigma (population);  rect+pixels: over the full 60k training set;
    rect+gauss: closed form (dual activation)."""
    if cfg.rule == "linear":
        HS = H @ D.Sigma_t
        return HS @ H.T, HS
    if cfg.input == "gauss":
        from hebb.nonlin.readout import predict, _Tab
        tab = getattr(D, "_tab", {}).get(cfg.tau)
        if tab is None:
            tab = _Tab(cfg.tau, device=H.device, dtype=torch.float64)
            D._tab = {**getattr(D, "_tab", {}), cfg.tau: tab}
        Pii, Pib = predict(H.double(), D.Sigma_t.double(), tab)
        return (c * c * Pii).to(H.dtype), (c * c * Pib).to(H.dtype)
    n = D.Gtr.shape[0]
    N, B = H.shape
    s1 = torch.zeros(N, dtype=torch.float64, device=H.device); s2 = torch.zeros(B, dtype=torch.float64, device=H.device)
    Sii = torch.zeros(N, N, dtype=H.dtype, device=H.device); Sib = torch.zeros(N, B, dtype=H.dtype, device=H.device)
    for i in range(0, n, chunk):
        Ri, Sb = responses(H, D.Gtr[i:i + chunk], D, cfg, c)
        s1 += Ri.double().sum(0); s2 += Sb.double().sum(0)
        Sii += Ri.T @ Ri; Sib += Ri.T @ Sb
    m1, m2 = s1 / n, s2 / n
    Cii = Sii.double() / n - torch.outer(m1, m1); Cib = Sib.double() / n - torch.outer(m1, m2)
    return Cii.to(H.dtype), Cib.to(H.dtype)


def step(Wii, Wib, cfg: Cfg, D: Data, G, c):
    """One Euler step; same order as hebb.core.step / hebb.mnist.model.step."""
    Wii, Wib = clip(Wii, Wib, cfg.wmin)
    H = harmonic(Wii, Wib)
    Cii, Cib = covs_batch(H, G, D, cfg, c)
    w0 = -cfg.eta * cfg.theta
    Wii = Wii + cfg.dt * (w0 - cfg.lam * Wii + cfg.gamma * cfg.eta * Cii)
    Wib = Wib + cfg.dt * (w0 - cfg.lam * Wib + cfg.eta * Cib)
    Wii, Wib = clip(Wii, Wib, cfg.wmin)
    return Wii, Wib, H


def residual(Wii, Wib, D, cfg, c):
    """Projected stationarity residual of the EXPECTED drift.  Edges with w <= 2 w_min (the
    readout's "inactive" cutoff) and negative drift count 0: time-averaged SGD weights leave
    edges that are at the floor most of the time slightly above it, and counting their full
    drift (~ -theta) would report the floor-toggling noise, not a failure of stationarity.
    Also returns res_max_strict with the strict floor tolerance w_min(1+1e-6).
    Returns (stats dict, Cii, Cib, H)."""
    H = harmonic(Wii, Wib)
    Cii, Cib = covs_expected(H, D, cfg, c)
    w0 = -cfg.eta * cfg.theta
    dii = w0 - cfg.lam * Wii + cfg.gamma * cfg.eta * Cii
    dib = w0 - cfg.lam * Wib + cfg.eta * Cib
    dii.fill_diagonal_(0.0)
    strict = []
    for tol in (cfg.wmin * (1 + 1e-6), 2 * cfg.wmin):
        pii = torch.where((Wii <= tol) & (dii < 0), torch.zeros_like(dii), dii)
        pib = torch.where((Wib <= tol) & (dib < 0), torch.zeros_like(dib), dib)
        strict.append(max(float(pii.abs().max()), float(pib.abs().max())))
    dii = torch.where((Wii <= 2 * cfg.wmin) & (dii < 0), torch.zeros_like(dii), dii)
    dib = torch.where((Wib <= 2 * cfg.wmin) & (dib < 0), torch.zeros_like(dib), dib)
    a = torch.cat([dii.abs().flatten(), dib.abs().flatten()])
    act = torch.cat([(Wii > 2 * cfg.wmin).flatten(), (Wib > 2 * cfg.wmin).flatten()])
    wmax = max(float(Wib.max()), float(Wii.max()), 1e-12)
    st = dict(res_max=float(a.max()), res_mean=float(a.mean()),
              res_active_mean=float(a[act].mean()) if bool(act.any()) else 0.0,
              res_rel=float(a.max()) / wmax, w_max=wmax, res_max_strict=strict[0])
    return st, Cii, Cib, H
