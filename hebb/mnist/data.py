"""MNIST loading and kernel lifts g = phi(x) for the Hebbian resistor network.

Stimulus x: MNIST digit, 2x2 average-pooled to 14x14 (196 values in [0, 1]).

Lifts (all deterministic given `seed`):
  pixels     g = x                                   B = 196, linear kernel
  rff        g = sqrt(2/D) cos(Omega x + b)          B = D,  Gaussian kernel, bandwidth
             sigma = bw_mult * median ||x - x'|| (median heuristic, bw_mult in {0.5, 1, 2})
  relu       g = relu(V x + c), V rows ~ N(0, I/196), c ~ N(0, 1/196 * E|x|^2) ... see make_lift
             (random ReLU features = arc-cosine kernel of degree 1; nonnegative features)

Scaling: every lift is multiplied by ONE global constant so that the mean feature
variance is 1 (tr Sigma_g = B).  Covariance entries are then on a correlation-like scale and
theta is directly comparable to them.  A single constant (not per-feature
standardisation) keeps the kernel geometry: per-feature z-scoring would blow up the
near-constant border pixels and change which directions dominate.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np

MNIST_DIR = Path(os.environ.get("MNIST_DIR", "/var/tmp/hebb-mnist/data/mnist"))


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist(split: str = "train", root: Path | None = None):
    """Returns (x14 float32 (n,196) in [0,1], labels int64 (n,))."""
    root = Path(root or MNIST_DIR)
    pre = "train" if split == "train" else "t10k"
    imgs = _read_idx(root / f"{pre}-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    labels = _read_idx(root / f"{pre}-labels-idx1-ubyte.gz").astype(np.int64)
    x14 = imgs.reshape(-1, 14, 2, 14, 2).mean((2, 4)).reshape(-1, 196)
    return x14, labels


def median_distance(x: np.ndarray, n: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    s = x[rng.choice(len(x), n, replace=False)].astype(np.float64)
    sq = (s * s).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * s @ s.T, 0)
    iu = np.triu_indices(n, 1)
    return float(np.sqrt(np.median(d2[iu])))


class Lift:
    """phi(x) = scale * raw(x).  `apply` works on numpy or torch (same formula)."""

    def __init__(self, kind: str, B: int, seed: int = 0, bw_mult: float = 1.0, x_ref=None):
        self.kind, self.B, self.seed, self.bw_mult = kind, B, seed, bw_mult
        rng = np.random.default_rng(10_000 + seed)
        self.sigma = None
        if kind == "pixels":
            assert B == 196
        elif kind == "rff":
            self.sigma = bw_mult * median_distance(x_ref)
            self.Omega = (rng.standard_normal((196, B)) / self.sigma).astype(np.float32)
            self.bias = rng.uniform(0, 2 * np.pi, B).astype(np.float32)
        elif kind == "relu":
            # random ReLU features on the centred image: relu(v.(x - xbar) + c), v ~ N(0, I/196)
            self.xbar = x_ref.mean(0).astype(np.float32)
            V = rng.standard_normal((196, B)).astype(np.float32)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            self.Omega = V
            proj_sd = float(((x_ref[:5000] - self.xbar) @ V).std())
            self.bias = (rng.standard_normal(B) * 0.5 * proj_sd).astype(np.float32)
        else:
            raise ValueError(kind)
        self.scale = 1.0
        raw = self.raw(x_ref)
        self.scale = float(1.0 / np.sqrt(raw.var(0).mean()))

    def raw(self, x):
        if self.kind == "pixels":
            out = x
        elif self.kind == "rff":
            out = np.sqrt(2.0 / self.B) * np.cos(x @ self.Omega + self.bias)
        else:
            out = np.maximum((x - self.xbar) @ self.Omega + self.bias, 0)
        return out * self.scale if self.scale != 1.0 else out

    def apply_torch(self, x, torch):
        if self.kind == "pixels":
            return x * self.scale
        Om = torch.as_tensor(self.Omega, device=x.device, dtype=x.dtype)
        b = torch.as_tensor(self.bias, device=x.device, dtype=x.dtype)
        if self.kind == "rff":
            return self.scale * (2.0 / self.B) ** 0.5 * torch.cos(x @ Om + b)
        xb = torch.as_tensor(self.xbar, device=x.device, dtype=x.dtype)
        return self.scale * torch.relu((x - xb) @ Om + b)

    def describe(self) -> dict:
        d = dict(kind=self.kind, B=self.B, seed=self.seed, scale=self.scale)
        if self.kind == "rff":
            d.update(bw_mult=self.bw_mult, sigma=self.sigma)
        return d


def make_lift(kind: str, B: int, seed: int = 0, bw_mult: float = 1.0, x_train=None) -> Lift:
    if x_train is None:
        x_train, _ = load_mnist("train")
    return Lift(kind, B, seed, bw_mult, x_train)


def lifted_stats(lift: Lift, x: np.ndarray, chunk: int = 10000):
    """Exact mean and (population) covariance of g over x, float64."""
    B = lift.B
    s1 = np.zeros(B); s2 = np.zeros((B, B)); n = 0
    for i in range(0, len(x), chunk):
        g = lift.raw(x[i:i + chunk]).astype(np.float64)
        s1 += g.sum(0); s2 += g.T @ g; n += len(g)
    mu = s1 / n
    return mu, s2 / n - np.outer(mu, mu)


def offdiag_quantiles(S: np.ndarray, qs=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)):
    iu = np.triu_indices(len(S), 1)
    v = S[iu]
    return {"q": list(qs), "value": [float(np.quantile(v, q)) for q in qs],
            "frac_negative": float((v < 0).mean()), "mean": float(v.mean()),
            "diag_mean": float(np.diag(S).mean()), "diag_max": float(np.diag(S).max()),
            "max_offdiag": float(v.max())}
