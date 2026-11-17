"""Derived data for the Homeostasis chapter (site/data/homeo/covs.json).

For every input covariance offered in the chapter's live explorer this writes
  * Sigma itself (so the page and the sweeps use bit-identical matrices; the power-law ones come from
    numpy's RNG in hebb/sweeps/common.py and cannot be regenerated in JS),
  * the stable parts on a theta grid of step 0.01 (findings.md F1): nonnegative Perron vectors h of
    A_SS = (Sigma - theta 11^T)_SS with eigenvalue c > 0 that is strictly the top one, and
    (Sigma h)_b < theta strictly off the support S.

How the parts are found
  * B <= 16: exact enumeration of all 2^B - 1 supports (hebb.sweeps.common.enumerate_parts).
    At a grid value that sits exactly on a correlation level some parts are marginal (zero gap or
    zero margin); there the counts just below and just above (theta -/+ 1e-6) are stored as `tie`.
  * B > 16 (the MNIST covariance, B = 49): the lone-neuron flow dw/dt = (Sigma h - theta) - w (Euler dt 0.4, 1200 steps),
    h = w / sum(w), floor 1e-6, from 49 vertex starts + 500 random Exp(0.2) starts + 300 starts on
    random 3x3 patches; every distinct endpoint is then checked with the exact test above.  This finds
    every part with a non-negligible basin but is a lower bound in principle (flagged `method: flow`).

The MNIST covariance: train images (60k), 4x4 average-pooled 28x28 -> 7x7 (B = 49), covariance of the
pooled pixels (centred), multiplied by ONE constant so that the mean variance is 1 (tr Sigma = B), the
same convention as hebb/mnist/data.py.  Raw files are read the way hebb/mnist/data.py reads them
(MNIST_DIR, default /var/tmp/hebb-mnist/data/mnist).

Run:  uv run python -m hebb.homeo_cov        (about 10 minutes on 4 processes)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from hebb.sweeps.common import (cov_block, cov_hier, cov_powerlaw, cov_powerlaw_pos, cov_ring,
                                enumerate_parts, pmap)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data" / "homeo"
STEP = 0.01


def mnist49():
    from hebb.mnist.data import MNIST_DIR, _read_idx
    imgs = _read_idx(MNIST_DIR / "train-images-idx3-ubyte.gz").astype(np.float64) / 255.0
    x = imgs.reshape(-1, 7, 4, 7, 4).mean((2, 4)).reshape(-1, 49)
    S = np.cov(x.T, bias=True)
    scale = 49.0 / np.trace(S)
    S *= scale
    mean = x.mean(0)
    ev, V = np.linalg.eigh(S)
    ev, V = ev[::-1], V[:, ::-1]
    for k in range(V.shape[1]):          # sign: largest |entry| positive
        if V[np.argmax(np.abs(V[:, k])), k] < 0:
            V[:, k] *= -1
    extra = dict(w=7, h=7, mean_image=np.round(mean, 4).tolist(), scale=float(scale), n_images=int(len(x)),
                 eigvals=np.round(ev, 5).tolist(), eigvecs=np.round(V[:, :8].T, 4).tolist())
    return S, extra


def presets():
    out = {}
    v6 = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    out["graded6"] = dict(S=np.diag(v6), label="graded variances (diagonal, B = 6)",
                          desc="independent inputs with variances 0.5, 0.75, 1, 1.5, 2, 3 (the allocation testbed of findings F6)",
                          groups=None)
    S, lab = cov_block([3, 3, 3, 3], 0.6, 0.0, var=[0.7, 1.0, 1.4, 2.0])
    out["planted_energy"] = dict(S=S, label="planted blocks, unequal energy (B = 12)",
                                 desc="4 blocks of 3, within-block correlation 0.6, across 0, block variances 0.7, 1, 1.4, 2",
                                 groups=lab.tolist())
    S, lab = cov_block([1, 2, 3, 6], 0.6, 0.0)
    out["planted_size"] = dict(S=S, label="planted blocks, unequal size (B = 12)",
                               desc="blocks of 1, 2, 3 and 6 inputs, unit variance, within-block correlation 0.6, across 0",
                               groups=lab.tolist())
    S, lab = cov_block([4, 4, 4], 0.5, 0.1)
    out["block12"] = dict(S=S, label="block12: 3 blocks of 4 (B = 12)",
                          desc="3 blocks of 4, unit variance, within 0.5, across 0.1 (the phase-diagram covariance)",
                          groups=lab.tolist())
    out["hier8"] = dict(S=cov_hier((0.6, 0.3, 0.05), (2, 2, 2)), label="hierarchical (B = 8)",
                        desc="ultrametric: siblings 0.6, cousins 0.3, halves 0.05, unit variance",
                        groups=[0, 0, 1, 1, 2, 2, 3, 3])
    out["ring16"] = dict(S=cov_ring(16, 1.5), label="ring (B = 16)",
                         desc="16 inputs on a ring, Sigma = exp(-d^2 / (2 * 1.5^2))", groups=None)
    out["powerlaw12"] = dict(S=cov_powerlaw(12, 1.0, seed=0), label="power law, signed (B = 12)",
                             desc="random rotation of the spectrum k^-1, seed 0, tr Sigma = B (signed correlations)",
                             groups=None)
    out["powerlaw_pos12"] = dict(S=cov_powerlaw_pos(12, 1.0, seed=0), label="power law, nonnegative (B = 12)",
                                 desc="sparse nonnegative factor model, factor variances k^-1, seed 0, tr Sigma = B",
                                 groups=None)
    S, extra = mnist49()
    out["mnist49"] = dict(S=S, label="MNIST pixels, 7×7 (B = 49, real)",
                          desc="covariance of MNIST training digits average-pooled to 7x7, scaled to unit mean variance",
                          groups=None, extra=extra, real=True)
    return out


def check_part(S, theta, h, tol=1e-9):
    """Exact stability test of a candidate endpoint h (findings F1). Returns the part dict or None."""
    B = len(h)
    sup = np.nonzero(h > 1e-4)[0]
    if len(sup) == 0:
        return None
    A = S[np.ix_(sup, sup)] - theta
    ev, V = np.linalg.eigh(A)
    v = V[:, -1] * np.sign(V[:, -1].sum())
    if v.min() <= tol or ev[-1] <= tol:
        return None
    gap = ev[-1] - ev[-2] if len(sup) > 1 else np.inf
    hh = np.zeros(B)
    hh[sup] = v / v.sum()
    off = np.setdiff1d(np.arange(B), sup)
    margin = theta - (S[off] @ hh).max() if len(off) else np.inf
    if gap <= tol or margin <= tol:
        return None
    return dict(S=sup.tolist(), h=hh, c=float(ev[-1]))


def flow_parts(S, theta, seed=0):
    B = S.shape[0]
    rng = np.random.default_rng(seed)
    starts = [np.eye(B) * 1.0 + rng.exponential(0.002, (B, B)), rng.exponential(0.2, (500, B))]
    side = int(round(np.sqrt(B)))
    P = np.full((300, B), 1e-6)
    for k in range(300):                      # random 3x3 patches (for the image covariance)
        r, c = rng.integers(0, side - 2, 2)
        for dr in range(3):
            for dc in range(3):
                P[k, (r + dr) * side + c + dc] = rng.exponential(0.2)
    starts.append(P)
    W = np.vstack(starts)
    dt = 0.4
    for _ in range(1200):
        W = np.maximum(W, 1e-6)
        H = W / W.sum(1, keepdims=True)
        W = W + dt * (H @ S - theta - W)
    W = np.maximum(W, 1e-6)
    H = W / W.sum(1, keepdims=True)
    found = {}
    for h in H:
        key = tuple(np.nonzero(h > 1e-4)[0])
        if key in found:
            continue
        p = check_part(S, theta, h)
        found[key] = p
    return [p for p in found.values() if p is not None]


def summarise(S, parts):
    rows = []
    for p in parts:
        sup = p["S"]
        h = p["h"]
        lam = float(np.linalg.eigvalsh(S[np.ix_(sup, sup)])[-1])
        rows.append(dict(S=sup, h=[round(float(h[b]), 4) for b in sup], c=round(p["c"], 5), lam=round(lam, 5),
                         E=round(float(h @ S @ h), 5)))
    rows.sort(key=lambda r: -r["lam"])
    return rows


def job(args):
    name, S, theta, method = args
    if method == "enum":
        allp = enumerate_parts(S, theta)
        P = [p for p in allp if p["stable"]]
        tie = None
        if any(abs(p["margin"]) < 1e-7 or abs(p["gap"]) < 1e-7 for p in allp):
            n_lo = sum(p["stable"] for p in enumerate_parts(S, theta - 1e-6))
            n_hi = sum(p["stable"] for p in enumerate_parts(S, theta + 1e-6))
            tie = [n_lo, n_hi] if (n_lo != len(P) or n_hi != len(P)) else None
    else:
        tie = None
        if theta == 0:   # pixels that never change have Sigma row 0: every part is marginal at theta = 0 exactly
            n_lo, n_hi = len(flow_parts(S, -1e-6)), len(flow_parts(S, 1e-6))
            tie = [n_lo, n_hi]
        P = flow_parts(S, theta)
    return name, theta, summarise(S, P), tie


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pres = presets()
    jobs = []
    for name, pr in pres.items():
        S = pr["S"]
        B = S.shape[0]
        hi = np.ceil((np.diag(S).max() + 0.05) / STEP) * STEP
        grid = [round(x, 2) for x in np.arange(-0.2, hi + 1e-9, STEP)]
        pr["grid"] = grid
        method = "enum" if B <= 16 else "flow"
        pr["method"] = method
        jobs += [(name, S, th, method) for th in grid]
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    res = pmap(job, jobs, procs=procs)
    table = {}
    for name, th, parts, tie in res:
        table.setdefault(name, {})[th] = (parts, tie)
    covs = {}
    for name, pr in pres.items():
        S = pr["S"]
        B = S.shape[0]
        off = S[~np.eye(B, dtype=bool)]
        d = np.diag(S)
        rows = []
        for th in pr["grid"]:
            parts, tie = table[name][th]
            r = dict(theta=th, n=len(parts), parts=parts)
            if tie:
                r["tie"] = tie
            rows.append(r)
        covs[name] = dict(
            label=pr["label"], desc=pr["desc"], B=B, real=bool(pr.get("real", False)), groups=pr["groups"],
            Sigma=[[round(float(x), 6) for x in row] for row in S],
            stats=dict(kappa0=round(float(S.sum() / B ** 2), 5), offdiag_min=round(float(off.min()), 5),
                       offdiag_max=round(float(off.max()), 5), var_min=round(float(d.min()), 5),
                       var_max=round(float(d.max()), 5)),
            method=pr["method"], parts=rows, **({"image": pr["extra"]} if "extra" in pr else {}))
    meta = dict(
        script="hebb/homeo_cov.py",
        description="Input covariances for the Homeostasis chapter's live explorer, and the stable parts of each "
                    "on a theta grid (step 0.01). Part: support S (input indices), h on S (sums to 1), c = top "
                    "eigenvalue of (Sigma - theta 11^T)_SS, lam = top eigenvalue of Sigma_SS, E = h^T Sigma h. "
                    "method enum = exact enumeration of supports (B <= 16; tie = counts at theta -/+ 1e-6 where a "
                    "part is marginal); flow = lone-neuron flow from 849 starts, endpoints checked exactly "
                    "(a lower bound). See findings.md F1 for the stability criterion.",
        step=STEP)
    path = OUT / "covs.json"
    path.write_text(json.dumps(dict(meta=meta, covs=covs), separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")
    for name, c in covs.items():
        ns = [r["n"] for r in c["parts"]]
        print(name, c["B"], c["method"], c["stats"], "n_stable range", min(ns), max(ns))


if __name__ == "__main__":
    main()
