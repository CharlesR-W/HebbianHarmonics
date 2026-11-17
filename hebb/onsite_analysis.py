"""Analysis helpers for the v2 sweeps (hebb/sweeps2). Imports hebb.onsite, never edits it.

Conventions (SPEC.md §3):
- Two input banks per seed: a TRAINING bank G (S rows) that defines the learning expectation
  (fixed-point identities are exact for its uncentred second moment C_hat = G^T G / S) and an
  independent EVALUATION bank used for every population claim.
- Seed s -> rng streams: network (attachments) [s, 0]; thresholds [s, 3]; training bank [s, 1];
  evaluation bank [s, 2]. Covariance presets with random structure use a fixed seed
  (SIGMA_SEED) so that seeds vary the network and the bank, not the input statistics.
- Equilibrium is claimed only when the projected residual max|dw/dt| < RES_TOL.
"""
import json
import pathlib
import time

import numpy as np

from hebb import onsite as o

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data" / "sweeps2"
RES_TOL = 1e-6          # equilibrium claims require projected residual below this
RUN_TOL = 1e-9          # integration stops when max|dW|/dt falls below this
SIGMA_SEED = 12345
S_TRAIN = 256
S_EVAL = 4096

# Default rectifying on-site potential (as in scratch/onsite_proto.py): stiff below threshold.
RECT = dict(klo=3.0, khi=0.0, sigma_t=0.7)
LINEAR = dict(klo=0.0, khi=0.0, sigma_t=0.0)


# ----------------------------------------------------------------------------- input statistics
def sigma_blocks(B=12, var=(2.0, 1.0, 0.5, 0.25), rho=0.8):
    """Four equal blocks; within-block correlation rho, block variances var, no cross-block
    correlation (the prototype's 'blocks' preset)."""
    nb = len(var)
    m = B // nb
    S = np.zeros((B, B))
    for k in range(nb):
        sl = slice(k * m, (k + 1) * m)
        S[sl, sl] = rho * var[k]
        S[range(k * m, (k + 1) * m), range(k * m, (k + 1) * m)] = var[k]
    return S


def sigma_power(B=12, exponent=1.0, seed=SIGMA_SEED):
    """Random eigenbasis, eigenvalues ~ a^-exponent, trace B."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((B, B)))
    lam = 1.0 / np.arange(1, B + 1) ** exponent
    lam *= B / lam.sum()
    return (Q * lam) @ Q.T


def contrast_basis(B):
    """Orthonormal B x B basis whose first column is 1/sqrt(B) and the rest are contrasts
    (normalised Hadamard for B a power of 2, else DCT-II)."""
    if B & (B - 1) == 0:
        H = np.array([[1.0]])
        while len(H) < B:
            H = np.block([[H, H], [H, -H]])
        return H.T / np.sqrt(B)
    n = np.arange(B)
    V = np.cos(np.pi * (n[:, None] + 0.5) * n[None, :] / B)
    return V / np.linalg.norm(V, axis=0)


def sigma_from_eig(V, lam):
    return (V * np.asarray(lam, float)) @ V.T


def eig_desc(S):
    lam, V = np.linalg.eigh(S)
    return lam[::-1], V[:, ::-1]


# ----------------------------------------------------------------------------- seeds and banks
def rng_for(seed, stream):
    return np.random.default_rng([int(seed), int(stream)])


def banks(Sigma, seed, S=S_TRAIN, S_eval=S_EVAL, kind="gauss"):
    """Training bank and an independent evaluation bank."""
    return (o.sample_inputs(Sigma, S, rng_for(seed, 1), kind),
            o.sample_inputs(Sigma, S_eval, rng_for(seed, 2), kind))


def make_net(N, B, seed, s=3, deg=1.0, klo=0.0, khi=0.0, sigma_t=0.0, A=None):
    """Attachments from stream [seed,0] and thresholds from [seed,3], so that arms differing
    only in the potential share attachments exactly."""
    if A is None:
        A = o.make_attachments(N, B, s, rng_for(seed, 0), deg)
    t = sigma_t * rng_for(seed, 3).standard_normal(N)
    return o.Net(A=A.copy(), t=t, klo=klo, khi=khi)


def block_attachments(N, B, nblocks, s, seed, deg=1.0, counts=None):
    """Each neuron attached to s distinct inputs inside ONE block (blocks of B/nblocks inputs);
    neurons assigned to blocks in equal numbers (or per `counts`). Dirichlet(1) shares."""
    rng = rng_for(seed, 0)
    m = B // nblocks
    counts = [N // nblocks] * nblocks if counts is None else list(counts)
    A = np.zeros((sum(counts), B))
    home = np.repeat(np.arange(nblocks), counts)
    for i, k in enumerate(home):
        idx = k * m + rng.choice(m, min(s, m), replace=False)
        A[i, idx] = rng.dirichlet(np.ones(len(idx))) * deg
    return A, home


# ----------------------------------------------------------------------------- solving
def default_dt(alpha):
    """Euler step for the slow flow. The Hessian of J is alpha*I plus a negative semidefinite
    part (E* is concave in W), so dt <= 1/alpha is stable at every stable fixed point."""
    return float(min(0.5, 1.0 / alpha))


def solve(net, G, b, alpha, dt=None, T=40000, tol=RUN_TOL):
    """Integrate the projected flow to stationarity from net.W (in place).
    Returns dict(U, residual, steps, converged) with converged = residual < RES_TOL."""
    t0 = time.time()
    r = o.run(net, G, b, alpha, dt=default_dt(alpha) if dt is None else dt, T=T, tol=tol)
    r["converged"] = bool(r["residual"] < RES_TOL)
    r["seconds"] = time.time() - t0
    return r


def fixed_point_check(W, U, b, alpha, eps=1e-9):
    """Deviation from the stated fixed point w = (b - D^2/2)_+/alpha on the training bank.
    Returns max over live edges of |alpha w - (b - D^2/2)| and min over dead pairs of
    (D^2 - 2b) (should be >= -2*residual)."""
    D2 = o.dist2(U)
    N = len(W)
    iu = np.triu_indices(N, 1)
    w, d2 = W[iu], D2[iu]
    live = w > eps
    dev_live = float(np.abs(alpha * w[live] - (b - d2[live] / 2)).max()) if live.any() else 0.0
    slack_dead = float((d2[~live] - 2 * b).min()) if (~live).any() else float("inf")
    return dict(dev_live=dev_live, slack_dead=slack_dead, n_live=int(live.sum()))


# ----------------------------------------------------------------------------- clusters
def cluster_summary(W, U, b, eps=1e-9):
    """Connected components of the live graph with their diameters on the responses U.
    diam_max / diam_mean: max / mean RMS distance D of members to the prototype (cluster_stats);
    span: max pairwise D inside the component. Singletons have zero diameter."""
    lab = o.components(W, eps)
    st = o.cluster_stats(U, lab)
    D = np.sqrt(np.clip(o.dist2(U), 0, None))
    sizes = np.array([c["size"] for c in st])
    span = np.array([D[np.ix_(c["members"], c["members"])].max() for c in st])
    big = sizes > 1
    return dict(
        lab=lab, K=int(len(st)), K_multi=int(big.sum()), sizes=sorted(sizes.tolist(), reverse=True),
        largest=int(sizes.max()),
        diam_max=float(max(c["diam_max"] for c in st)),
        diam_mean=float(np.mean([c["diam_mean"] for c in st if c["size"] > 1])) if big.any() else 0.0,
        span_max=float(span.max()),
        span_over_range=float(span.max() / np.sqrt(2 * b)),
        stats=st,
    )


def pair_violation(lab_fine, lab_coarse):
    """Nestedness check: fraction of pairs co-clustered in the FINE partition (smaller b) that
    are split in the COARSE partition (larger b). 0 means the fine partition refines the coarse."""
    same_f = lab_fine[:, None] == lab_fine[None, :]
    same_c = lab_coarse[:, None] == lab_coarse[None, :]
    off = ~np.eye(len(lab_fine), dtype=bool)
    n = (same_f & off).sum()
    return float(((same_f & ~same_c) & off).sum() / n) if n else 0.0


# ----------------------------------------------------------------------------- function fits
def affine_fit_err(Y_fit, X_fit, Y_test, X_test, ridge=1e-10):
    """Held-out relative MSE of an affine least-squares fit Y ~ c + X beta, per column of Y."""
    Xf = np.hstack([np.ones((len(X_fit), 1)), X_fit])
    Xt = np.hstack([np.ones((len(X_test), 1)), X_test])
    beta = np.linalg.solve(Xf.T @ Xf + ridge * np.eye(Xf.shape[1]), Xf.T @ Y_fit)
    res = Y_test - Xt @ beta
    var = Y_test.var(0)
    return np.where(var > 1e-14, (res ** 2).mean(0) / np.maximum(var, 1e-300), 0.0)


def spline_features(z, knots):
    """Piecewise-linear basis in a scalar z: 1, z, (z - k)_+ for each knot."""
    return np.hstack([np.ones((len(z), 1)), z[:, None], np.maximum(z[:, None] - knots[None], 0)])


def spline_fit_err(Y_fit, z_fit, Y_test, z_test, n_knots=24, ridge=1e-10):
    """Held-out relative MSE of a 1-D piecewise-linear fit Y ~ f(z) with knots at quantiles of
    z_fit (flexible nonlinear function of ONE scalar)."""
    knots = np.quantile(z_fit, np.linspace(0, 1, n_knots + 2)[1:-1])
    Ff, Ft = spline_features(z_fit, knots), spline_features(z_test, knots)
    beta = np.linalg.solve(Ff.T @ Ff + ridge * np.eye(Ff.shape[1]), Ff.T @ Y_fit)
    res = Y_test - Ft @ beta
    var = Y_test.var(0)
    return np.where(var > 1e-14, (res ** 2).mean(0) / np.maximum(var, 1e-300), 0.0)


def deff(C):
    tr, tr2 = np.trace(C), (C ** 2).sum()
    return float(tr ** 2 / tr2) if tr2 > 0 else 0.0


# ----------------------------------------------------------------------------- spectra
def water_fill(lam, total):
    """q_a = (lam_a - mu)_+ with sum q = total (mu may be negative). Exact by bisection."""
    lam = np.asarray(lam, float)
    lo, hi = lam.min() - total, lam.max()
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        if np.maximum(lam - mu, 0).sum() > total:
            lo = mu
        else:
            hi = mu
    mu = 0.5 * (lo + hi)
    return np.maximum(lam - mu, 0), mu


def psd_sqrt(M):
    lam, V = np.linalg.eigh((M + M.T) / 2)
    return (V * np.sqrt(np.clip(lam, 0, None))) @ V.T


def bures(A, Bm):
    """Bures-Wasserstein distance between PSD matrices."""
    rA = psd_sqrt(A)
    cross = np.trace(psd_sqrt(rA @ Bm @ rA))
    return float(np.sqrt(max(np.trace(A) + np.trace(Bm) - 2 * cross, 0.0)))


def fro_cos(A, Bm):
    return float((A * Bm).sum() / (np.linalg.norm(A) * np.linalg.norm(Bm) + 1e-300))


# ----------------------------------------------------------------------------- output
def _round(x, sig=5):
    if isinstance(x, dict):
        return {k: _round(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, sig) for v in x]
    if isinstance(x, np.ndarray):
        return _round(x.tolist(), sig)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not np.isfinite(x):
            return None if np.isnan(x) else ("inf" if x > 0 else "-inf")
        return float(f"{x:.{sig}g}")
    return x


def dump(name, meta, data, sig=5):
    """Write site/data/sweeps2/<name>.json = {meta, ...data}; meta must hold script,
    description, config, seeds and fields (name -> description with units)."""
    OUT.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta.setdefault("res_tol", RES_TOL)
    meta.setdefault("seed_streams", "network attachments [seed,0]; thresholds [seed,3]; training bank [seed,1]; "
                                    "evaluation bank [seed,2]; random covariance presets fixed seed %d" % SIGMA_SEED)
    obj = dict(meta=meta, **data)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(_round(obj, sig), separators=(",", ":")))
    kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(ROOT)} ({kb:.0f} kB)")
    return path


def pool_map(fn, items, procs=4):
    """Parallel map over independent runs (BLAS pinned to one thread by hebb.sweeps2)."""
    if procs <= 1:
        return [fn(x) for x in items]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(procs) as p:
        return p.map(fn, items, chunksize=1)
