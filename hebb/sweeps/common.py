"""Shared utilities for the experimental sweeps (owned by the sweeps; imports hebb.core, never edits it).

Conventions: lam = eta = 1 unless stated (they only set weight scale and time), so weights are
in covariance units and theta = -w0/eta. Lateral (I-I) edges use the SAME rho(w) = w0 - lam*w as
I-B edges and Hebbian gain gamma*eta, exactly as hebb.core.step does; hence the lateral
threshold is theta/gamma (see findings.md, F0).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import json
import pathlib
import time

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from hebb import core
from hebb.core import Params, harmonic, init_state

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "data" / "sweeps"
RAW = ROOT / "scratch" / "out" / "sweeps"   # gitignored scratch for big intermediate files


# ----------------------------------------------------------------------------- covariances
def cov_block(sizes, rho_w=0.5, rho_a=0.1, var=None):
    """Blocks of the given sizes, within-block correlation rho_w (scalar or per block),
    across-block correlation rho_a; optional per-block variance multipliers var
    (covariance = D C D with D = sqrt(var) per block)."""
    sizes = list(sizes)
    B = sum(sizes)
    rw = np.broadcast_to(np.asarray(rho_w, float), (len(sizes),))
    C = np.full((B, B), float(rho_a))
    lab = np.repeat(np.arange(len(sizes)), sizes)
    off = 0
    for k, m in enumerate(sizes):
        C[off:off + m, off:off + m] = rw[k]
        off += m
    np.fill_diagonal(C, 1.0)
    v = np.ones(len(sizes)) if var is None else np.asarray(var, float)
    d = np.sqrt(v[lab])
    return C * np.outer(d, d), lab


def cov_hier(levels=(0.6, 0.3, 0.05), branching=(2, 2, 2)):
    """Hierarchical (ultrametric) correlation: B = prod(branching) leaves; two leaves whose
    lowest common ancestor is at depth d (0 = siblings) have correlation levels[d]."""
    B = int(np.prod(branching))
    idx = np.array(list(itertools.product(*[range(b) for b in reversed(branching)])))[:, ::-1]
    # idx[:, 0] = finest digit
    C = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            if i == j:
                C[i, j] = 1.0
                continue
            diff = np.nonzero(idx[i] != idx[j])[0]
            C[i, j] = levels[diff.max()]
    return C


def cov_ring(B=24, ell=2.0):
    """Stationary covariance on a ring: Sigma_bc = exp(-d(b,c)^2 / 2 ell^2). All correlations
    positive; eigenvectors are Fourier modes (signed); spectrum decays like a Gaussian."""
    d = np.abs(np.subtract.outer(np.arange(B), np.arange(B)))
    d = np.minimum(d, B - d)
    return np.exp(-d ** 2 / (2 * ell ** 2))


def cov_powerlaw(B=16, alpha=1.0, seed=0, normalize=True):
    """Random rotation of a power-law spectrum k^-alpha (has negative correlations).
    Normalized to unit mean variance (trace = B)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((B, B)))
    ev = np.arange(1, B + 1, dtype=float) ** (-alpha)
    S = (Q * ev) @ Q.T
    if normalize:
        S *= B / np.trace(S)
    return S


def cov_powerlaw_pos(B=16, alpha=1.0, seed=0, K=None):
    """Nonnegative-correlation covariance with power-law-ish spectrum: Sigma = A diag(s) A^T + I*eps,
    A nonnegative random sparse loadings (each factor loads a random subset), factor
    variances s_k = k^-alpha. Normalized to unit mean variance."""
    rng = np.random.default_rng(seed)
    K = K or B
    A = (rng.random((B, K)) < 0.25) * rng.random((B, K))
    s = np.arange(1, K + 1, dtype=float) ** (-alpha)
    S = (A * s) @ A.T + 0.05 * np.eye(B)
    S *= B / np.trace(S)
    return S


# ----------------------------------------------------------------------------- simulation
def simulate(Sigma, N, theta, gamma=1.0, wmin=1e-3, seed=0, p=1.0, ii_scale=0.0, ib_scale=0.2,
             dt=0.05, Tmax=20000, tol=1e-5, batch=None, check=200, state=None, lam=1.0, eta=1.0,
             Tmin=0, record=None, sample_seed=None):
    """Run hebb.core.step until max|dW/dt| < tol (mean-field) or for Tmax steps (stochastic,
    batch=int: minibatch covariance from `batch` Gaussian samples per step).
    record: optional int, store H every `record` steps."""
    B = Sigma.shape[0]
    rng = np.random.default_rng(seed)
    if state is None:
        Wii, Wib, Mii, Mib = init_state(N, B, rng, p, ii_scale, ib_scale)
    else:
        Wii, Wib, Mii, Mib = [x.copy() for x in state]
    P = Params(theta=theta, lam=lam, eta=eta, gamma=gamma, wmin=wmin, dt=dt)
    srng = np.random.default_rng(sample_seed if sample_seed is not None else seed + 10_000)
    Lc = np.linalg.cholesky(Sigma + 1e-12 * np.eye(B)) if batch else None
    hist, t, conv, rate = [], 0, False, np.inf
    while t < Tmax:
        Wii0, Wib0 = Wii, Wib
        for _ in range(check):
            if batch:
                g = srng.standard_normal((batch, B)) @ Lc.T
                g -= g.mean(0)
                S = g.T @ g / (batch - 1)
            else:
                S = Sigma
            Wii, Wib, _ = core.step(Wii, Wib, Mii, Mib, S, P)
            t += 1
            if record and t % record == 0:
                hist.append(harmonic(Wii, Wib))
        rate = max(np.abs(Wii - Wii0).max(), np.abs(Wib - Wib0).max()) / (check * dt)
        if not batch and t >= Tmin and rate < tol:
            conv = True
            break
    H = harmonic(Wii, Wib)
    res = residual(Wii, Wib, Mii, Mib, Sigma, theta, gamma, wmin, lam, eta)
    return dict(Wii=Wii, Wib=Wib, Mii=Mii, Mib=Mib, H=H, steps=t, converged=conv, rate=rate,
                res=res, hist=hist)


def residual(Wii, Wib, Mii, Mib, Sigma, theta, gamma=1.0, wmin=1e-3, lam=1.0, eta=1.0):
    """SPEC §1 projected stationarity residual max|dw/dt| (floor edges with negative drift -> 0)."""
    H = harmonic(Wii, Wib)
    HS = H @ Sigma
    dii = (-eta * theta - lam * Wii + gamma * eta * (HS @ H.T)) * Mii
    np.fill_diagonal(dii, 0)
    dib = (-eta * theta - lam * Wib + eta * HS) * Mib
    dii = np.where((Wii <= wmin * (1 + 1e-9)) & (dii < 0), 0, dii)
    dib = np.where((Wib <= wmin * (1 + 1e-9)) & (dib < 0), 0, dib)
    return float(max(np.abs(dii).max(), np.abs(dib).max()))


# ----------------------------------------------------------------------------- analysis
def cluster_embed(H, tol=0.1, method="average"):
    """Cluster embeddings by average-linkage on L1 distance, cut at tol. Returns labels 0..K-1
    sorted by decreasing size, sizes, centroids."""
    if len(H) == 1:
        return np.zeros(1, int), np.array([1]), H.copy()
    Z = linkage(H, method=method, metric="cityblock")
    lab = fcluster(Z, t=tol, criterion="distance") - 1
    ks, cnt = np.unique(lab, return_counts=True)
    order = ks[np.argsort(-cnt)]
    remap = np.empty(lab.max() + 1, int)
    remap[order] = np.arange(len(order))
    lab = remap[lab]
    sizes = np.bincount(lab)
    cents = np.array([H[lab == k].mean(0) for k in range(len(sizes))])
    return lab, sizes, cents


def n_major(sizes, N, frac=0.02):
    return int((sizes >= max(2, frac * N)).sum())


def fp_residual(h, Sigma, theta):
    """L1 distance between h and F(h) = (Sigma h - theta)_+ / sum."""
    r = np.maximum(Sigma @ h - theta, 0)
    if r.sum() <= 0:
        return np.inf
    return np.abs(h - r / r.sum()).sum()


def enumerate_parts(Sigma, theta, maxB=16, tol=1e-9, slack=0.0):
    """All fixed points of the single-neuron map h = (Sigma h - theta)_+/norm, found by
    enumerating supports S: h_S is a positive eigenvector of A = Sigma_SS - theta*11^T with
    eigenvalue c > 0, off-support (Sigma h)_b < theta. Stable iff c is the top eigenvalue
    of A (strictly) and the off-support inequality is strict (derivation in findings.md F1).
    Returns list of dicts."""
    B = Sigma.shape[0]
    assert B <= maxB
    out = []
    for r in range(1, B + 1):
        for S in itertools.combinations(range(B), r):
            S = list(S)
            A = Sigma[np.ix_(S, S)] - theta
            ev, V = np.linalg.eigh(A)
            for k in range(r):
                v = V[:, k]
                if v.sum() < 0:
                    v = -v
                if v.min() <= tol or ev[k] <= tol:
                    continue
                h = np.zeros(B)
                h[S] = v / v.sum()
                off = np.setdiff1d(np.arange(B), S)
                margin = theta - (Sigma[off] @ h).max() if len(off) else np.inf
                if margin < -slack:
                    continue
                gap = ev[k] - (np.delete(ev, k).max() if r > 1 else -np.inf)
                stable = (k == r - 1) and gap > tol and margin > tol
                out.append(dict(S=S, h=h, c=float(ev[k]), E=float(h @ Sigma @ h), margin=float(margin),
                                gap=float(gap), stable=bool(stable)))
    return out


def single_neuron_parts(Sigma, theta, M=3000, seed=0, ib_scale=0.2, wmin=1e-6, dt=0.05, T=6000,
                        init=None, ctol=0.05):
    """Gamma=0 / no-lateral limit: M independent neurons, each w_ib' = (Sigma h - theta)_b - w_b,
    h = w/sum(w). Returns endpoints H, cluster labels, sizes (basin fractions), centroids."""
    rng = np.random.default_rng(seed)
    B = Sigma.shape[0]
    W = rng.exponential(ib_scale, (M, B)) if init is None else init.copy()
    for t in range(T):
        W = np.maximum(W, wmin)
        H = W / W.sum(1, keepdims=True)
        W = W + dt * (H @ Sigma - theta - W)
    W = np.maximum(W, wmin)
    H = W / W.sum(1, keepdims=True)
    lab, sizes, cents = cluster_embed(H, tol=ctol) if M <= 4000 else _greedy(H, ctol)
    return H, lab, sizes, cents


def _greedy(H, tol):
    cents, lab = core.clusters(H, tol)
    sizes = np.bincount(lab)
    o = np.argsort(-sizes)
    remap = np.empty(len(o), int); remap[o] = np.arange(len(o))
    lab = remap[lab]
    return lab, np.bincount(lab), np.array([H[lab == k].mean(0) for k in range(len(o))])


def match_parts(cents, parts, tol=0.1):
    """Index of the nearest enumerated part (L1) for each centroid, or -1 if none within tol."""
    if not parts:
        return np.full(len(cents), -1), np.full(len(cents), np.inf)
    P = np.array([p["h"] for p in parts])
    D = np.abs(cents[:, None, :] - P[None]).sum(-1)
    j = D.argmin(1)
    d = D[np.arange(len(cents)), j]
    return np.where(d < tol, j, -1), d


def within_spread(H, lab, sizes):
    """Mean over clusters (size>=2) of mean L1 distance of members to their centroid."""
    out = []
    for k, n in enumerate(sizes):
        if n >= 2:
            X = H[lab == k]
            out.append(np.abs(X - X.mean(0)).sum(1).mean())
    return float(np.mean(out)) if out else float("nan")


# ----------------------------------------------------------------------------- readout (SPEC §3)
def _sym_basis(B):
    iu = np.triu_indices(B)
    return iu


def readout(Wii, Wib, theta=None, lam=1.0, eta=1.0, gamma=1.0, wmin=1e-3, use=("ii", "ib"),
            ridge=1e-6, active_factor=1.5, fit_theta=True):
    """Masked least-squares readout of C from weights alone.
    Active I-B edge (w > active_factor*wmin):  (H C)_ib = theta + (lam/eta) w_ib
    Active I-I edge:                           h_i^T C h_j = (theta + (lam/eta) w_ij)/gamma
    Unknown theta is handled by fitting it jointly (it is identified only together with the
    1 1^T direction of C, so we fix C's component along 11^T to zero and fit theta as a free
    offset) when fit_theta=True. Returns C_hat (B,B) symmetric."""
    N, B = Wib.shape
    H = harmonic(np.maximum(Wii, wmin * (Wii > 0)), Wib)
    iu = np.triu_indices(B)
    nb = len(iu[0])
    rows, rhs = [], []

    def coef(a, b):  # vector over upper-tri params for a^T C b with C symmetric
        M = np.outer(a, b)
        M = M + M.T - np.diag(np.diag(M))
        return M[iu]

    if "ib" in use:
        I, Bk = np.nonzero(Wib > active_factor * wmin)
        for i, b in zip(I, Bk):
            e = np.zeros(B); e[b] = 1.0
            rows.append(np.r_[coef(H[i], e), -1.0 / 1.0])
            rhs.append(lam / eta * Wib[i, b])
    if "ii" in use:
        I, J = np.nonzero(np.triu(Wii > active_factor * wmin, 1))
        for i, j in zip(I, J):
            rows.append(np.r_[gamma * coef(H[i], H[j]), -1.0])
            rhs.append(lam / eta * Wii[i, j])
    if not rows:
        return np.zeros((B, B)), dict(n_eq=0)
    X = np.array(rows); y = np.array(rhs)
    if not fit_theta:
        y = y + theta * (-X[:, -1])
        X = X[:, :-1]
    # remove 1 1^T direction ambiguity: add a soft constraint 1^T C 1 = 0 when fitting theta
    if fit_theta:
        X = np.vstack([X, np.r_[coef(np.ones(B), np.ones(B)), 0.0] * 10])
        y = np.r_[y, 0.0]
    XtX = X.T @ X
    sol = np.linalg.solve(XtX + ridge * np.trace(XtX) / XtX.shape[0] * np.eye(XtX.shape[0]), X.T @ y)
    c = sol[:nb]
    C = np.zeros((B, B)); C[iu] = c; C = C + C.T - np.diag(np.diag(C))
    info = dict(n_eq=len(rhs), theta_hat=float(sol[-1]) if fit_theta else theta)
    return C, info


def naive_readout(Wii, Wib, theta, lam=1.0, eta=1.0, gamma=1.0, wmin=1e-3):
    """The note's formula C = H^+ R H^+T with R_ij = theta + (lam/eta) w_ij / gamma on ALL I-I
    pairs (clipped ones included, diagonal filled with the row mean of R as a guess)."""
    H = harmonic(np.maximum(Wii, wmin * (Wii > 0)), Wib)
    R = (theta + lam / eta * Wii) / gamma
    np.fill_diagonal(R, R.sum(1) / max(1, len(R) - 1))
    Hp = np.linalg.pinv(H, rcond=1e-6)
    return Hp @ R @ Hp.T


def center(C):
    B = len(C)
    P = np.eye(B) - np.ones((B, B)) / B
    return P @ C @ P


def psd(C):
    ev, V = np.linalg.eigh((C + C.T) / 2)
    return (V * np.maximum(ev, 0)) @ V.T


def _sqrtm_psd(C):
    ev, V = np.linalg.eigh((C + C.T) / 2)
    return (V * np.sqrt(np.maximum(ev, 0))) @ V.T


def bures(A, Bm):
    sA = _sqrtm_psd(A)
    M = _sqrtm_psd(sA @ Bm @ sA)
    return float(np.sqrt(max(np.trace(A) + np.trace(Bm) - 2 * np.trace(M), 0)))


def fidelity(A, Bm):
    sB = _sqrtm_psd(Bm)
    return float(np.trace(_sqrtm_psd(sB @ A @ sB)))


def bw_norm(A, Bm, fit_scale=True):
    """SPEC §7.4: BW(A,B)^2/tr(B), A PSD-clipped first; with fit_scale, A is scaled to minimise it
    (closed form: 1 - F^2/(trA trB), F = tr sqrt(sqrtB A sqrtB)). All-zero A -> 1."""
    ev = np.linalg.eigvalsh((A + A.T) / 2)
    neg = float(-ev[ev < 0].sum() / (np.abs(ev).sum() + 1e-300))
    Ap = psd(A)
    ta, tb = np.trace(Ap), np.trace(Bm)
    if ta <= 1e-12 * max(tb, 1e-300):
        return 1.0, neg
    if fit_scale:
        F = fidelity(Ap, Bm)
        return float(max(0.0, 1 - F * F / (ta * tb))), neg
    return float(bures(Ap, Bm) ** 2 / tb), neg


def score3(Chat_known, Chat_free, Sigma, k=3):
    """The three SPEC §7.3 settings.
    known: readout with true theta/scale, scored on the full covariance, no projection.
    scale: same readout, best positive scale fitted (BW scale-fitted, cosine).
    contrast: readout with theta fitted; scored on P C P (contrast subspace, P = I - 11^T/B)."""
    out = {}
    A, S = Chat_known, Sigma
    relF = float(np.linalg.norm(A - S) / np.linalg.norm(S))
    bwk, neg = bw_norm(A, S, fit_scale=False)
    out["known"] = dict(relF=relF, bw=bwk, neg=neg, cos=_cos(A, S), angle=_angle(A, S, k))
    bws, _ = bw_norm(A, S, fit_scale=True)
    out["scale"] = dict(bw=bws, cos=_cos(A, S))
    Ac, Sc = center(Chat_free), center(Sigma)
    bwc, negc = bw_norm(Ac, Sc, fit_scale=True)
    out["contrast"] = dict(bw=bwc, neg=negc, cos=_cos(Ac, Sc), angle=_angle(Ac, Sc, k))
    return out


def _cos(A, S):
    return float((A * S).sum() / (np.linalg.norm(A) * np.linalg.norm(S) + 1e-300))


def _angle(A, S, k):
    ea, Va = np.linalg.eigh((A + A.T) / 2); es, Vs = np.linalg.eigh(S)
    return float((np.linalg.svd(Va[:, -k:].T @ Vs[:, -k:], compute_uv=False) ** 2).mean())


def metrics(Chat, Sigma, k=3):
    """SPEC §3 metrics after projecting out 1 and fitting the best scale.
    cos: Frobenius cosine of centered matrices (scale-free).
    bw: Bures-Wasserstein between PSD-projected centered matrices, each scaled to unit trace
        (so 0 = identical up to scale, max sqrt(2)); reported / sqrt(2) in [0,1].
    angle: mean cos^2 of principal angles between top-k eigenspaces."""
    A, S = center(Chat), center(Sigma)
    cos = float((A * S).sum() / (np.linalg.norm(A) * np.linalg.norm(S) + 1e-300))
    Ap, Sp = psd(A), psd(S)
    ta, ts = np.trace(Ap), np.trace(Sp)
    bw = bures(Ap / ta, Sp / ts) / np.sqrt(2) if ta > 1e-12 else 1.0
    ea, Va = np.linalg.eigh(A); es, Vs = np.linalg.eigh(S)
    Ua, Us = Va[:, -k:], Vs[:, -k:]
    ang = float((np.linalg.svd(Ua.T @ Us, compute_uv=False) ** 2).mean())
    return dict(cos=cos, bw=float(bw), angle=ang)


# ----------------------------------------------------------------------------- io
def save_json(name, obj, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    obj = dict(meta=meta, **obj)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(_clean(obj), separators=(",", ":")))
    return path


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if not np.isfinite(v):
            return None
        return float(f"{v:.4g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def pmap(fn, jobs, procs=3):
    """Multiprocessing map (fork) over a list of kwargs dicts or args."""
    import multiprocessing as mp
    with mp.get_context("fork").Pool(procs) as pool:
        return pool.map(fn, jobs, chunksize=1)


class Timer:
    def __init__(self, tag):
        self.tag = tag

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *a):
        print(f"[{self.tag}] {time.time() - self.t:.1f}s", flush=True)
