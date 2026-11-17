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


# ═══════════════════════════════════════════════════════════════════════════════
# Additions mirroring site/js/engine.js (same names in snake_case; cross-checked by
# tests/test_engine_parity.py).  Everything below is additive; nothing above changed.
# Fixed point (SPEC §1/§7.1): I–B  (H Σ)_ib = θ + (λ/η) w_ib ;
#                             I–I  γ h_iᵀ Σ h_j = θ + (λ/η) w_ij   (threshold θ/γ).
# ═══════════════════════════════════════════════════════════════════════════════
import math

_M32 = 0xFFFFFFFF


class JsRng:
    """mulberry32, bit-identical to engine.js makeRng (uniform/normal/exponential/int)."""

    def __init__(self, seed=0):
        self.a = int(seed) & _M32

    def uniform(self):
        self.a = (self.a + 0x6D2B79F5) & _M32
        a = self.a
        t = ((a ^ (a >> 15)) * (1 | a)) & _M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & _M32)) & _M32) ^ t
        return ((t ^ (t >> 14)) & _M32) / 4294967296.0

    def normal(self):
        u1 = self.uniform(); u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)

    def exponential(self, scale=1.0):
        return -scale * math.log(1.0 - self.uniform())

    def int(self, n):
        return int(math.floor(self.uniform() * n))


def create_state(N, B, seed=0, p=1.0, ii_scale=0.0, ib_scale=0.2):
    """Same recipe as init_state, with the engine's PRNG and draw order (engine.createState)."""
    r = JsRng(seed)
    Mii = np.zeros((N, N)); Mib = np.zeros((N, B))
    for i in range(N):
        for j in range(i + 1, N):
            if r.uniform() < p:
                Mii[i, j] = Mii[j, i] = 1.0
    for i in range(N):
        for b in range(B):
            Mib[i, b] = 1.0 if r.uniform() < p else 0.0
    for i in range(N):
        if Mib[i].sum() == 0:
            Mib[i, r.int(B)] = 1.0
    Wii = np.zeros((N, N)); Wib = np.zeros((N, B))
    for i in range(N):
        for j in range(i + 1, N):
            Wii[i, j] = Wii[j, i] = r.uniform() * ii_scale * Mii[i, j]
    for i in range(N):
        for b in range(B):
            Wib[i, b] = r.exponential(ib_scale) * Mib[i, b]
    return Wii, Wib, Mii, Mib


# ── rectified-rate mode (engine params.nonlin = 'relu') ──
# u = H g stays linear; plasticity sees r = relu(u − E[u]).  Arc-cosine kernel:
# Cov(relu x, relu y) = sqrt(vi vj)/(2π)·(sin φ + (π−φ)cos φ − 1), φ = acos(c/sqrt(vi vj)).

def relu_corr(rho):
    rho = np.clip(np.asarray(rho, float), -1.0, 1.0)
    return (np.sqrt(np.maximum(0.0, 1 - rho * rho)) + (np.pi - np.arccos(rho)) * rho - 1) / (2 * np.pi)


def relu_corr_inv(t):
    """Inverse of relu_corr (vectorised Newton with bisection safeguard); clamps to ±1."""
    t = np.asarray(t, float); lo0, hi0 = -1 / (2 * np.pi), (np.pi - 1) / (2 * np.pi)
    x = np.clip(4 * t, -1, 1); lo = np.full_like(t, -1.0); hi = np.full_like(t, 1.0)
    for _ in range(60):
        f = relu_corr(x) - t
        hi = np.where(f > 0, x, hi); lo = np.where(f > 0, lo, x)
        d = (np.pi - np.arccos(x)) / (2 * np.pi)
        xn = np.where(d > 1e-12, x - f / np.where(d > 1e-12, d, 1.0), 0.5 * (lo + hi))
        bad = ~((xn > lo) & (xn < hi)); xn = np.where(bad, 0.5 * (lo + hi), xn)
        done = np.abs(xn - x) < 1e-15
        x = xn
        if done.all():
            break
    x = np.where(t >= hi0, 1.0, np.where(t > lo0, x, -1.0))
    return x


def dual_relu(vi, vj, c):
    """Cov(relu x, relu y) for zero-mean Gaussians with variances vi, vj and covariance c."""
    s = np.sqrt(np.asarray(vi, float) * np.asarray(vj, float))
    ok = s > 0
    return np.where(ok, s * relu_corr(np.asarray(c, float) / np.where(ok, s, 1.0)), 0.0)


def dual_relu_inv(vi, vj, k):
    s = np.sqrt(np.asarray(vi, float) * np.asarray(vj, float)); ok = s > 0
    return np.where(ok, s * relu_corr_inv(np.asarray(k, float) / np.where(ok, s, 1.0)), 0.0)


def plastic_cov(H, Sigma, nonlin='linear', HS=None):
    """(C_ii, C_ib): the covariances plasticity sees. linear: (HΣHᵀ, HΣ); relu: arc-cosine transform."""
    HS = H @ Sigma if HS is None else HS
    Cii, Cib = HS @ H.T, HS
    if nonlin == 'relu':
        v = (HS * H).sum(1)
        Cii = dual_relu(v[:, None], v[None, :], Cii)
        Cib = dual_relu(v[:, None], np.diag(Sigma)[None, :], Cib)
    return Cii, Cib


def step_nl(Wii, Wib, Mii, Mib, Sigma, P: Params, nonlin='linear'):
    """core.step with the engine's params.nonlin ('linear' is exactly core.step)."""
    if nonlin == 'linear':
        return step(Wii, Wib, Mii, Mib, Sigma, P)
    Wii = np.maximum(Wii, P.wmin * Mii); Wib = np.maximum(Wib, P.wmin * Mib)
    H = harmonic(Wii, Wib)
    Cii, Cib = plastic_cov(H, Sigma, nonlin)
    w0 = -P.eta * P.theta
    Wii = Wii + P.dt * (w0 - P.lam * Wii + P.gamma * P.eta * Cii)
    Wib = Wib + P.dt * (w0 - P.lam * Wib + P.eta * Cib)
    Wii = np.maximum(Wii, P.wmin) * Mii; np.fill_diagonal(Wii, 0.0)
    Wib = np.maximum(Wib, P.wmin) * Mib
    return Wii, Wib, H


def drift(Wii, Wib, Mii, Mib, Sigma, P: Params, nonlin='linear'):
    """dw/dt at the floor-clipped weights: (f_ii, f_ib, H, Wii_clipped, Wib_clipped)."""
    Wii = np.maximum(Wii, P.wmin * Mii); Wib = np.maximum(Wib, P.wmin * Mib)
    H = harmonic(Wii, Wib)
    Cii, Cib = plastic_cov(H, Sigma, nonlin)
    w0 = -P.eta * P.theta
    fii = ((w0 - P.lam * Wii) + P.gamma * P.eta * Cii) * Mii
    fib = ((w0 - P.lam * Wib) + P.eta * Cib) * Mib
    return fii, fib, H, Wii, Wib


def stationarity(Wii, Wib, Mii, Mib, Sigma, P: Params, nonlin='linear'):
    """Projected residual (SPEC §1): max |dw/dt| over potential edges, floor edges with
    negative drift count 0.  Returns dict(r, r_ii, r_ib)."""
    fii, fib, _, Wc, Wbc = drift(Wii, Wib, Mii, Mib, Sigma, P, nonlin)
    pii = np.where((Wc <= P.wmin) & (fii < 0), 0.0, np.abs(fii)) * Mii
    pib = np.where((Wbc <= P.wmin) & (fib < 0), 0.0, np.abs(fib)) * Mib
    r_ii = float(np.triu(pii, 1).max()) if Mii.any() else 0.0
    r_ib = float(pib.max()) if Mib.any() else 0.0
    return dict(r=max(r_ii, r_ib), r_ii=r_ii, r_ib=r_ib)


def sample_cov(G, center=False):
    """G (batch,B). center=False: GᵀG/batch (known zero mean); True: unbiased sample cov."""
    G = np.asarray(G, float)
    if center:
        X = G - G.mean(0); return X.T @ X / (len(G) - 1)
    return G.T @ G / len(G)


def step_samples(Wii, Wib, Mii, Mib, G, P: Params, center=False, nonlin='linear'):
    """Stochastic step with an explicit minibatch G (batch,B).  linear: Sigma replaced by
    sample_cov(G).  relu: rates relu(u − mu) from sampled voltages u = H g (mu = 0, or the
    batch mean if center), batch-mean-centred rate covariance (/(batch−1)); needs batch >= 2."""
    if nonlin == 'linear':
        return step(Wii, Wib, Mii, Mib, sample_cov(G, center), P)
    G = np.asarray(G, float)
    if len(G) < 2:
        raise ValueError("relu stochastic step needs batch >= 2")
    Wii = np.maximum(Wii, P.wmin * Mii); Wib = np.maximum(Wib, P.wmin * Mib)
    H = harmonic(Wii, Wib)
    U = G @ H.T
    if center:
        U = U - U.mean(0); Gc = G - G.mean(0)
    else:
        Gc = G
    R = np.maximum(U, 0); R = R - R.mean(0)
    Rb = np.maximum(Gc, 0); Rb = Rb - Rb.mean(0)
    n = len(G) - 1
    Cii, Cib = R.T @ R / n, R.T @ Rb / n
    w0 = -P.eta * P.theta
    Wii = Wii + P.dt * (w0 - P.lam * Wii + P.gamma * P.eta * Cii)
    Wib = Wib + P.dt * (w0 - P.lam * Wib + P.eta * Cib)
    Wii = np.maximum(Wii, P.wmin) * Mii; np.fill_diagonal(Wii, 0.0)
    Wib = np.maximum(Wib, P.wmin) * Mib
    return Wii, Wib, H


def dt_advice(Sigma, P: Params):
    """Euler: monotone for dt*lam < 1; weights bounded by max(w0, max(1,γ)η(max diag Σ − θ)/λ)."""
    return dict(dt_max=1.0 / P.lam, w_bound=max(0.0, max(1.0, P.gamma) * P.eta * (np.diag(Sigma).max() - P.theta) / P.lam))


# ── covariance generators (deterministic; same algorithms as the engine) ──

def clamp_psd(A, unit_diag=False):
    A = np.asarray(A, float); A = (A + A.T) / 2
    lam, V = np.linalg.eigh(A)
    if lam[0] >= 0:
        return A.copy()
    M = (V * np.maximum(lam, 0)) @ V.T
    if unit_diag:
        d = np.diag(M); s = np.where(d > 0, 1 / np.sqrt(np.where(d > 0, d, 1)), 0.0)
        M = M * np.outer(s, s); np.fill_diagonal(M, 1.0)
    return M


def _scale_var(R, v):
    v = np.asarray(v, float); return R * np.sqrt(np.outer(v, v))


def cov3(variances=(1, 1, 1), corr=(0, 0, 0)):
    a, b, c = corr
    R = clamp_psd(np.array([[1, a, b], [a, 1, c], [b, c, 1]], float), unit_diag=True)
    return _scale_var(R, variances)


def cov_blocks(sizes=(4, 4, 4), within=0.5, between=0.1, variances=1.0):
    """Returns (Sigma, labels)."""
    labels = np.repeat(np.arange(len(sizes)), sizes)
    wv = np.broadcast_to(np.asarray(within, float), (len(sizes),))
    R = np.where(labels[:, None] == labels[None, :], wv[labels][:, None], between).astype(float)
    np.fill_diagonal(R, 1.0)
    v = np.broadcast_to(np.asarray(variances, float), (len(sizes),))[labels]
    return _scale_var(clamp_psd(R, unit_diag=True), v), labels


def cov_hierarchical(branching=(3, 4), corr=(0.1, 0.5), variance=1.0):
    B = int(np.prod(branching)); D = len(branching)
    dg = np.array(np.unravel_index(np.arange(B), branching)).T      # (B, D), most significant first
    eq = np.cumprod(dg[:, None, :] == dg[None, :, :], axis=2).sum(2)  # length of common prefix
    R = np.asarray(corr, float)[np.minimum(eq, D - 1)]
    np.fill_diagonal(R, 1.0)
    return clamp_psd(R, unit_diag=True) * variance, dg[:, 0]


def random_orthogonal(B, rng: JsRng):
    Q = np.array([rng.normal() for _ in range(B * B)]).reshape(B, B)
    for k in range(B):
        for j in range(k):
            Q[:, k] -= (Q[:, j] @ Q[:, k]) * Q[:, j]
        Q[:, k] /= np.sqrt(Q[:, k] @ Q[:, k])
    return Q


def _norm_mean_diag(S):
    return S * (len(S) / np.trace(S))


def cov_power_law(B, alpha=1.0, seed=0, nonneg=False, normalize=True):
    Q = random_orthogonal(B, JsRng(seed))
    S = (Q * np.arange(1, B + 1, dtype=float) ** -alpha) @ Q.T
    S = (S + S.T) / 2
    if nonneg:
        off = S[~np.eye(B, dtype=bool)]
        S = S + max(0.0, -off.min())
    return _norm_mean_diag(S) if normalize else S


def cov_wishart(B, dof=None, seed=0, normalize=True):
    dof = B if dof is None else dof
    r = JsRng(seed)
    X = np.array([r.normal() for _ in range(dof * B)]).reshape(dof, B)
    S = X.T @ X / dof
    return _norm_mean_diag(S) if normalize else S


# ── analysis ──

def _l1_dist(H):
    return np.abs(H[:, None, :] - H[None, :, :]).sum(-1)


def _compact(roots):
    m = {}; return np.array([m.setdefault(r, len(m)) for r in roots], int)


def mst_l1(H):
    """Prim MST under L1 (same tie-breaking as engine.mstL1). Returns (parent, weight)."""
    D = _l1_dist(np.asarray(H, float)); N = len(D)
    parent = -np.ones(N, int); weight = np.zeros(N)
    best = np.full(N, np.inf); inT = np.zeros(N, bool)
    if N == 0:
        return parent, weight
    best[0] = 0.0
    for it in range(N):
        cand = np.where(~inT, best, np.inf); u = int(np.argmin(cand))
        inT[u] = True; weight[u] = 0.0 if it == 0 else best[u]
        upd = (~inT) & (D[u] < best)
        best[upd] = D[u][upd]; parent[upd] = u
    return parent, weight


def _cut(parent, weight, tol):
    N = len(parent); par = list(range(N))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(N):
        if parent[i] >= 0 and weight[i] < tol:
            a, b = find(i), find(parent[i])
            if a != b:
                par[max(a, b)] = min(a, b)
    return _compact([find(i) for i in range(N)])


def clusters_l1(H, tol=0.1):
    """Single-linkage clusters in L1 at fixed tol (order-independent). Returns labels."""
    return _cut(*mst_l1(H), tol)


def clusters_auto(H, min_sep=0.05, min_gap=3.0, fallback_tol=0.1, floor=1e-9):
    """Headline clusters (engine.clustersAuto): single linkage in L1, cut in the largest
    multiplicative gap of MST merge heights >= min_sep. Returns dict(labels, K, tol, gap, robust, heights)."""
    parent, weight = mst_l1(H)
    hs = np.sort(weight[parent >= 0])
    if len(hs) == 0 or hs[-1] < min_sep:
        lab = _cut(parent, weight, np.inf)
        return dict(labels=lab, K=int(lab.max(initial=-1) + 1), tol=np.inf, gap=np.inf, robust=True, heights=hs)
    best_r, tol = 0.0, np.inf
    for k in range(len(hs)):
        if hs[k] < min_sep:
            continue
        lo = max(hs[k - 1], floor) if k > 0 else floor
        r = hs[k] / lo
        if r > best_r:
            best_r, tol = r, math.sqrt(hs[k] * lo)
    robust = best_r >= min_gap
    t = tol if robust else fallback_tol
    lab = _cut(parent, weight, t)
    return dict(labels=lab, K=int(lab.max() + 1), tol=t, gap=best_r, robust=robust, heights=hs)


def cluster_count_curve(H, tols):
    parent, weight = mst_l1(H)
    w = weight[parent >= 0]
    return [1 + int((w >= t).sum()) for t in tols]


def clusters_synapse(Wii, Mii, wmin, eps=1e-6):
    """Connected components of the active lateral graph (Wii > wmin(1+eps)). Returns labels."""
    from scipy.sparse.csgraph import connected_components
    A = (Mii > 0) & (Wii > wmin * (1 + eps)); A = np.triu(A, 1); A = A | A.T
    _, lab = connected_components(A.astype(float), directed=False)
    return _compact(lab)


def cluster_stats(H, labels, Sigma):
    """part = member mean of h; energy = partᵀΣpart; spread = mean L1 distance to part."""
    labels = np.asarray(labels); K = int(labels.max() + 1)
    sizes = np.bincount(labels, minlength=K)
    parts = np.array([H[labels == k].mean(0) for k in range(K)])
    spread = np.array([np.abs(H[labels == k] - parts[k]).sum(1).mean() for k in range(K)])
    energy = np.einsum('kb,bc,kc->k', parts, Sigma, parts)
    return dict(sizes=sizes, parts=parts, energy=energy, spread=spread)


def effective_rank(H):
    """Participation ratio of squared singular values: ||H||_F^4 / ||HᵀH||_F^2."""
    G = H.T @ H; g2 = (G * G).sum()
    return float(np.trace(G) ** 2 / g2) if g2 > 0 else 0.0


def similarity(H, Sigma):
    return H @ Sigma @ H.T


def pca3(H, prev_basis=None):
    """engine.pca3: top-3 PCA of embeddings; Procrustes-aligned to prev_basis if given,
    else each axis signed so its largest-|.| entry is positive. Returns dict(coords, basis, mean, variance)."""
    H = np.asarray(H, float); N, B = H.shape
    mean = H.mean(0); X = H - mean
    lam, V = np.linalg.eigh(X.T @ X / N); V = V[:, ::-1]
    basis = np.zeros((B, 3)); kk = min(3, B); basis[:, :kk] = V[:, :kk]
    aligned = False
    if prev_basis is not None and np.shape(prev_basis) == (B, 3):
        K = basis.T @ prev_basis; KtK = K.T @ K
        ev, Q = np.linalg.eigh(KtK)
        if ev[0] > 1e-10 * max(ev[-1], 1e-300):
            basis = basis @ (K @ (Q / np.sqrt(ev)) @ Q.T); aligned = True
    if not aligned:
        for k in range(kk):
            bi = 0
            for b in range(1, B):
                if abs(basis[b, k]) > abs(basis[bi, k]) + 1e-12:
                    bi = b
            if basis[bi, k] < 0:
                basis[:, k] = -basis[:, k]
    coords = X @ basis
    return dict(coords=coords, basis=basis, mean=mean, variance=(coords ** 2).mean(0))


# ── weight-space readout (SPEC §3, §7.3) ──

def project1(C):
    """P C P with P = I − 11ᵀ/B."""
    C = np.asarray(C, float)
    r = C.mean(1); c = C.mean(0); return C - r[:, None] - c[None, :] + C.mean()


def psd_project(A):
    """(A_+, neg_mass, neg_frac): clip negative eigenvalues."""
    lam, V = np.linalg.eigh((A + A.T) / 2)
    neg = float(-lam[lam < 0].sum()); tot = float(np.abs(lam).sum())
    return (V * np.maximum(lam, 0)) @ V.T, neg, (neg / tot if tot > 0 else 0.0)


def readout_equations(Wii, Wib, Mii, Mib, theta, lam, eta, gamma, wmin, eps=1e-6, use='both',
                      theta_rhs=None, dWii=None, dWib=None, active_ii=None, active_ib=None):
    """Active-edge fixed-point equations (engine.readoutEquations).
    I–I rows (i<j): γ h_iᵀ C h_j = rhs;  I–B rows: (H C)_ib = rhs;
    rhs = θ_rhs + (λ/η) w + (1/η) dw/dt.  Returns dict(ii (m,2), ii_r, ib (m,2), ib_r, gamma)."""
    N, B = Wib.shape; thr = wmin * (1 + eps); kap = lam / eta
    th = theta if theta_rhs is None else theta_rhs
    use_ii = use != 'ib' and gamma != 0; use_ib = use != 'ii'
    out = dict(gamma=gamma)
    iu, ju = np.triu_indices(N, 1)
    act = (active_ii if active_ii is not None else Wii)[iu, ju] > thr
    sel = (Mii[iu, ju] > 0) & act & use_ii
    out['ii'] = np.stack([iu[sel], ju[sel]], 1)
    out['ii_r'] = th + kap * Wii[iu, ju][sel] + (dWii[iu, ju][sel] / eta if dWii is not None else 0.0)
    ii_, bb_ = np.nonzero((Mib > 0) & ((active_ib if active_ib is not None else Wib) > thr) & use_ib)
    out['ib'] = np.stack([ii_, bb_], 1)
    out['ib_r'] = th + kap * Wib[ii_, bb_] + (dWib[ii_, bb_] / eta if dWib is not None else 0.0)
    return out


class _Op:
    """Measurement operator A: Sym(B) (+τ) → R^m and its adjoint (engine makeOperator)."""

    def __init__(self, H, eq, fit, w_ii=1.0, w_ib=1.0):
        self.H = H; self.eq = eq; self.fit = fit; self.g = eq['gamma']
        self.w_ii, self.w_ib = w_ii, w_ib
        self.mII = len(eq['ii']); self.mIB = len(eq['ib'])
        i, j = eq['ii'].T if self.mII else (np.zeros(0, int), np.zeros(0, int))
        self.i, self.j = i, j
        nrm = (H * H).sum(1)
        d = (H[i] * H[j]).sum(1)
        ib_i, ib_b = eq['ib'].T if self.mIB else (np.zeros(0, int), np.zeros(0, int))
        self.ib_i, self.ib_b = ib_i, ib_b
        self.trace = float(w_ii * self.g ** 2 * ((nrm[i] * nrm[j] + d * d) / 2).sum()
                           + w_ib * ((nrm[ib_i] + H[ib_i, ib_b] ** 2) / 2).sum())

    def A(self, C, tau):
        HC = self.H @ C
        y_ii = self.g * (HC[self.i] * self.H[self.j]).sum(1) - tau
        y_ib = HC[self.ib_i, self.ib_b] - tau
        return np.concatenate([y_ii, y_ib])

    def At(self, r):
        N, B = self.H.shape
        r_ii = self.w_ii * r[:self.mII]; r_ib = self.w_ib * r[self.mII:]
        T = np.zeros((N, B))
        np.add.at(T, self.i, (self.g * r_ii)[:, None] * self.H[self.j])
        np.add.at(T, (self.ib_i, self.ib_b), r_ib)
        G = self.H.T @ T
        return (G + G.T) / 2, (-(r_ii.sum() + r_ib.sum()) if self.fit else 0.0)


def readout_ls(H, Wii, Wib, Mii, Mib, theta, lam=1.0, eta=1.0, gamma=1.0, wmin=1e-3, eps=1e-6,
               use='both', theta_mode='known', ridge=1e-6, tol=1e-10, max_iter=2000, psd=False,
               x0=None, weight_ii=1.0, weight_ib=1.0, method='cg', dWii=None, dWib=None,
               active_ii=None, active_ib=None):
    """Masked ridge LS readout (engine.readoutLS; same options).  method='cg' runs the same
    conjugate-gradient iteration as the engine; method='direct' forms the normal equations
    in an orthonormal basis of Sym(B) (exact; small B only)."""
    eq = readout_equations(Wii, Wib, Mii, Mib, theta, lam, eta, gamma, wmin, eps, use,
                           theta_rhs=theta if theta_mode == 'known' else 0.0,
                           dWii=dWii, dWib=dWib, active_ii=active_ii, active_ib=active_ib)
    return solve_equations(H, eq, theta=theta, gamma=gamma, use=use, theta_mode=theta_mode, ridge=ridge,
                           tol=tol, max_iter=max_iter, psd=psd, x0=x0, weight_ii=weight_ii,
                           weight_ib=weight_ib, method=method)


def solve_equations(H, eq, theta=0.0, gamma=1.0, use='both', theta_mode='known', ridge=1e-6, tol=1e-10,
                    max_iter=2000, psd=False, x0=None, weight_ii=1.0, weight_ib=1.0, method='cg'):
    """engine.solveEquations: ridge LS for symmetric C given equations (rhs taken as given)."""
    N, B = H.shape
    warnings = []
    if gamma == 0 and use != 'ib':
        warnings.append('gamma = 0: I–I edges carry no information about C; using I–B only')
    if theta_mode == 'project' and gamma != 1:
        warnings.append('thetaMode "project" assumes gamma = 1; use "fit"')
    fit = theta_mode == 'fit'
    op = _Op(H, eq, fit, weight_ii, weight_ib)
    p = B * (B + 1) // 2
    alpha = ridge * op.trace / p
    eps2 = 1e-4 if theta_mode == 'project' else 1.0
    rhs = np.concatenate([eq['ii_r'], eq['ib_r']])

    def reg(C):
        if theta_mode == 'project':
            Pc = project1(C); return alpha * (Pc + eps2 * (C - Pc))
        return alpha * C

    def Op(C, tau):
        G, s = op.At(op.A(C, tau)); return G + reg(C), (s if fit else 0.0)

    if method == 'direct':
        iu, ju = np.triu_indices(B)
        basis = []
        for b, c in zip(iu, ju):
            E = np.zeros((B, B))
            if b == c: E[b, b] = 1.0
            else: E[b, c] = E[c, b] = 1 / math.sqrt(2)
            basis.append(E)
        n = p + (1 if fit else 0)
        cols = [op.A(E, 0.0) for E in basis] + ([op.A(np.zeros((B, B)), 1.0)] if fit else [])
        A = np.stack(cols, 1) if cols else np.zeros((len(rhs), n))
        w = np.concatenate([np.full(op.mII, weight_ii), np.full(op.mIB, weight_ib)])
        R = np.zeros((n, n))
        for u, E in enumerate(basis):
            RE = reg(E)
            for v, F in enumerate(basis):
                R[u, v] = (RE * F).sum()
        x = np.linalg.solve(A.T @ (w[:, None] * A) + R, A.T @ (w * rhs))
        C = sum(xu * E for xu, E in zip(x[:p], basis)); tau = x[p] if fit else 0.0
        iters, relres = 0, 0.0
    else:
        bC, bt = op.At(rhs)
        C = np.zeros((B, B)) if x0 is None else np.array(x0, float).reshape(B, B).copy()
        tau = 0.0
        qC, qt = Op(C, tau)
        rC, rt = bC - qC, (bt - qt if fit else 0.0)
        dC, dt_ = rC.copy(), rt
        rr = (rC * rC).sum() + rt * rt
        bn = math.sqrt((bC * bC).sum() + bt * bt) or 1.0
        iters = 0
        while iters < max_iter and math.sqrt(rr) > tol * bn:
            qC, qt = Op(dC, dt_)
            a = rr / ((dC * qC).sum() + dt_ * qt)
            C = C + a * dC; tau = tau + a * dt_
            rC = rC - a * qC; rt = rt - a * qt
            rn = (rC * rC).sum() + rt * rt; be = rn / rr; rr = rn
            dC = rC + be * dC; dt_ = rt + be * dt_
            iters += 1
        relres = math.sqrt(rr) / bn
    y = op.A(C, tau if fit else 0.0) - rhs
    rms_ii = float(np.sqrt((y[:op.mII] ** 2).mean())) if op.mII else 0.0
    rms_ib = float(np.sqrt((y[op.mII:] ** 2).mean())) if op.mIB else 0.0
    if theta_mode == 'project':
        C = project1(C)
    neg = 0.0
    if psd:
        C, neg, _ = psd_project(C)
    th = tau if fit else (theta if theta_mode == 'known' else float('nan'))
    return dict(C=C, theta=float(th), iters=iters, rel_residual=relres, rms_ii=rms_ii, rms_ib=rms_ib,
                m_ii=op.mII, m_ib=op.mIB, alpha=alpha, neg_mass=neg, warnings=warnings)


def readout_state(Wii, Wib, Mii, Mib, P: Params, **kw):
    H = harmonic(Wii, Wib)
    return readout_ls(H, Wii, Wib, Mii, Mib, P.theta, P.lam, P.eta, P.gamma, P.wmin, **kw)


def readout_relu(H, Wii, Wib, Mii, Mib, theta, lam=1.0, eta=1.0, gamma=1.0, wmin=1e-3, eps=1e-6, use='both',
                 ridge=1e-6, tol=1e-12, max_iter=5000, psd=False, outer=50, outer_tol=1e-10, method='cg'):
    """engine.readoutRelu: alternating readout under nonlin 'relu' (theta known)."""
    eq = readout_equations(Wii, Wib, Mii, Mib, theta, lam, eta, gamma, wmin, eps, use, theta_rhs=theta)
    return solve_relu(H, eq, ridge=ridge, tol=tol, max_iter=max_iter, psd=psd, outer=outer,
                      outer_tol=outer_tol, method=method)


def solve_relu(H, eq, ridge=1e-6, tol=1e-12, max_iter=5000, psd=False, outer=50, outer_tol=1e-10, method='cg'):
    N, B = H.shape; g = eq['gamma']; tiny = 1e-12
    kII = eq['ii_r'] / g if len(eq['ii_r']) else eq['ii_r']; kIB = eq['ib_r']
    lin = dict(ii=eq['ii'], ib=eq['ib'], gamma=1.0, ii_r=3 * kII, ib_r=3 * kIB)
    kw = dict(ridge=ridge, tol=tol, max_iter=max_iter, method=method)
    res = solve_equations(H, lin, **kw)
    C = res['C']; change = np.inf; t = 0
    i, j = (eq['ii'].T if len(eq['ii']) else (np.zeros(0, int),) * 2)
    bi, bb = (eq['ib'].T if len(eq['ib']) else (np.zeros(0, int),) * 2)
    for t in range(1, outer + 1):
        v = np.maximum(((H @ C) * H).sum(1), tiny)
        lin['ii_r'] = dual_relu_inv(v[i], v[j], kII)
        lin['ib_r'] = dual_relu_inv(v[bi], np.maximum(np.diag(C)[bb], tiny), kIB)
        res = solve_equations(H, lin, x0=C, **kw)
        change = float(np.sqrt(((res['C'] - C) ** 2).sum() / ((res['C'] ** 2).sum() or 1)))
        C = res['C']
        if change < outer_tol:
            break
    HC = H @ C; v = np.maximum((HC * H).sum(1), tiny)
    rII = g * dual_relu(v[i], v[j], (HC[i] * H[j]).sum(1)) - eq['ii_r']
    rIB = dual_relu(v[bi], np.maximum(np.diag(C)[bb], tiny), HC[bi, bb]) - eq['ib_r']
    neg = 0.0
    if psd:
        C, neg, _ = psd_project(C)
    return dict(res, C=C, neg_mass=neg, outer=t, change=change,
                rms_rate_ii=float(np.sqrt((rII ** 2).mean())) if len(rII) else 0.0,
                rms_rate_ib=float(np.sqrt((rIB ** 2).mean())) if len(rIB) else 0.0)


def readout_transient(prev, nxt, Mii, Mib, P: Params, **kw):
    """Derivative-aware readout from consecutive Euler snapshots prev=(Wii,Wib) → nxt (engine.readoutTransient)."""
    Wc = np.maximum(prev[0], P.wmin * Mii); Wbc = np.maximum(prev[1], P.wmin * Mib)
    H = harmonic(Wc, Wbc)
    return readout_ls(H, Wc, Wbc, Mii, Mib, P.theta, P.lam, P.eta, P.gamma, P.wmin,
                      dWii=(nxt[0] - Wc) / P.dt, dWib=(nxt[1] - Wbc) / P.dt,
                      active_ii=nxt[0], active_ib=nxt[1], **kw)


def readout_naive(H, Wii, theta, lam=1.0, eta=1.0, gamma=1.0, ridge=1e-6):
    """The note's C = H⁺ R H⁺ᵀ (engine.readoutNaive): ridge pseudo-inverse, R_ij = (θ + (λ/η)w_ij)/γ,
    diagonal filled with the row max of the off-diagonal. None if gamma == 0."""
    if gamma == 0:
        return None
    N, B = H.shape
    R = (theta + (lam / eta) * Wii) / gamma
    off = R.copy(); np.fill_diagonal(off, -np.inf)
    np.fill_diagonal(R, off.max(1) if N > 1 else theta / gamma)
    K = H.T @ H; alpha = ridge * np.trace(K) / B
    X = np.linalg.solve(K + alpha * np.eye(B), H.T).T
    C = X.T @ R @ X
    return (C + C.T) / 2


def measurement_spectrum(H, eq, rtol=1e-10):
    """Per-family spectrum of AᵀA on Sym(B) (orthonormal basis). dict(ii, ib, both, p),
    each dict(rank, cond, eig)."""
    N, B = H.shape; p = B * (B + 1) // 2
    iu, ju = np.triu_indices(B); sc = np.where(iu == ju, 1.0, math.sqrt(2) / 2)
    g = eq['gamma']
    if len(eq['ii']):
        i, j = eq['ii'].T
        Aii = g * sc * np.where(iu == ju, H[i][:, iu] * H[j][:, ju], H[i][:, iu] * H[j][:, ju] + H[i][:, ju] * H[j][:, iu])
    else:
        Aii = np.zeros((0, p))
    if len(eq['ib']):
        i, bb = eq['ib'].T
        Hi = H[i]
        Aib = sc * (np.where(ju == bb[:, None], Hi[:, iu], 0.0) + np.where((iu == bb[:, None]) & (iu != ju), Hi[:, ju], 0.0))
    else:
        Aib = np.zeros((0, p))

    def summ(A):
        ev = np.sort(np.linalg.eigvalsh(A.T @ A))[::-1]; mx = max(ev[0], 0.0)
        keep = ev > rtol * mx if mx > 0 else np.zeros_like(ev, bool)
        rank = int(keep.sum())
        return dict(rank=rank, cond=float(np.sqrt(mx / ev[keep].min())) if rank else np.inf, eig=ev)
    return dict(ii=summ(Aii), ib=summ(Aib), both=summ(np.vstack([Aii, Aib])), p=p)


# ── metrics (SPEC §3, §7.4) ──

def _sqrtm_psd(S):
    lam, V = np.linalg.eigh(S); return (V * np.sqrt(np.maximum(lam, 0))) @ V.T


def compare_cov(C_hat, Sigma, setting='contrast', k=3):
    """engine.compareCov. setting: 'known' | 'scale' | 'contrast' (P·P, 'contrast-subspace score')."""
    A = np.asarray(C_hat, float); A = (A + A.T) / 2; S = np.asarray(Sigma, float).copy()
    if setting == 'contrast':
        A, S = project1(A), project1(S)
    aa, ss, as_ = (A * A).sum(), (S * S).sum(), (A * S).sum()
    cosine = as_ / math.sqrt(aa * ss) if aa > 0 and ss > 0 else 0.0
    scale = 1.0 if setting == 'known' else (as_ / aa if aa > 0 else 0.0)
    rel_err = math.sqrt(((scale * A - S) ** 2).sum() / ss)
    Ap, neg, negf = psd_project(A)
    trS, trA = np.trace(S), np.trace(Ap)
    Sh = _sqrtm_psd(S)
    F = np.sqrt(np.maximum(np.linalg.eigvalsh(Sh @ Ap @ Sh), 0)).sum()
    if trA <= 0:
        bw = 1.0
    elif setting == 'known':
        bw = (trA + trS - 2 * F) / trS
    else:
        bw = max(0.0, 1 - F * F / (trA * trS))
    la, Va = np.linalg.eigh(A); ls, Vs = np.linalg.eigh(S)
    la, Va, ls, Vs = la[::-1], Va[:, ::-1], ls[::-1], Vs[:, ::-1]
    kk = min(k, len(A))
    ov = ((Va[:, :kk].T @ Vs[:, :kk]) ** 2).sum() / kk
    return dict(setting=setting, scale=scale, cosine=cosine, rel_err=rel_err, bw=float(bw), neg_mass=neg,
                neg_frac=negf, subspace=float(ov), k=kk, eig_hat=la * scale, eig_true=ls)


def baselines(Sigma, k=3):
    lam, V = np.linalg.eigh(Sigma); lam, V = lam[::-1], V[:, ::-1]
    return dict(identity=np.eye(len(Sigma)), diag=np.diag(np.diag(Sigma)),
                rankK=(V[:, :k] * lam[:k]) @ V[:, :k].T)
