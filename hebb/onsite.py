"""Reference implementation of the v2 model (SPEC.md §1): a conductance network with
convex on-site potentials and the distance-based plasticity rule.

    E_W(u; g) = 1/2 sum_{i<j} w_ij (u_i-u_j)^2 + 1/2 sum_{i,b} a_ib (u_i-g_b)^2 + sum_i V_i(u_i)
    V_i(u)    = klo_i/2 (u-t_i)_-^2 + khi_i/2 (u-t_i)_+^2          (two-slope, convex)
    dw_ij/dt  = b - alpha w_ij - 1/2 E_g[(u_i-u_j)^2],  w >= 0      (envelope-theorem gradient)

Everything is deterministic given the arrays passed in, so the browser engine can be
checked against it. Expectations are plain means over the rows of the input bank G
(uncentred), exactly as in the learning objective.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Net:
    A: np.ndarray            # (N,B) fixed input attachments, >= 0
    t: np.ndarray            # (N,) on-site thresholds
    klo: np.ndarray          # (N,) on-site stiffness below threshold
    khi: np.ndarray          # (N,) on-site stiffness above threshold
    W: np.ndarray = field(default=None)   # (N,N) symmetric, zero diagonal, >= 0

    def __post_init__(self):
        N = self.A.shape[0]
        for name in ('t', 'klo', 'khi'):
            setattr(self, name, np.broadcast_to(np.asarray(getattr(self, name), float), (N,)).copy())
        if self.W is None:
            self.W = np.zeros((N, N))

    @property
    def N(self):
        return self.A.shape[0]

    @property
    def B(self):
        return self.A.shape[1]


def make_attachments(N, B, s, rng, deg=1.0, equal=False):
    """Each neuron attached to s distinct random inputs, total conductance deg.
    Shares are Dirichlet(1) (or equal)."""
    A = np.zeros((N, B))
    for i in range(N):
        idx = rng.choice(B, s, replace=False)
        A[i, idx] = (np.full(s, 1.0 / s) if equal else rng.dirichlet(np.ones(s))) * deg
    return A


def make_net(N, B, rng, s=3, deg=1.0, klo=0.0, khi=0.0, sigma_t=0.0, equal=False):
    return Net(A=make_attachments(N, B, s, rng, deg, equal), t=sigma_t * rng.standard_normal(N),
               klo=klo, khi=khi)


def sample_inputs(Sigma, S, rng, kind='gauss'):
    """Zero-mean inputs with covariance Sigma. kind='gauss' or 'laplace' (independent
    unit-variance Laplace sources mixed by Sigma^{1/2}; same covariance, heavy tails)."""
    B = len(Sigma)
    lam, V = np.linalg.eigh(Sigma)
    root = (V * np.sqrt(np.clip(lam, 0, None))) @ V.T
    Z = rng.standard_normal((S, B)) if kind == 'gauss' else rng.laplace(size=(S, B)) / np.sqrt(2)
    return Z @ root


def laplacian(net):
    d = net.W.sum(1) + net.A.sum(1)
    return np.diag(d) - net.W


def energy(net, U, G):
    """Mean energy over the bank (U: (S,N), G: (S,B))."""
    iu = np.triu_indices(net.N, 1)
    e = 0.5 * (net.W[iu] * (U[:, iu[0]] - U[:, iu[1]]) ** 2).sum(1)
    e += 0.5 * (net.A[None] * (U[:, :, None] - G[:, None, :]) ** 2).sum((1, 2))
    x = U - net.t
    e += (0.5 * np.where(x < 0, net.klo, net.khi) * x ** 2).sum(1)
    return e.mean()


def equilibrate(net, G, U0=None, tol=1e-10, max_iter=100):
    """Exact equilibrium responses U (S,N) for each input row of G.
    Semismooth Newton on the piecewise-linear node equations, with a backtracking
    safeguard on the gradient norm. Linear case (klo = khi = 0) is one solve."""
    L = laplacian(net)
    rhs = G @ net.A.T
    N = net.N
    if not (net.klo.any() or net.khi.any()):
        return np.linalg.solve(L, rhs.T).T
    kbar = 0.5 * (net.klo + net.khi)
    U = np.linalg.solve(L + np.diag(kbar), (rhs + kbar * net.t).T).T if U0 is None else U0.copy()

    def grad(U):
        x = U - net.t
        k = np.where(x < 0, net.klo, net.khi)
        return U @ L.T + k * x - rhs, k

    g, k = grad(U)
    for _ in range(max_iter):
        gn = np.abs(g).max(1)
        act = gn > tol
        if not act.any():
            break
        Hs = L[None] + k[act][:, :, None] * np.eye(N)[None]
        dU = np.linalg.solve(Hs, g[act][..., None])[..., 0]
        step = np.ones(act.sum())
        Ua = U[act]
        for _ in range(30):
            Ut = U.copy()
            Ut[act] = Ua - step[:, None] * dU
            g2, k2 = grad(Ut)
            worse = np.abs(g2[act]).max(1) > gn[act] * (1 + 1e-12)
            if not worse.any():
                break
            step[worse] *= 0.5
        U, g, k = Ut, g2, k2
    return U


def harmonic(net):
    """Linear zero-potential response matrix H (N,B): u = H g. Rows sum to 1."""
    return np.linalg.solve(laplacian(net), net.A)


def dist2(U, V=None):
    """D2[i,j] = mean over the bank of (U_i - V_j)^2."""
    V = U if V is None else V
    S = len(U)
    return ((U ** 2).mean(0)[:, None] + (V ** 2).mean(0)[None, :] - 2 * U.T @ V / S)


def drift(net, G, b, alpha, U=None):
    U = equilibrate(net, G) if U is None else U
    dW = b - alpha * net.W - 0.5 * dist2(U)
    np.fill_diagonal(dW, 0.0)
    return dW, U


def residual(net, G, b, alpha, U=None):
    """Projected stationarity residual: max |dw/dt|, edges at zero with negative drift count 0."""
    dW, U = drift(net, G, b, alpha, U)
    dW = np.where((net.W <= 0) & (dW < 0), 0.0, dW)
    return np.abs(dW).max()


def step(net, G, b, alpha, dt, U=None):
    """One projected Euler step of the slow dynamics, in place. Returns the U used."""
    dW, U = drift(net, G, b, alpha, U)
    net.W = np.maximum(net.W + dt * dW, 0.0)
    return U


def run(net, G, b, alpha, dt=0.2, T=5000, tol=1e-8, callback=None):
    """Integrate to stationarity. Returns dict(steps, residual, U)."""
    U = None
    for n in range(T):
        U = equilibrate(net, G, U0=U)
        W_old = net.W
        step(net, G, b, alpha, dt, U)
        if callback is not None:
            callback(n, net, U)
        if np.abs(net.W - W_old).max() / dt < tol:
            break
    U = equilibrate(net, G, U0=U)
    return dict(steps=n + 1, residual=residual(net, G, b, alpha, U), U=U)


def components(W, eps=1e-9):
    """Connected components of the live graph (w > eps). Labels ordered by first member."""
    N = len(W)
    lab = -np.ones(N, int)
    c = 0
    for i in range(N):
        if lab[i] >= 0:
            continue
        stack = [i]
        lab[i] = c
        while stack:
            j = stack.pop()
            for k in np.nonzero(W[j] > eps)[0]:
                if lab[k] < 0:
                    lab[k] = c
                    stack.append(k)
        c += 1
    return lab


def cluster_stats(U, lab):
    """Per component: size, prototype (mean response), max and mean distance D to the
    prototype, and d_eff = (Tr C)^2 / Tr(C^2) of the centred member responses."""
    out = []
    Uc = U - U.mean(0)
    for k in range(lab.max() + 1):
        idx = np.nonzero(lab == k)[0]
        proto = U[:, idx].mean(1)
        D = np.sqrt(((U[:, idx] - proto[:, None]) ** 2).mean(0))
        C = Uc[:, idx].T @ Uc[:, idx] / len(U)
        tr, tr2 = np.trace(C), (C ** 2).sum()
        out.append(dict(members=idx, size=len(idx), proto=proto, diam_max=D.max(), diam_mean=D.mean(),
                        d_eff=tr ** 2 / tr2 if tr2 > 0 else 0.0))
    return out


def population_modes(U):
    """Eigen-decomposition of the population response operator on the bank.
    Returns q (descending), V (neuron-space eigenvectors, columns), Psi (S, r) unit-norm
    eigenfunctions psi_a = q_a^{-1/2} sum_i V_ia F~_i, and d_eff."""
    Uc = U - U.mean(0)
    C = Uc.T @ Uc / len(U)
    q, V = np.linalg.eigh(C)
    q, V = np.clip(q[::-1], 0, None), V[:, ::-1]
    keep = q > 1e-12 * max(q[0], 1e-300)
    Psi = Uc @ V[:, keep] / np.sqrt(q[keep])
    return dict(q=q, V=V, Psi=Psi, d_eff=q.sum() ** 2 / (q ** 2).sum(), C=C)
