"""Distance rule (SPEC §1, hebb/onsite.py) at N = 300-500: fast integrators with the same
semantics as onsite.run (projected Euler, dt = min(0.5, 1/alpha), stop when max|dW|/dt < tol,
final equilibrium and projected residual), plus snapshot capture for films.

- lin_run   (numpy): klo = khi = 0. u = H g with H = L^{-1} A, so the training-bank distances are
              D^2_ij = (h_i - h_j)^T C_hat (h_i - h_j), C_hat = G^T G / S: identical to
              onsite.dist2(onsite.equilibrate(...)) without the per-sample solve.
- rect_run  (torch, float64, CPU or CUDA): batched semismooth Newton with the same backtracking
              safeguard as onsite.equilibrate, Cholesky solves (L + diag k is SPD).
Both are validated against hebb/onsite.py by hebb/big/validate.py.

Snapshots: at the steps in `snap_steps` the state W_n is stored with its own equilibrium U_n
(post-update weights with their equilibrium, R4), its projected residual, live-edge count,
component count and the strongest <= 1500 live edges. Responses for film coordinates are kept
for the snapshot steps only (float32), and reduced to coordinates by the caller.
"""
import time

import numpy as np

from hebb.big import common as cm

EPS = 1e-9            # live edge: w > EPS (as onsite.components / onsite_analysis)
RUN_TOL = 1e-9
RES_TOL = 1e-6


def default_dt(alpha):
    return float(min(0.5, 1.0 / alpha))


def _snap(W, extra):
    e, w = cm.top_edges(W, EPS)
    iu = np.triu(W, 1) > EPS
    return dict(edges=e, w=w, n_live=int(iu.sum()), n_comp=cm.n_components(W, EPS), **extra)


# ----------------------------------------------------------------------------- linear (numpy)
def lin_run(A, G, b, alpha, W0=None, dt=None, Tmax=40000, tol=RUN_TOL, snap_steps=(), keep_H=True):
    N = A.shape[0]
    dt = default_dt(alpha) if dt is None else dt
    Chat = G.T @ G / len(G)
    W = np.zeros((N, N)) if W0 is None else W0.copy()
    a = A.sum(1)
    snaps = {}
    t0 = time.time()

    def drift(W):
        L = np.diag(W.sum(1) + a) - W
        H = np.linalg.solve(L, A)
        M = H @ Chat @ H.T
        d = np.diag(M)
        D2 = d[:, None] + d[None, :] - 2 * M
        dW = b - alpha * W - 0.5 * D2
        np.fill_diagonal(dW, 0.0)
        return dW, H, D2

    n = 0
    for n in range(Tmax):
        dW, H, _ = drift(W)
        if n in snap_steps:
            r = float(np.abs(np.where((W <= 0) & (dW < 0), 0.0, dW)).max())
            snaps[n] = _snap(W, dict(r=r, H=H.astype(np.float32) if keep_H else None))
        Wn = np.maximum(W + dt * dW, 0.0)
        done = np.abs(Wn - W).max() / dt < tol
        W = Wn
        if done:
            n += 1
            break
    else:
        n = Tmax
    dW, H, D2 = drift(W)
    r = float(np.abs(np.where((W <= 0) & (dW < 0), 0.0, dW)).max())
    snaps[n] = _snap(W, dict(r=r, H=H.astype(np.float32) if keep_H else None))
    return dict(W=W, H=H, D2=D2, steps=n, t=n * dt, residual=r, converged=bool(r < RES_TOL),
                seconds=time.time() - t0, snaps=snaps, dt=dt)


# ----------------------------------------------------------------------------- rectifier (torch)
def equilibrate_t(L, A, t, klo, khi, G, U0=None, tol=1e-10, max_iter=100, chunk=None):
    """Torch port of onsite.equilibrate (same algorithm). All arguments torch float64 tensors on
    one device: L (N,N), A (N,B), t/klo/khi (N,), G (S,B). Returns (U (S,N), info)."""
    import torch
    if chunk is not None and len(G) > chunk:
        outs, infos = [], []
        for s in range(0, len(G), chunk):
            u, inf = equilibrate_t(L, A, t, klo, khi, G[s:s + chunk], None if U0 is None else U0[s:s + chunk],
                                   tol, max_iter)
            outs.append(u)
            infos.append(inf)
        return torch.cat(outs), dict(iters=max(i["iters"] for i in infos), gmax=max(i["gmax"] for i in infos))
    rhs = G @ A.T
    if U0 is None:
        kbar = 0.5 * (klo + khi)
        U = torch.linalg.solve(L + torch.diag(kbar), (rhs + kbar * t).T).T
    else:
        U = U0.clone()

    def grad(U):
        x = U - t
        k = torch.where(x < 0, klo, khi)
        return U @ L.T + k * x - rhs, k

    g, k = grad(U)
    it = 0
    for it in range(max_iter):
        gn = g.abs().amax(1)
        act = gn > tol
        if not bool(act.any()):
            break
        ia = torch.nonzero(act).squeeze(1)
        Hs = L.unsqueeze(0) + torch.diag_embed(k[ia])
        C, bad = torch.linalg.cholesky_ex(Hs)
        if bool(bad.any()):
            dU = torch.linalg.solve(Hs, g[ia].unsqueeze(-1)).squeeze(-1)
        else:
            dU = torch.cholesky_solve(g[ia].unsqueeze(-1), C).squeeze(-1)
        step = torch.ones(len(ia), dtype=L.dtype, device=L.device)
        Ua = U[ia]
        for _ in range(30):
            Ut = U.clone()
            Ut[ia] = Ua - step[:, None] * dU
            g2, k2 = grad(Ut)
            worse = g2[ia].abs().amax(1) > gn[ia] * (1 + 1e-12)
            if not bool(worse.any()):
                break
            step = torch.where(worse, step * 0.5, step)
        U, g, k = Ut, g2, k2
    return U, dict(iters=it, gmax=float(g.abs().max()))


def rect_run(A, t, klo, khi, G, b, alpha, W0=None, dt=None, Tmax=40000, tol=RUN_TOL, snap_steps=(),
             keep_U_steps=None, device=None, log_every=0):
    """Distance rule with on-site potentials, torch. Arrays in numpy; returns numpy.
    keep_U_steps: steps at which to keep the training-bank responses U (float32) for film coords
    (None = all snapshot steps)."""
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    f = lambda x: torch.as_tensor(np.asarray(x, float), dtype=torch.float64, device=dev)  # noqa: E731
    N = A.shape[0]
    dt = default_dt(alpha) if dt is None else dt
    At, tt, kl, kh, Gt = f(A), f(np.broadcast_to(t, (N,))), f(np.broadcast_to(klo, (N,))), \
        f(np.broadcast_to(khi, (N,))), f(G)
    a = At.sum(1)
    W = torch.zeros((N, N), dtype=torch.float64, device=dev) if W0 is None else f(W0)
    S = len(G)
    snaps = {}
    t0 = time.time()
    U = None
    newton_iters = []

    def lap(W):
        return torch.diag(W.sum(1) + a) - W

    def drift(W, U):
        m2 = (U ** 2).mean(0)
        D2 = m2[:, None] + m2[None, :] - 2 * (U.T @ U) / S
        dW = b - alpha * W - 0.5 * D2
        dW.fill_diagonal_(0.0)
        return dW

    def proj_res(W, dW):
        return float(torch.where((W <= 0) & (dW < 0), torch.zeros_like(dW), dW).abs().max())

    n = 0
    for n in range(Tmax):
        U, info = equilibrate_t(lap(W), At, tt, kl, kh, Gt, U0=U)
        newton_iters.append(info["iters"])
        dW = drift(W, U)
        if n in snap_steps:
            r = proj_res(W, dW)
            keep = keep_U_steps is None or n in keep_U_steps
            snaps[n] = _snap(W.cpu().numpy(), dict(r=r, U=U.float().cpu().numpy() if keep else None))
        Wn = torch.clamp(W + dt * dW, min=0.0)
        delta = float((Wn - W).abs().max()) / dt
        done = delta < tol
        W = Wn
        if log_every and n % log_every == 0:
            print(f"  step {n} dW/dt {delta:.2e} newton {info['iters']} {time.time() - t0:.0f}s", flush=True)
        if done:
            n += 1
            break
    else:
        n = Tmax
    U, info = equilibrate_t(lap(W), At, tt, kl, kh, Gt, U0=U)
    dW = drift(W, U)
    r = proj_res(W, dW)
    Wn = W.cpu().numpy()
    snaps[n] = _snap(Wn, dict(r=r, U=U.float().cpu().numpy()))
    return dict(W=Wn, U=U.cpu().numpy(), steps=n, t=n * dt, residual=r, converged=bool(r < RES_TOL),
                seconds=time.time() - t0, snaps=snaps, dt=dt, equil_gmax=info["gmax"],
                newton_iters_mean=float(np.mean(newton_iters)) if newton_iters else 0.0,
                newton_iters_max=int(max(newton_iters)) if newton_iters else 0, device=str(dev))


def eval_responses(A, t, klo, khi, W, Ge, device=None, chunk=256):
    """Equilibrium responses on the evaluation bank (chunked), numpy float64."""
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    f = lambda x: torch.as_tensor(np.asarray(x, float), dtype=torch.float64, device=dev)  # noqa: E731
    N = A.shape[0]
    Wt, At = f(W), f(A)
    L = torch.diag(Wt.sum(1) + At.sum(1)) - Wt
    U, info = equilibrate_t(L, At, f(np.broadcast_to(t, (N,))), f(np.broadcast_to(klo, (N,))),
                            f(np.broadcast_to(khi, (N,))), f(Ge), chunk=chunk)
    return U.cpu().numpy(), info
