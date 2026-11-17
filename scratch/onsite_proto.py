"""Prototype: conductance network with on-site potentials, distance-based rule.
E = 1/2 sum w_ij (u_i-u_j)^2 + 1/2 sum a_ib (u_i-g_b)^2 + sum V_i(u_i)
V_i(u) = k_lo/2 (u-t_i)_-^2 + k_hi/2 (u-t_i)_+^2   (two-slope: convex, piecewise linear V')
dw_ij = b - alpha w - 1/2 E (u_i-u_j)^2, w>=0.  A (I-B) fixed.
"""
import numpy as np, sys

def make_sigma(B, kind, rng):
    if kind == 'blocks':
        S = np.full((B, B), 0.0); m = B // 4
        var = [2.0, 1.0, 0.5, 0.25]
        for k in range(4):
            S[k*m:(k+1)*m, k*m:(k+1)*m] = 0.8 * var[k]
            S[range(k*m, (k+1)*m), range(k*m, (k+1)*m)] = var[k]
        return S
    if kind == 'power':
        Q, _ = np.linalg.qr(rng.standard_normal((B, B)))
        lam = 1.0 / np.arange(1, B + 1) ** 1.0
        return (Q * lam) @ Q.T * B / lam.sum()

def equilibrate(W, A, G, t, klo, khi, iters=60):
    """G: (S,B) samples. returns U (S,N). Newton on convex energy."""
    N = W.shape[0]; d = W.sum(1) + A.sum(1)
    L = np.diag(d) - W
    rhs = G @ A.T                      # (S,N)
    U = np.linalg.solve(L + np.diag(np.full(N, .5*(klo+khi))), rhs.T).T
    for it in range(iters):
        x = U - t
        k = np.where(x < 0, klo, khi)  # (S,N)
        grad = U @ L.T + k * x - rhs
        if np.abs(grad).max() < 1e-9: break
        # batched Newton: solve (L + diag(k_s)) dU = grad
        Hs = L[None] + np.einsum('sn,nm->snm', k, np.eye(N))
        dU = np.linalg.solve(Hs, grad[..., None])[..., 0]
        U = U - dU
    return U

def run(N=40, B=12, kind='blocks', b=0.05, alpha=1.0, klo=0.0, khi=0.0, tspread=0.0,
        deg=1.0, sparse=3, S=256, T=1500, dt=0.2, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    Sig = make_sigma(B, kind, rng)
    G = rng.multivariate_normal(np.zeros(B), Sig, size=S)
    A = np.zeros((N, B))
    for i in range(N):
        idx = rng.choice(B, sparse, replace=False)
        A[i, idx] = rng.dirichlet(np.ones(sparse)) * deg
    t = tspread * rng.standard_normal(N)
    klo_i = klo * np.ones(N); khi_i = khi * np.ones(N)
    W = np.zeros((N, N))
    for step in range(T):
        U = equilibrate(W, A, G, t, klo_i, khi_i)
        D2 = ((U[:, :, None] - U[:, None, :]) ** 2).mean(0)
        dW = b - alpha * W - 0.5 * D2
        np.fill_diagonal(dW, 0)
        Wn = np.maximum(W + dt * dW, 0)
        res = np.abs(Wn - W).max() / dt
        W = Wn
        if res < 1e-7: break
    U = equilibrate(W, A, G, t, klo_i, khi_i)
    return dict(W=W, A=A, U=U, G=G, Sig=Sig, t=t, res=res, steps=step)

def comps(W, eps=1e-6):
    N = len(W); lab = -np.ones(N, int); c = 0
    for i in range(N):
        if lab[i] >= 0: continue
        st = [i]; lab[i] = c
        while st:
            j = st.pop()
            for k in np.where(W[j] > eps)[0]:
                if lab[k] < 0: lab[k] = c; st.append(k)
        c += 1
    return lab

def summarize(r, tag=''):
    W, U = r['W'], r['U']
    lab = comps(W); K = lab.max() + 1
    Uc = U - U.mean(0)
    C = Uc.T @ Uc / len(U)
    ev = np.linalg.eigvalsh(C)[::-1]; ev = np.clip(ev, 0, None)
    deff = ev.sum() ** 2 / (ev ** 2).sum()
    sizes = np.bincount(lab)
    # within-cluster diversity: d_eff of each cluster's covariance, and mean pairwise corr
    div = []
    for k in range(K):
        idx = np.where(lab == k)[0]
        if len(idx) < 2: continue
        e = np.clip(np.linalg.eigvalsh(C[np.ix_(idx, idx)]), 0, None)
        div.append((len(idx), e.sum() ** 2 / (e ** 2).sum()))
    print(f"{tag} res={r['res']:.1e} steps={r['steps']} K={K} sizes={sorted(sizes)[::-1][:10]} d_eff={deff:.2f} "
          f"within(n,deff)={[(n, round(d,2)) for n,d in sorted(div)[::-1][:5]]} nedges={(W>1e-6).sum()//2}")
    return lab

if __name__ == '__main__':
    for b in [0.01, 0.03, 0.1, 0.3, 1.0]:
        summarize(run(b=b), f'linear b={b}')
    for b in [0.03, 0.1, 0.3]:
        summarize(run(b=b, klo=3.0, khi=0.0, tspread=0.7), f'rect  b={b}')
