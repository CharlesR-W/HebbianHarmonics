"""Plastic attachments under the same distance rule with fixed input degree per neuron."""
import numpy as np, sys
sys.path.insert(0, 'scratch')
from onsite_proto import make_sigma, equilibrate, comps

def proj_simplex(V, z):
    # rows of V projected onto {a>=0, sum=z}
    n = V.shape[1]; Us = -np.sort(-V, axis=1); css = Us.cumsum(1) - z
    k = np.arange(1, n + 1); cond = Us - css / k > 0
    rho = cond.sum(1); tau = css[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - tau[:, None], 0)

def run(N=40, B=12, kind='blocks', b=0.05, alpha=1.0, alphaB=1.0, klo=0., khi=0., tspread=0., deg=1.0,
        S=256, T=3000, dt=0.2, seed=0, plasticA=True, lateral=True, Sig=None, init_sparse=None):
    rng = np.random.default_rng(seed)
    if Sig is None: Sig = make_sigma(B, kind, rng)
    B = len(Sig)
    G = rng.multivariate_normal(np.zeros(B), Sig, size=S)
    A = rng.dirichlet(np.ones(B), size=N) * deg
    t = tspread * rng.standard_normal(N)
    kl = klo * np.ones(N); kh = khi * np.ones(N)
    W = np.zeros((N, N))
    for step in range(T):
        U = equilibrate(W, A, G, t, kl, kh)
        Wn = W
        if lateral:
            D2 = ((U[:, :, None] - U[:, None, :]) ** 2).mean(0)
            dW = b - alpha * W - 0.5 * D2; np.fill_diagonal(dW, 0)
            Wn = np.maximum(W + dt * dW, 0)
        An = A
        if plasticA:
            D2B = ((U[:, :, None] - G[:, None, :]) ** 2).mean(0)   # (N,B)
            An = proj_simplex(A + dt * (-alphaB * A - 0.5 * D2B), deg)
        res = max(np.abs(Wn - W).max(), np.abs(An - A).max()) / dt
        W, A = Wn, An
        if res < 1e-7: break
    U = equilibrate(W, A, G, t, kl, kh)
    return dict(W=W, A=A, U=U, G=G, Sig=Sig, t=t, res=res, steps=step)

def report(r, tag, m=3):
    A, W, U = r['A'], r['W'], r['U']; B = A.shape[1]
    blk = np.arange(B) // m
    mass = np.stack([A[:, blk == k].sum(1) for k in range(B // m)], 1) / A.sum(1, keepdims=True)
    home = mass.argmax(1); pure = mass.max(1)
    nsup = (A > 1e-6).sum(1)
    lab = comps(W); Uc = U - U.mean(0); C = Uc.T @ Uc / len(U)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None); deff = ev.sum() ** 2 / (ev ** 2).sum()
    print(f"{tag} res={r['res']:.0e} st={r['steps']} alloc/block={np.bincount(home, minlength=B//m)} purity={pure.mean():.2f} "
          f"support={nsup.mean():.1f} K={lab.max()+1} sizes={sorted(np.bincount(lab))[::-1][:6]} deff={deff:.2f} var_u={np.diag(C).mean():.2f}")

if __name__ == '__main__':
    for aB in [0.1, 1, 10]:
        report(run(alphaB=aB, lateral=False), f'noLat aB={aB}')
    for b in [0.02, 0.1, 0.3]:
        report(run(alphaB=1, b=b), f'lat b={b} aB=1')
    for b in [0.02, 0.1]:
        report(run(alphaB=1, b=b, klo=3., tspread=.7), f'rect lat b={b} aB=1')
