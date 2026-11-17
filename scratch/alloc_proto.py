"""Function-space particles: similarity matching + finite-range fusion, fixed per-neuron variance nu.
x_i in R^B are whitened response vectors (F_i = x_i . Sigma^{-1/2} g), |x_i|^2 = nu."""
import numpy as np
def run(lam, N=60, nu=0.1, b=0.0, alpha=1.0, T=20000, lr=0.02, seed=0):
    rng = np.random.default_rng(seed); B = len(lam); Sig = np.diag(lam)
    X = rng.standard_normal((N, B)); X *= np.sqrt(nu) / np.linalg.norm(X, axis=1, keepdims=True)
    for t in range(T):
        Gm = X.T @ X
        grad = 2 * X @ (Gm - Sig)                     # d/dX 1/2|Sig - X^T X|^2
        if b > 0:
            D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
            k = np.maximum(b - D2 / 2, 0) / alpha; np.fill_diagonal(k, 0)   # = w_ij*
            grad += k.sum(1)[:, None] * X - k @ X      # d/dX sum_{i<j} -(1/2a)[b-D2/2]^2 = sum_j w_ij (x_i-x_j)
        X = X - lr * grad
        X *= np.sqrt(nu) / np.linalg.norm(X, axis=1, keepdims=True)
    return X
def report(X, lam, nu, tag):
    N = len(X); q = np.diag(X.T @ X); off = np.abs(X.T @ X - np.diag(q)).max()
    # water-filling prediction
    mu_grid = np.linspace(-5, lam.max(), 200001); tot = np.maximum(lam[None] - mu_grid[:, None], 0).sum(1)
    mu = mu_grid[np.argmin(np.abs(tot - N * nu))]; qpred = np.maximum(lam - mu, 0)
    purity = (X ** 2).max(1) / nu; home = np.abs(X).argmax(1)
    cnt = np.bincount(home[purity > 0.9], minlength=len(lam))
    print(f"{tag}\n  q     ={np.round(q,3)}\n  qpred ={np.round(qpred,3)}  offdiag={off:.3f}\n  pure frac={np.mean(purity>0.9):.2f} n_a(pure)={cnt}  n_pred={np.round(qpred/nu,1)}")
lam = np.array([3.0, 2.0, 1.5, 1.0, 0.6, 0.3, 0.1, 0.05])
for b in [0.0, 0.02, 0.1]:
    for seed in [0, 1]:
        X = run(lam, b=b, seed=seed); report(X, lam, 0.1, f'b={b} seed={seed}')
