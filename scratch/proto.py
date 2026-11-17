"""Prototype: Hebbian resistor network, expected-covariance (mean-field) dynamics."""
import numpy as np, sys

def harmonic(Wii, Wib):
    deg = Wii.sum(1) + Wib.sum(1)
    L = np.diag(deg) - Wii
    return np.linalg.solve(L, Wib)

def run(Sigma, N=60, theta=0.2, lam=1.0, eta=1.0, wmin=1e-3, p=1.0, dt=0.05, T=4000, seed=0, kappa=0.0, dstar=1.0, log=None):
    rng = np.random.default_rng(seed)
    B = Sigma.shape[0]
    Wii = rng.uniform(0.0, 0.2, (N, N)); Wii = (Wii + Wii.T) / 2; np.fill_diagonal(Wii, 0)
    Wib = rng.uniform(0.0, 0.2, (N, B))
    hist = []
    for t in range(T):
        H = harmonic(Wii, Wib)
        Cii = H @ Sigma @ H.T; Cib = H @ Sigma
        deg = Wii.sum(1) + Wib.sum(1)
        # homeostatic synaptic scaling: multiplicative push of each neuron's degree to dstar
        sc_ii = -kappa * ((deg[:, None] - dstar) + (deg[None, :] - dstar)) * Wii
        sc_ib = -kappa * (deg[:, None] - dstar) * Wib
        Wii = Wii + dt * (eta * (Cii - theta) - lam * Wii**p + sc_ii)
        Wib = Wib + dt * (eta * (Cib - theta) - lam * Wib**p + sc_ib)
        Wii = np.maximum(Wii, wmin); np.fill_diagonal(Wii, 0); Wib = np.maximum(Wib, wmin)
        if log and t % log == 0: hist.append(H.copy())
    return Wii, Wib, harmonic(Wii, Wib), hist

def nclusters(H, tol=0.05):
    # greedy cluster count in L1 distance
    cents = []
    counts = []
    for h in H:
        for k, c in enumerate(cents):
            if np.abs(h - c).sum() < tol: counts[k] += 1; break
        else: cents.append(h); counts.append(1)
    return np.array(cents), np.array(counts)

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    Sigma = np.eye(3)
    for theta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]:
        for lam in [0.1, 1.0, 10.0]:
            Wii, Wib, H, _ = run(Sigma, theta=theta, lam=lam)
            c, n = nclusters(H)
            o = np.argsort(-n)[:6]
            print(f"theta={theta} lam={lam}: {len(n)} clusters; top:", [(c[k].round(2).tolist(), int(n[k])) for k in o])
