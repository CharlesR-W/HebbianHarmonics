import numpy as np
from proto import harmonic, nclusters
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def run(Sigma, N=60, theta=0.2, lam=1.0, wmin=1e-3, dt=0.05, T=6000, seed=0, gii=1.0, init_ii=0.0, kappa=0.0, dstar=1.0, noise=0.0):
    rng = np.random.default_rng(seed); B = Sigma.shape[0]
    Wii = init_ii*rng.uniform(0, 1, (N, N)); Wii = (Wii + Wii.T)/2; np.fill_diagonal(Wii, 0)
    Wib = rng.exponential(0.2, (N, B))
    for t in range(T):
        H = harmonic(np.maximum(Wii,wmin*(1-np.eye(N))), np.maximum(Wib,wmin))
        Cii = H@Sigma@H.T; Cib = H@Sigma
        deg = Wii.sum(1)+Wib.sum(1)
        Wii = Wii + dt*(gii*(Cii-theta) - lam*Wii - kappa*((deg[:,None]-dstar)+(deg[None,:]-dstar))*Wii)
        Wib = Wib + dt*((Cib-theta) - lam*Wib - kappa*(deg[:,None]-dstar)*Wib)
        Wii = np.maximum(Wii, 0); np.fill_diagonal(Wii, 0); Wib = np.maximum(Wib, 0)
    return Wii, Wib, harmonic(np.maximum(Wii,wmin*(1-np.eye(N))), np.maximum(Wib,wmin))
def show(tag, H):
    c, n = nclusters(H, 0.1); o = np.argsort(-n)[:8]
    print(tag, f"{len(n)} clusters:", [(c[k].round(2).tolist(), int(n[k])) for k in o])
S3 = np.eye(3)
for theta in [0.05, 0.2, 0.4]:
    for gii in [1.0, 1/60]:
        for wmin in [1e-3, 1e-5]:
            _,_,H = run(S3, theta=theta, gii=gii, wmin=wmin); show(f"I3 theta={theta} gii={gii:.3f} wmin={wmin}", H)
# correlated: nodes 0,1 correlated 0.6, node 2 indep, unequal variance
S = np.array([[1,.6,0],[.6,1,0],[0,0,.7]])
for theta in [0.05, 0.2, 0.4, 0.65]:
    _,_,H = run(S, theta=theta, gii=1.0, wmin=1e-4); show(f"corr theta={theta}", H)
# 12 boundary nodes in 3 blocks of 4 with within-corr .5 and cross .1
B=12; S = np.full((B,B),.1); 
for k in range(3): S[4*k:4*k+4,4*k:4*k+4]=.5
np.fill_diagonal(S,1)
for theta in [0.05, 0.2, 0.3, 0.45, 0.6]:
    _,_,H = run(S, N=120, theta=theta, wmin=1e-4); show(f"block12 theta={theta}", H)
