"""Q4: multistability and dynamics.

(a) Hysteresis: theta swept slowly up then down (state carried over), block12 and hier8, N=120,
    gamma in {1, 0}, wmin=1e-4. At each theta step run until residual < 1e-4 or 3000 steps.
(b) Convergence time vs theta (steps to max|dw/dt| < 1e-5), block12, 4 seeds.
(c) Mean-field vs stochastic: block12 at theta in {0.15, 0.3, 0.45}, batch in {8, 32, 128, 512}:
    K, allocation per block, distance of learned centroids to mean-field parts, same seeds.
(d) Init: ii_scale vs ib_scale grid at theta = 0.3 and 0.45 (K, dead fraction, max cluster fraction).

Output: site/data/sweeps/hysteresis.json, dynamics.json.
"""
import numpy as np

from hebb.sweeps.common import Timer, cluster_embed, cov_block, cov_hier, pmap, residual, save_json, simulate

S12 = cov_block([4, 4, 4], 0.5, 0.1)[0]
COVS = {"block12": S12, "hier8": cov_hier((0.6, 0.3, 0.05), (2, 2, 2))}
N = 120
WMIN = 1e-4


def state_summary(r, theta, S, wmin=WMIN):
    l, sizes, cents = cluster_embed(r["H"], 0.1)
    dead = r["Wib"].max(1) <= 1.5 * wmin
    alive = [int(s) for k, s in enumerate(sizes) if not dead[l == k].all()]
    K = sum(1 for s in alive if s >= max(2, 0.02 * N))
    H = r["H"]
    E = float(np.mean(np.einsum("ib,bc,ic->i", H, S, H)))
    supp = float(np.mean((H > 0.02).sum(1)))
    return dict(theta=theta, K=K, dead=float(dead.mean()), maxfrac=max(alive) / N if alive else 0.0, E=E,
                supp=supp, res=r["res"], steps=r["steps"], sizes=sorted(alive, reverse=True)[:16])


def hyst_job(args):
    cname, gamma, seed = args
    S = COVS[cname]
    th_up = np.round(np.arange(0.02, 0.801, 0.02), 3)
    path = list(th_up) + list(th_up[::-1][1:])
    st = None
    out = []
    for i, th in enumerate(path):
        r = simulate(S, N, th, gamma=gamma, wmin=WMIN, seed=seed, state=st, Tmax=3000, tol=1e-4, check=100)
        st = (r["Wii"], r["Wib"], r["Mii"], r["Mib"])
        d = state_summary(r, float(th), S)
        d["dir"] = "up" if i < len(th_up) else "down"
        out.append(d)
    return dict(cov=cname, gamma=gamma, seed=seed, path=out)


def conv_job(args):
    theta, seed = args
    r = simulate(S12, N, theta, gamma=1.0, wmin=WMIN, seed=seed, Tmax=60000, tol=1e-5, check=50)
    return dict(theta=theta, seed=seed, steps=r["steps"], conv=r["converged"], res=r["res"],
                K=state_summary(r, theta, S12)["K"])


def parts_of(H, S12lab):
    l, sizes, cents = cluster_embed(H, 0.2)
    return l, sizes, cents


def stoch_job(args):
    theta, batch, seed = args
    lab = np.repeat(np.arange(3), 4)
    if batch:
        r = simulate(S12, N, theta, gamma=1.0, wmin=WMIN, seed=seed, batch=batch, Tmax=16000, record=4000)
    else:
        r = simulate(S12, N, theta, gamma=1.0, wmin=WMIN, seed=seed, Tmax=30000)
    l, sizes, cents = cluster_embed(r["H"], 0.2)
    dead = r["Wib"].max(1) <= 1.5 * WMIN
    # allocation by planted block (>=80% mass) and by singleton input
    blk = np.zeros(4)
    for k, n in enumerate(sizes):
        if dead[l == k].all():
            blk[3] += n; continue
        m = np.array([cents[k][lab == j].sum() for j in range(3)])
        blk[m.argmax() if m.max() > 0.8 else 3] += n
    # per-neuron assignment trajectory: how many neurons change block between recorded snapshots
    hops = None
    if batch and r["hist"]:
        asg = [np.array([np.argmax([h[lab == j].sum() for j in range(3)]) for h in H]) for H in r["hist"] + [r["H"]]]
        hops = int(sum((a != b).sum() for a, b in zip(asg[:-1], asg[1:])))
    K = sum(1 for k, n in enumerate(sizes) if n >= max(2, 0.02 * N) and not dead[l == k].all())
    return dict(theta=theta, batch=batch, seed=seed, K=K, alloc=blk / N, hops=hops,
                cents=np.round(cents[sizes >= 3], 3))


def init_job(args):
    theta, ii, ib, seed = args
    r = simulate(S12, N, theta, gamma=1.0, wmin=WMIN, seed=seed, ii_scale=ii, ib_scale=ib, Tmax=30000)
    return dict(ii=ii, ib=ib, seed=seed, **{k: v for k, v in state_summary(r, theta, S12).items() if k != "sizes"})


def main():
    hj = [(c, g, s) for c in COVS for g in (1.0, 0.0) for s in (0, 1)]
    cj = [(th, s) for th in [0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.48, 0.5, 0.52, 0.55, 0.6, 0.7]
          for s in range(4)]
    sj = [(th, b, s) for th in (0.15, 0.3, 0.45) for b in (None, 8, 32, 128, 512) for s in range(3)]
    ij = [(th, ii, ib, s) for th in (0.3, 0.45) for ii in (0.0, 0.03, 0.1, 0.3, 1.0) for ib in (0.02, 0.2, 1.0) for s in range(3)]
    with Timer("all dynamics"):
        R = pmap(_dispatch, [("h", j) for j in hj] + [("s", j) for j in sj] + [("c", j) for j in cj] + [("i", j) for j in ij])
    H = [r for t, r in R if t == "h"]
    C = [r for t, r in R if t == "c"]
    St = [r for t, r in R if t == "s"]
    I = [r for t, r in R if t == "i"]
    save_json("hysteresis", dict(loops=H), dict(
        script="hebb/sweeps/s5_dynamics.py (a)",
        description="theta swept 0.02 -> 0.80 -> 0.02 in steps of 0.02, state carried over; at each theta up to 3000 "
                    "steps (dt=0.05) or until max|dw/dt| < 1e-4. K = alive clusters >= 2% N (avg-linkage L1 cut 0.1); "
                    "E = mean h_i^T Sigma h_i; supp = mean number of inputs with h_ib > 0.02.",
        N=N, wmin=WMIN, params=dict(lam=1, eta=1, dt=0.05, ii_scale=0.0, ib_scale=0.2, p=1.0),
        covs=dict(block12="3 blocks of 4, within 0.5, across 0.1", hier8="ultrametric 8: 0.6/0.3/0.05")))
    for h in H:
        print(h["cov"], h["gamma"], h["seed"], " ".join(f"{p['theta']:.2f}{p['dir'][0]}:{p['K']}" for p in h["path"][::2]))
    for th in sorted(set(c["theta"] for c in C)):
        cc = [c for c in C if c["theta"] == th]
        print("conv", th, [c["steps"] for c in cc], [c["K"] for c in cc])
    for r in St:
        print("stoch", r["theta"], r["batch"], r["seed"], r["K"], np.round(r["alloc"], 2), r["hops"])
    for r in I:
        print("init", r["theta"], r["ii"], r["ib"], r["seed"], r["K"], round(r["dead"], 2), round(r["maxfrac"], 2))
    save_json("dynamics", dict(convergence=C, stochastic=St, init=I), dict(
        script="hebb/sweeps/s5_dynamics.py (b-d)",
        convergence="block12, N=120, gamma=1, wmin=1e-4: steps (dt=0.05) until max|dW|/(50 dt) < 1e-5 (cap 60000)",
        stochastic="block12, gamma=1: mean-field (batch=null) vs minibatch covariance (Gaussian samples, centred, "
                   "unbiased); 16000 steps dt=0.05; alloc = fraction of N in clusters with >=80% mass on block 0/1/2, "
                   "last entry = unassigned or dead; hops = neuron block-assignment changes between snapshots at steps "
                   "4000, 8000, 12000, 16000",
        init="block12, gamma=1: Wii ~ U(0, ii) dense, Wib ~ Exp(ib); K, dead fraction, largest cluster fraction",
        N=N, wmin=WMIN))


def _dispatch(a):
    t, j = a
    f = dict(h=hyst_job, c=conv_job, s=stoch_job, i=init_job)[t]
    return t, f(j)


if __name__ == "__main__":
    main()
