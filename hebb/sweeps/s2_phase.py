"""Phase diagrams: number of populated clusters K over (theta, gamma), (theta, wmin), (theta, ii_scale),
and N-dependence (does N*wmin collapse the wmin axis?). Covariance: block12 (3 blocks of 4, within 0.5,
across 0.1) and hier8. Also records the predicted 'collective' boundary theta = gamma*kappa0, with
kappa0 = mean_{i != j} h_i(0)^T Sigma h_j(0) at initialisation.

Output: site/data/sweeps/phase.json.  ~10 min on 3 procs.
"""
import sys

import numpy as np

from hebb.core import harmonic, init_state
from hebb.sweeps.common import Timer, cluster_embed, cov_block, cov_hier, pmap, save_json, simulate

# grid avoids the exact correlation levels of the test covariances (0.05, 0.1, 0.3, 0.5, 0.6), where parts are marginal
THETA = [0.02, 0.07, 0.12, 0.17, 0.22, 0.27, 0.32, 0.37, 0.42, 0.47, 0.53, 0.57, 0.63, 0.7, 0.8]
GAMMA = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
WMIN = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
IISC = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
SEEDS = [0, 1, 2]
SEEDS_OF = dict(block12=[0, 1, 2], hier8=[0, 1])
BASE = dict(N=120, gamma=1.0, wmin=1e-4, ii_scale=0.0, ib_scale=0.2)


def cov(name):
    if name == "block12":
        return cov_block([4, 4, 4], 0.5, 0.1)[0]
    return cov_hier((0.6, 0.3, 0.05), (2, 2, 2))


def summarize(r, wmin, N):
    l, sizes, cents = cluster_embed(r["H"], 0.1)
    dead = r["Wib"].max(1) <= 1.5 * wmin
    alive = [int(s) for k, s in enumerate(sizes) if not dead[l == k].all()]
    K = sum(1 for s in alive if s >= max(2, 0.02 * N))
    K05 = sum(1 for s in np.bincount(cluster_embed(r["H"], 0.05)[0]) if s >= max(2, 0.02 * N))
    K20 = sum(1 for s in np.bincount(cluster_embed(r["H"], 0.2)[0]) if s >= max(2, 0.02 * N))
    return dict(K=K, K05=K05, K20=K20, dead=float(dead.mean()), maxfrac=max(alive) / N if alive else 0.0,
                res=r["res"], steps=r["steps"])


def job(args):
    cname, axis, theta, val, seed, kw = args
    S = cov(cname)
    p = dict(BASE, **kw)
    r = simulate(S, p["N"], theta, gamma=p["gamma"], wmin=p["wmin"], seed=seed, ii_scale=p["ii_scale"],
                 ib_scale=p["ib_scale"], Tmax=30000)
    return dict(cov=cname, axis=axis, theta=theta, val=val, seed=seed, **summarize(r, p["wmin"], p["N"]))


def kappa0(S, N, seed, ii_scale=0.0, ib_scale=0.2):
    rng = np.random.default_rng(seed)
    Wii, Wib, Mii, Mib = init_state(N, S.shape[0], rng, 1.0, ii_scale, ib_scale)
    H = harmonic(np.maximum(Wii, 1e-4 * Mii), Wib)
    C = H @ S @ H.T
    return float((C.sum() - np.trace(C)) / (N * (N - 1)))


def main(covs=("block12", "hier8")):
    jobs = []
    for c in covs:
        for th in THETA:
            for s in SEEDS_OF[c]:
                for g in GAMMA:
                    jobs.append((c, "gamma", th, g, s, dict(gamma=g)))
                if c != "block12":
                    continue
                for w in WMIN:
                    jobs.append((c, "wmin", th, w, s, dict(wmin=w)))
                if c == "block12":
                    for ii in IISC:
                        jobs.append((c, "ii_scale", th, ii, s, dict(ii_scale=ii)))
                    for Nn in (60, 240):
                        for w in (1e-4, 1e-3):
                            if Nn == 240 and s == 2:
                                continue
                            jobs.append((c, f"N{Nn}", th, w, s, dict(N=Nn, wmin=w)))
    jobs.sort(key=lambda j: -j[5].get("N", 120))  # big ones first
    with Timer(f"phase ({len(jobs)})"):
        R = pmap(job, jobs)
    out = {}
    for c in covs:
        S = cov(c)
        axes = {}
        for axis, vals in (("gamma", GAMMA), ("wmin", WMIN), ("ii_scale", IISC), ("N60", [1e-4, 1e-3]),
                           ("N240", [1e-4, 1e-3])):
            rr = [r for r in R if r["cov"] == c and r["axis"] == axis]
            if not rr:
                continue
            grid = {}
            for key in ("K", "K05", "K20", "dead", "maxfrac", "res"):
                grid[key] = [[[r[key] for r in rr if r["theta"] == th and r["val"] == v]
                              for th in THETA] for v in vals]
            axes[axis] = dict(values=vals, **grid)
        k0 = [kappa0(S, 120, s) for s in SEEDS_OF[c]]
        k0ii = {str(ii): float(np.mean([kappa0(S, 120, s, ii_scale=ii) for s in SEEDS])) for ii in IISC}
        out[c] = dict(axes=axes, kappa0=float(np.mean(k0)), kappa0_vs_ii_scale=k0ii)
        for axis in axes:
            print(c, axis)
            for v, KK in zip(axes[axis]["values"], axes[axis]["K"]):
                print(f"  {v:<8g}", " ".join(f"{np.mean(x):4.1f}" for x in KK))
    meta = dict(script="hebb/sweeps/s2_phase.py",
                description="Number of populated clusters K (alive clusters with >= max(2, 2% N) members; average-linkage "
                            "L1 cut 0.1; K05/K20 = cuts 0.05/0.2) over theta x second knob. Grids are [value][theta][seed].",
                theta=THETA, seeds=SEEDS_OF, base=BASE, note_seeds="N240 uses seeds 0,1 only; hier8 only has the gamma axis",
                axes=dict(gamma=GAMMA, wmin=WMIN, ii_scale=IISC, N60="N=60, values = wmin", N240="N=240, values = wmin"),
                params=dict(lam=1, eta=1, dt=0.05, p=1.0, conv_tol=1e-5, Tmax=30000),
                covs=dict(block12="3 blocks of 4, within 0.5, across 0.1", hier8="ultrametric 8: 0.6/0.3/0.05"),
                kappa0="mean off-diagonal init covariance h_i^T Sigma h_j; theory: collective (one-cluster) regime for "
                       "theta < gamma*kappa0 when lateral weights start at the floor")
    save_json("phase", dict(covs=out), meta)


if __name__ == "__main__":
    main()
