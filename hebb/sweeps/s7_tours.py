"""Verify the suggested explorer tours (named parameter states) with N in 60-200, 3 seeds each.

Output: site/data/sweeps/tours.json — for each tour: the exact parameters, the covariance, and per
seed K (alive clusters >= 2% N), cluster sizes, centroids (rounded), dead count, residual.
"""
import numpy as np

from hebb.sweeps.common import Timer, cluster_embed, cov_block, cov_hier, cov_ring, pmap, save_json, simulate

V6 = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def S_of(spec):
    kind = spec["kind"]
    if kind == "matrix":
        return np.array(spec["Sigma"], float)
    if kind == "diag":
        return np.diag(spec["v"])
    if kind == "block12":
        return cov_block([4, 4, 4], 0.5, 0.1)[0]
    if kind == "hier8":
        return cov_hier((0.6, 0.3, 0.05), (2, 2, 2))
    if kind == "ring":
        return cov_ring(spec["B"], spec["ell"])
    raise ValueError(kind)


I3 = np.eye(3).tolist()
C3 = [[1, .3, .3], [.3, 1, .3], [.3, .3, 1]]
P3 = [[1, .6, 0], [.6, 1, 0], [0, 0, 1]]
A3 = (np.eye(3) - np.ones((3, 3)) / 3).tolist()
U3 = np.diag([1, 1.5, 2.5]).tolist()

TOURS = [
    # chapter 1 (B = 3)
    ("ch1", "consensus", dict(kind="matrix", Sigma=C3), dict(N=90, theta=0.1, gamma=1.0, wmin=1e-4),
     "positive correlations, theta below every covariance: one cluster near the centroid"),
    ("ch1", "neutral-identity", dict(kind="matrix", Sigma=I3), dict(N=90, theta=0.0, gamma=1.0, wmin=1e-4),
     "Sigma = I, theta = 0: every point is a fixed point for a lone neuron; lateral coupling makes consensus"),
    ("ch1", "three-corners", dict(kind="matrix", Sigma=I3), dict(N=90, theta=0.4, gamma=1.0, wmin=1e-4),
     "pruning drift above the initial mean covariance (1/3): three vertex clusters"),
    ("ch1", "corners-collective", dict(kind="matrix", Sigma=I3), dict(N=90, theta=0.2, gamma=1.0, wmin=1e-4),
     "same Sigma, theta = 0.2 < 1/3: the population moves together and picks ONE vertex"),
    ("ch1", "edge-midpoint", dict(kind="matrix", Sigma=P3), dict(N=90, theta=0.5, gamma=1.0, wmin=1e-4),
     "inputs 0,1 correlated 0.6: a cluster at the 0-1 edge midpoint plus a vertex cluster at input 2"),
    ("ch1", "midpoint-splits", dict(kind="matrix", Sigma=P3), dict(N=90, theta=0.65, gamma=1.0, wmin=1e-4),
     "theta above the 0.6 correlation: the midpoint part is unstable, three vertices"),
    ("ch1", "anticorrelated", dict(kind="matrix", Sigma=A3), dict(N=90, theta=0.02, gamma=1.0, wmin=1e-4),
     "Sigma = I - 11^T/3: vertices with (almost) no pruning drift — pruning by anticorrelation"),
    ("ch1", "winner-takes-all", dict(kind="matrix", Sigma=U3), dict(N=90, theta=0.2, gamma=1.0, wmin=1e-4),
     "unequal variances 1, 1.5, 2.5; collective regime: everyone goes to the loudest input"),
    ("ch1", "independent-tilt", dict(kind="matrix", Sigma=U3), dict(N=90, theta=0.2, gamma=0.0, wmin=1e-4),
     "same, no lateral plasticity: all three inputs populated, louder inputs get more neurons"),
    ("ch1", "partial-extinction", dict(kind="matrix", Sigma=U3), dict(N=90, theta=1.2, gamma=1.0, wmin=1e-4),
     "theta above input 0's variance: that input cannot host a cluster"),
    ("ch1", "floor-too-high", dict(kind="matrix", Sigma=I3), dict(N=90, theta=0.6, gamma=1.0, wmin=1e-2),
     "N*wmin ~ 1: the floor conductance leaks every neuron to the centroid; the network dies"),
    # chapter 2 (many inputs)
    ("ch2", "blocks-one", dict(kind="block12"), dict(N=150, theta=0.05, gamma=1.0, wmin=1e-4),
     "theta below the across-block correlation 0.1: one cluster on all 12 inputs"),
    ("ch2", "blocks-three", dict(kind="block12"), dict(N=150, theta=0.35, gamma=1.0, wmin=1e-4),
     "0.1 < theta < 0.5 and above kappa0 ~ 0.27: three block clusters"),
    ("ch2", "blocks-twelve", dict(kind="block12"), dict(N=150, theta=0.7, gamma=1.0, wmin=1e-4),
     "theta above the within-block correlation: twelve vertex clusters"),
    ("ch2", "blocks-collective", dict(kind="block12"), dict(N=150, theta=0.2, gamma=1.0, wmin=1e-4),
     "three stable parts exist but theta < kappa0: one block wins everything"),
    ("ch2", "hier-pairs", dict(kind="hier8"), dict(N=150, theta=0.45, gamma=1.0, wmin=1e-4),
     "ultrametric 8: theta between cousin (0.3) and sibling (0.6) levels: four pair clusters"),
    ("ch2", "ring-tiling-independent", dict(kind="ring", B=16, ell=1.5), dict(N=160, theta=0.4, gamma=0.0, wmin=1e-4),
     "ring covariance (Fourier eigenvectors): localized bumps of width 4 at every position"),
    ("ch2", "ring-packing", dict(kind="ring", B=16, ell=1.5), dict(N=160, theta=0.4, gamma=1.0, wmin=1e-4),
     "same with lateral plasticity: only a non-overlapping packing of bumps survives"),
    # chapter 4 (homeostasis)
    ("ch4", "filter-steep", dict(kind="diag", v=V6), dict(N=200, theta=0.05, gamma=0.0, wmin=1e-4),
     "independent neurons, low theta: allocation ∝ v^beta with large beta (near winner-take-all)"),
    ("ch4", "filter-soft", dict(kind="diag", v=V6), dict(N=200, theta=0.4, gamma=0.0, wmin=1e-4),
     "independent neurons, theta near the smallest variance: beta ~ 1.5"),
    ("ch4", "filter-collective", dict(kind="diag", v=V6), dict(N=200, theta=0.2, gamma=1.0, wmin=1e-4),
     "lateral plasticity on, theta below kappa0: everything to the loudest input"),
    ("ch4", "floor-leak", dict(kind="block12"), dict(N=150, theta=0.65, gamma=1.0, wmin=1e-3),
     "N*wmin = 0.15 at high theta: extinction (compare wmin = 1e-5)"),
    ("ch4", "floor-leak-ok", dict(kind="block12"), dict(N=150, theta=0.65, gamma=1.0, wmin=1e-5),
     "same with N*wmin = 0.0015: twelve vertex clusters"),
]


def job(args):
    idx, seed = args
    ch, name, cs, p, note = TOURS[idx]
    S = S_of(cs)
    r = simulate(S, p["N"], p["theta"], gamma=p["gamma"], wmin=p["wmin"], seed=seed, Tmax=40000)
    l, sizes, cents = cluster_embed(r["H"], 0.1)
    dead = r["Wib"].max(1) <= 1.5 * p["wmin"]
    keep = [k for k in range(len(sizes)) if not dead[l == k].all()]
    K = sum(1 for k in keep if sizes[k] >= max(2, 0.02 * p["N"]))
    return dict(idx=idx, seed=seed, K=K, sizes=[int(sizes[k]) for k in keep],
                cents=[np.round(cents[k], 3) for k in keep][:16], dead=int(dead.sum()), res=r["res"],
                steps=r["steps"])


def main():
    jobs = [(i, s) for i in range(len(TOURS)) for s in range(3)]
    jobs.sort(key=lambda j: -TOURS[j[0]][3]["N"])
    with Timer(f"tours ({len(jobs)})"):
        R = pmap(job, jobs)
    out = []
    for i, (ch, name, cs, p, note) in enumerate(TOURS):
        rr = sorted([r for r in R if r["idx"] == i], key=lambda r: r["seed"])
        out.append(dict(chapter=ch, name=name, cov=cs, params=dict(p, lam=1, eta=1, dt=0.05, ii_scale=0.0, ib_scale=0.2, p=1.0),
                        note=note, runs=rr))
        print(f"{ch} {name:24s} K={[r['K'] for r in rr]} sizes={[r['sizes'][:6] for r in rr]} dead={[r['dead'] for r in rr]} "
              f"res={max(r['res'] for r in rr):.1e} steps={[r['steps'] for r in rr]}")
        print("      cents(seed0):", [list(c) for c in rr[0]["cents"][:4]] if len(cs.get("Sigma", [])) == 3 else "")
    save_json("tours", dict(tours=out), dict(
        script="hebb/sweeps/s7_tours.py", seeds=[0, 1, 2],
        description="Suggested explorer presets, each verified on 3 seeds. K = alive clusters with >= max(2, 2% N) "
                    "members (average-linkage L1 cut 0.1). Mean-field, Tmax 40000 steps of dt=0.05, conv tol 1e-5."))


if __name__ == "__main__":
    main()
