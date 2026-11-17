"""Q3: diversity within a cluster ("reservoir of random functions"?).

block12 at theta=0.3 (three block clusters), N=120 default, gamma=1, wmin=1e-4.
Within-cluster spread: (a) mean L1 distance of members' h to the cluster centroid;
(b) functional: 1 - mean pairwise voltage correlation between members, corr = h_i'Sh_j/sqrt(..);
(c) participation ratio of the within-cluster voltage covariance H_k S H_k^T (1 = all copies).
Knobs: wmin, gamma, minibatch size (stochastic; averaged over the last 20% of the run), N,
and, as a labelled comparison only, a fixed sparse potential-synapse mask p < 1.

Output: site/data/sweeps/diversity.json.
"""
import numpy as np

from hebb.sweeps.common import Timer, cluster_embed, cov_block, pmap, save_json, simulate

S = cov_block([4, 4, 4], 0.5, 0.1)[0]
THETA = 0.3
BASE = dict(N=120, gamma=1.0, wmin=1e-4, batch=None, p=1.0)
SEEDS = [0, 1, 2]


def div_stats(H, wmin_dead=None, Wib=None):
    lab, sizes, cents = cluster_embed(H, 0.2)
    l1, fdiv, pr = [], [], []
    for k, n in enumerate(sizes):
        if n < 3:
            continue
        X = H[lab == k]
        l1.append(np.abs(X - X.mean(0)).sum(1).mean())
        C = X @ S @ X.T
        d = np.sqrt(np.diag(C))
        R = C / np.outer(d, d)
        iu = np.triu_indices(len(X), 1)
        fdiv.append(1 - R[iu].mean())
        ev = np.clip(np.linalg.eigvalsh(C), 0, None)
        pr.append(ev.sum() ** 2 / (ev ** 2).sum())
    # block level: all neurons with >= 80% of their mass on one planted block (the "mode" in the note's sense)
    lab_b = np.repeat(np.arange(3), 4)
    mass = np.stack([H[:, lab_b == j].sum(1) for j in range(3)], 1)
    bpr, bdiv = [], []
    for j in range(3):
        X = H[mass[:, j] >= 0.8]
        if len(X) < 3:
            continue
        C = X @ S @ X.T
        d = np.sqrt(np.diag(C)); R = C / np.outer(d, d)
        bdiv.append(1 - R[np.triu_indices(len(X), 1)].mean())
        ev = np.clip(np.linalg.eigvalsh(C), 0, None)
        bpr.append(ev.sum() ** 2 / (ev ** 2).sum())
    return dict(l1=float(np.mean(l1)) if l1 else None, fdiv=float(np.mean(fdiv)) if fdiv else None,
                pr=float(np.mean(pr)) if pr else None, K=int((sizes >= 3).sum()),
                block_pr=float(np.mean(bpr)) if bpr else None, block_fdiv=float(np.mean(bdiv)) if bdiv else None)


def job(args):
    axis, val, seed, kw = args
    p = dict(BASE, **kw)
    if p["batch"]:
        T = 16000
        r = simulate(S, p["N"], THETA, gamma=p["gamma"], wmin=p["wmin"], seed=seed, p=p["p"], batch=p["batch"],
                     Tmax=T, record=400)
        hs = [div_stats(H) for H in r["hist"][int(0.8 * len(r["hist"])):]]
        st = {k: float(np.mean([h[k] for h in hs if h[k] is not None])) for k in ("l1", "fdiv", "pr", "K", "block_pr", "block_fdiv")}
    else:
        r = simulate(S, p["N"], THETA, gamma=p["gamma"], wmin=p["wmin"], seed=seed, p=p["p"], Tmax=40000)
        st = div_stats(r["H"])
    return dict(axis=axis, val=val, seed=seed, res=r["res"], **st)


AXES = dict(
    wmin=[1e-5, 1e-4, 1e-3, 3e-3, 1e-2],
    gamma=[0.0, 0.1, 0.3, 1.0, 2.0],
    batch=[8, 32, 128, 512],
    N=[30, 60, 120, 240],
    p=[1.0, 0.7, 0.5, 0.3, 0.15],
    p_gamma0=[1.0, 0.7, 0.5, 0.3, 0.15],
)
KW = dict(p_gamma0=lambda v: dict(p=v, gamma=0.0))


def main():
    jobs = [(a, v, s, KW[a](v) if a in KW else {a: v}) for a, vals in AXES.items() for v in vals for s in SEEDS]
    jobs.sort(key=lambda j: -(j[3].get("N", 120) + (500 if j[0] == "batch" else 0)))
    with Timer(f"diversity ({len(jobs)})"):
        R = pmap(job, jobs)
    out = {}
    for a, vals in AXES.items():
        rows = []
        for v in vals:
            rr = [r for r in R if r["axis"] == a and r["val"] == v]
            keys = ("l1", "fdiv", "pr", "K", "block_pr", "block_fdiv", "res")
            rows.append(dict(val=v, **{k: [r[k] for r in rr] for k in keys}))
            print(a, v, {k: float(np.round(np.nanmean([x if x is not None else np.nan for x in rows[-1][k]]), 6)) for k in keys})
        out[a] = rows
    meta = dict(script="hebb/sweeps/s4_diversity.py",
                description="Within-cluster diversity on block12 at theta=0.3 (3 block clusters). l1: mean L1 distance "
                            "member-to-centroid of h; fdiv: 1 - mean pairwise voltage correlation within a cluster; pr: "
                            "participation ratio of the within-cluster voltage covariance (1 = identical copies; up to "
                            "the block size 4 = members span the block). Clusters with >= 3 members, average-linkage L1 cut 0.2. "
                            "block_pr / block_fdiv: same two measures over ALL neurons with >= 80% of their mass on one "
                            "planted block (the number of distinct functions available per mode).",
                theta=THETA, base=BASE, seeds=SEEDS, axes=AXES,
                stochastic="batch: minibatch covariance from `batch` Gaussian samples per step, dt=0.05, 16000 steps; stats "
                           "averaged over the last 20% (recorded every 400 steps). 'res' is not an equilibrium residual there.",
                mask="p < 1 is a LABELLED COMPARISON: fixed random potential-synapse mask (I-I symmetric and I-B), "
                     "every neuron keeps >= 1 boundary synapse; p_gamma0 = same masks with gamma = 0 (no lateral plasticity)",
                params=dict(lam=1, eta=1, dt=0.05, ii_scale=0.0, ib_scale=0.2, conv_tol=1e-5))
    save_json("diversity", dict(axes=out), meta)


if __name__ == "__main__":
    main()
