"""Q1: what is a cluster? Parts = stable fixed points of h ∝ (Sigma h - theta)_+  (= strict local
maxima of the nonnegative Rayleigh quotient of A = Sigma - theta 11^T), versus eigenvectors of
Sigma, and versus what the network actually populates.

For each test covariance and theta: enumerate all parts (exact, B <= 16), mark stable ones; run
the network (N=120) with gamma in {0, 1} and wmin in {1e-5, 1e-3}, 4 seeds; match every learned
cluster to the nearest stable part (cosine); record K, sizes, dead neurons, residual.

Output: site/data/sweeps/parts.json.   Runtime ~6 min on 3 procs.
"""
import numpy as np

from hebb.sweeps.common import (Timer, cluster_embed, cov_block, cov_hier, cov_powerlaw, cov_powerlaw_pos,
                                cov_ring, enumerate_parts, pmap, save_json, simulate)

N = 120
SEEDS = range(4)
CONFIGS = [  # (gamma, wmin)
    (1.0, 1e-5), (0.0, 1e-5), (1.0, 1e-3),
]
RES_TOL = 1e-4


def covs():
    S1, _ = cov_block([4, 4, 4], 0.5, 0.1)
    out = {
        "block12": dict(S=S1, desc="3 blocks of 4, within corr 0.5, across 0.1, unit variance"),
        "hier8": dict(S=cov_hier((0.6, 0.3, 0.05), (2, 2, 2)),
                      desc="ultrametric 8 inputs: siblings 0.6, cousins 0.3, halves 0.05"),
        "ring16": dict(S=cov_ring(16, 1.5), desc="ring of 16, Sigma=exp(-d^2/(2*1.5^2)); Fourier eigenvectors"),
        "powerlaw12": dict(S=cov_powerlaw(12, 1.0, seed=0),
                           desc="random rotation of spectrum k^-1, B=12 (signed correlations), trace=B"),
        "powerlaw_pos12": dict(S=cov_powerlaw_pos(12, 1.0, seed=0),
                               desc="nonnegative sparse factor model, factor variances k^-1, B=12, trace=B"),
        "signed3": dict(S=np.eye(3) - np.ones((3, 3)) / 3, desc="I - 11^T/3 (anticorrelated, rank 2)"),
    }
    return out


THETAS = [-0.1, 0.0, 0.03, 0.07, 0.13, 0.18, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]  # avoids exact correlation levels (marginal parts)


def theta_grid(S):
    return [round(t * float(np.mean(np.diag(S))), 4) for t in THETAS]


def eig_relations(S, parts, K):
    ev, V = np.linalg.eigh(S)
    V = V[:, ::-1]
    out = []
    for p in parts:
        h = p["h"] / np.linalg.norm(p["h"])
        cos = np.abs(V.T @ h)
        k = max(1, K)
        Sx = p["S"]
        if len(Sx) > 1:
            e2, V2 = np.linalg.eigh(S[np.ix_(Sx, Sx)])
            v2 = np.zeros(len(h)); v2[Sx] = np.abs(V2[:, -1]); v2 /= np.linalg.norm(v2)
            cos_restricted = float(abs(v2 @ h))
        else:
            cos_restricted = 1.0
        out.append(dict(max_cos_eig=float(cos.max()), argmax_eig=int(cos.argmax()),
                        cos_top=float(cos[0]), span_topK=float((cos[:k] ** 2).sum()),
                        cos_restricted_perron=cos_restricted))
    return out


def job(args):
    name, S, theta, gamma, wmin, seed, stable_h = args
    r = simulate(S, N, theta, gamma=gamma, wmin=wmin, seed=seed, Tmax=30000)
    H = r["H"]
    dead = r["Wib"].max(1) <= 1.5 * wmin
    lab, sizes, cents = cluster_embed(H, 0.1)
    Ks = {str(t): int(len(cluster_embed(H, t)[1])) for t in (0.05, 0.2)}
    alive_sizes, matched = [], []
    for k in range(len(sizes)):
        if dead[lab == k].all():
            continue
        alive_sizes.append(int(sizes[k]))
        if len(stable_h):
            c = cents[k] / np.linalg.norm(cents[k])
            cs = stable_h @ c
            matched.append(int(cs.argmax()) if cs.max() > 0.98 else -1)
        else:
            matched.append(-1)
    return dict(cov=name, theta=theta, gamma=gamma, wmin=wmin, seed=seed, K=len(alive_sizes),
                K_tol=Ks, sizes=alive_sizes, matched=matched, n_dead=int(dead.sum()),
                res=r["res"], steps=r["steps"], fp=bool(r["res"] < RES_TOL))


def enum_job(args):
    name, S, theta = args
    P = enumerate_parts(S, theta)
    return name, theta, P


def main():
    C = covs()
    with Timer("enumerate"):
        enum = pmap(enum_job, [(n, c["S"], th) for n, c in C.items() for th in theta_grid(c["S"])])
    parts = {(n, th): P for n, th, P in enum}
    jobs = []
    for n, c in C.items():
        for th in theta_grid(c["S"]):
            st = [p for p in parts[(n, th)] if p["stable"]]
            sh = np.array([p["h"] / np.linalg.norm(p["h"]) for p in st]) if st else np.zeros((0, len(c["S"])))
            for (g, wm) in CONFIGS:
                for s in SEEDS:
                    jobs.append((n, c["S"], th, g, wm, s, sh))
    with Timer(f"network runs ({len(jobs)})"):
        runs = pmap(job, jobs)
    out = {}
    for n, c in C.items():
        rows = []
        for th in theta_grid(c["S"]):
            P = parts[(n, th)]
            st = [p for p in P if p["stable"]]
            rel = eig_relations(c["S"], st, len(st))
            st_out = [dict(S=p["S"], h=np.round(p["h"], 4), c=p["c"], E=p["E"], margin=p["margin"], gap=p["gap"], **e)
                      for p, e in zip(st, rel)] if len(st) <= 40 else None
            R = [r for r in runs if r["cov"] == n and r["theta"] == th]
            per = {}
            for (g, wm) in CONFIGS:
                rr = [r for r in R if r["gamma"] == g and r["wmin"] == wm]
                matched_frac = [sum(sz for sz, m in zip(r["sizes"], r["matched"]) if m >= 0) / N for r in rr]
                per[f"g{g:g}_w{wm:g}"] = dict(
                    K=[r["K"] for r in rr], K_tol=[r["K_tol"] for r in rr], sizes=[r["sizes"] for r in rr],
                    matched=[r["matched"] for r in rr], matched_frac=matched_frac,
                    n_populated_parts=[len(set(m for m in r["matched"] if m >= 0)) for r in rr],
                    n_dead=[r["n_dead"] for r in rr], res=[r["res"] for r in rr], fp=[r["fp"] for r in rr])
            rows.append(dict(theta=th, n_fp=len(P), n_stable=len(st),
                             stable_support_sizes=sorted(len(p["S"]) for p in st),
                             stable_parts=st_out, runs=per,
                             summary=dict(mean_max_cos_eig=float(np.mean([e["max_cos_eig"] for e in rel])) if rel else None,
                                          mean_span_topK=float(np.mean([e["span_topK"] for e in rel])) if rel else None,
                                          mean_cos_restricted=float(np.mean([e["cos_restricted_perron"] for e in rel])) if rel else None)))
        ev, V = np.linalg.eigh(c["S"])
        out[n] = dict(desc=c["desc"], Sigma=np.round(c["S"], 4), eigvals=ev[::-1], eigvecs=np.round(V[:, ::-1].T, 4),
                      rows=rows)
    meta = dict(
        script="hebb/sweeps/s1_parts.py",
        description="Stable parts (exact enumeration of fixed points of h ∝ (Sigma h - theta)_+ with the "
                    "stability test 'h is the top eigenvector of (Sigma - theta 11^T)_SS and strict off-support "
                    "inequality') vs. clusters populated by the network.",
        N=N, seeds=list(SEEDS), configs=[dict(gamma=g, wmin=w) for g, w in CONFIGS],
        params=dict(lam=1, eta=1, dt=0.05, ii_scale=0.0, ib_scale=0.2, p=1.0, Tmax=30000, conv_tol=1e-5),
        cluster_def="average-linkage on L1 distance between embeddings, cut at 0.1 (K_tol gives cuts 0.05, 0.2); "
                    "clusters made only of dead neurons (all I-B weights at floor) excluded",
        match_def="learned centroid matched to a stable part if cosine > 0.98",
        res_tol=RES_TOL, theta_grid_note="theta = t * mean(diag Sigma), t in " + str(THETAS),
        fields=dict(n_stable="number of stable parts (function of Sigma and theta only)",
                    K="populated clusters per seed", matched_frac="fraction of neurons in clusters matching a stable part",
                    max_cos_eig="max |cos| between part and any single eigenvector of Sigma",
                    span_topK="fraction of the part's norm inside the top-K eigenspace of Sigma, K = n_stable",
                    cos_restricted_perron="cos between part and Perron vector of Sigma restricted to the part's support"))
    p = save_json("parts", dict(covs=out), meta)
    print("wrote", p)
    # brief console summary
    for n in out:
        print(n)
        for row in out[n]["rows"]:
            s = row["runs"]
            print(f"  th={row['theta']:+.3f} stable={row['n_stable']:3d} sizes={sorted(set(row['stable_support_sizes']))} "
                  + " ".join(f"{k}:K={s[k]['K']} m={np.mean(s[k]['matched_frac']):.2f} dead={s[k]['n_dead']} fp={sum(s[k]['fp'])}" for k in s)
                  + f" maxcos={row['summary']['mean_max_cos_eig']}")


if __name__ == "__main__":
    main()
