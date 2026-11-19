"""Covariance rule (v1 engine hebb/core.py, mean-field, dense masks) at N >> B.

Family A: B = 3, N = 300 films: independent unit inputs at theta = 0.2 and 0.4; correlated
pair rho12 = 0.6 at theta = 0.5. Family B: B = 12-16, N = 400: blocks (3 x 4), hierarchy (16),
ring (16), three theta each; per cluster the Perron report (hebb/big/common.perron_report).
Floor scaled with N (R3): w_min = omega / N, omega = 0.009 (the live demo's 90 x 1e-4);
sensitivity at omega / 3 (data only). lambda = eta = gamma = 1, dt = 0.05,
W_IB ~ Exp(0.2), W_II at the floor, seed s -> np.random.default_rng(s) (core.init_state).
Equilibrium claimed when the projected residual (core.stationarity) < RES_TOL.

Run:  uv run python -m hebb.big.cov_runs [pilot|A|B|all]
"""
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from hebb import core
from hebb.big import common as cm

SCRIPT = "hebb/big/cov_runs.py"
OMEGA = 0.009
RES_TOL = 1e-6        # equilibrium claim
STOP_TOL = 1e-7       # stop integrating
DT = 0.05
TMAX = 200000
CHECK = 100


def presets():
    S3_ind = np.eye(3)
    S3_pair = core.cov3((1, 1, 1), (0.6, 0, 0))
    blocks = cm.cov_block([4, 4, 4], 0.5, 0.1)
    hier = cm.cov_hier((0.8, 0.5, 0.25, 0.05), (2, 2, 2, 2))
    ring = cm.cov_ring(16, 1.5)
    return dict(
        ind3=dict(Sigma=S3_ind, desc="three independent unit-variance inputs"),
        pair3=dict(Sigma=S3_pair, desc="unit variances, rho12 = 0.6, g3 independent"),
        blocks12=dict(Sigma=blocks, desc="12 inputs, 3 blocks of 4, within 0.5, across 0.1, unit variances (v1 block12)"),
        hier16=dict(Sigma=hier, desc="16 inputs, binary hierarchy of depth 4, correlation 0.8 / 0.5 / 0.25 / 0.05 "
                                     "for lowest common ancestor at depth 0 (siblings) / 1 / 2 / 3"),
        ring16=dict(Sigma=ring, desc="ring of 16, Sigma_bc = exp(-d(b,c)^2 / (2 * 1.5^2)) (v1 ring16)"),
    )


def run_cov(Sigma, N, theta, omega=OMEGA, seed=0, gamma=1.0, film=True, Tmax=TMAX):
    B = Sigma.shape[0]
    wmin = omega / N
    P = core.Params(theta=theta, wmin=wmin, dt=DT, gamma=gamma)
    rng = np.random.default_rng(seed)
    Wii, Wib, Mii, Mib = core.init_state(N, B, rng, p=1.0, ii_scale=0.0, ib_scale=0.2)
    grid = cm.snapshot_grid(Tmax)
    snaps = {}
    t0 = time.time()

    def snap(n):
        Wc = np.maximum(Wii, wmin * Mii); np.fill_diagonal(Wc, 0)
        Wbc = np.maximum(Wib, wmin * Mib)
        H = core.harmonic(Wc, Wbc)
        r = core.stationarity(Wii, Wib, Mii, Mib, Sigma, P)["r"]
        e, w = cm.top_edges(Wc, wmin * (1 + 1e-3))
        snaps[n] = dict(H=H.astype(np.float32), Wib=Wbc.astype(np.float32), edges=e, w=w, r=r,
                        n_live=int((np.triu(Wc, 1) > wmin * (1 + 1e-3)).sum()),
                        n_comp=cm.n_components(Wc, wmin * (1 + 1e-3)))

    n = 0
    r = np.inf
    while n < Tmax:
        if film and n in grid:
            snap(n)
        if n % CHECK == 0:
            r = core.stationarity(Wii, Wib, Mii, Mib, Sigma, P)["r"]
            if r < STOP_TOL:
                break
        Wii, Wib, _ = core.step(Wii, Wib, Mii, Mib, Sigma, P)
        n += 1
    r = core.stationarity(Wii, Wib, Mii, Mib, Sigma, P)["r"]
    snap(n)
    Wii = np.maximum(Wii, wmin * Mii); np.fill_diagonal(Wii, 0)
    Wib = np.maximum(Wib, wmin * Mib)
    return dict(Wii=Wii, Wib=Wib, H=core.harmonic(Wii, Wib), steps=n, t=n * DT, residual=float(r),
                converged=bool(r < RES_TOL), seconds=time.time() - t0, snaps=snaps, wmin=wmin, P=P)


def analyse(res, Sigma, theta, parts):
    H, Wib, wmin = res["H"], res["Wib"], res["wmin"]
    N = len(H)
    dead = (Wib <= wmin * (1 + 1e-3)).all(1)
    labels = -np.ones(N, int)
    ca = core.clusters_auto(H[~dead]) if (~dead).sum() > 1 else dict(labels=np.zeros((~dead).sum(), int), robust=True, tol=0)
    labels[~dead] = ca["labels"]
    labels = cm.relabel_by_size(labels)
    rows = cm.perron_report(H, Wib, wmin, labels, Sigma, theta, parts)
    stable = [p for p in parts if p["stable"]]
    big = [r for r in rows if r["size"] >= max(2, 0.02 * N)]
    populated = {tuple(r.get("nearest_stable_part_support", [])) for r in big
                 if r.get("nearest_stable_part_cos", 0) > 0.98}
    lam, V = np.linalg.eigh(Sigma)
    metrics = dict(
        K=len(rows), K_major=len(big), n_dead=int(dead.sum()), sizes=[r["size"] for r in rows],
        clusters_robust=bool(ca.get("robust", True)), cluster_cut_L1=ca.get("tol"),
        n_parts=len(parts), n_stable_parts=len(stable), n_stable_populated=len(populated),
        frac_neurons_on_stable_part=float(sum(r["size"] for r in rows if r.get("nearest_stable_part_cos", 0) > 0.98) / N),
        max_spread=max([r["spread"] for r in rows] + [0.0]),
        kappa0=float(Sigma.mean()), residual=res["residual"], converged=res["converged"], steps=res["steps"],
        t=res["t"], seconds=res["seconds"], wmin=wmin, omega=wmin * N,
        Sigma_eigenvalues=lam[::-1],
        stable_parts=[dict(S=p["S"], h=p["h"], c=p["c"], margin=p["margin"],
                           best_eig_cos=float(np.max(np.abs(V.T @ p["h"])) / np.linalg.norm(p["h"])))
                      for p in stable][:64])
    return labels, rows, metrics


def export_film(fid, title, family, res, Sigma, theta, labels, rows, metrics, cfg, F):
    snaps = res["snaps"]
    steps = cm.pick_frames(snaps.keys(), res["steps"], F)
    B = Sigma.shape[0]
    N = len(res["H"])
    Hs = np.stack([snaps[s]["H"] for s in steps]).astype(float)
    if B == 3:
        coords, space, label = Hs, "simplex", "input space (harmonic embedding)"
        pca = None
    else:
        R = cm.psd_sqrt(Sigma)
        mean, basis, var = cm.pca_axes(Hs[-1] @ R, 3)
        coords = np.stack([(h @ R - mean) @ basis for h in Hs])
        space, label = "pca", "response-function space (PCA)"
        pca = dict(explained_var=var[:3], total_var=var.sum(),
                   features="x_i = Sigma^{1/2} h_i (Euclidean distance = RMS response distance); axes from the final frame")
    frames = [dict(t=s * DT, step=s, residual=snaps[s]["r"], n_live=snaps[s]["n_live"],
                   n_components=snaps[s]["n_comp"]) for s in steps]
    meta = dict(id=fid, title=title, family=family, rule="cov", nonlin="linear", B=B, N=N, seed=cfg["seed"],
                config=cfg, script=SCRIPT, space=space, space_label=label, pca=pca, frames=frames,
                edge_threshold=f"lateral w > w_min (1 + 1e-3) = {res['wmin'] * 1.001:.4g}",
                final=dict(labels=labels.tolist(), clusters=rows, metrics=metrics),
                network_layout_seed=1000 + cfg["seed"])
    return cm.write_film(meta, coords, H=Hs, edges=[snaps[s]["edges"] for s in steps],
                         edge_w=[snaps[s]["w"] for s in steps], in_w=np.stack([snaps[s]["Wib"] for s in steps]),
                         index_entry=dict(title=title, family=family, rule="cov", nonlin="linear", B=B, N=N))


def one(name, theta, N, omega=OMEGA, seed=0, film_id=None, title=None, family="A", F=80):
    P = presets()[name]
    Sigma = P["Sigma"]
    parts = cm.enumerate_parts(Sigma, theta)
    res = run_cov(Sigma, N, theta, omega, seed, film=film_id is not None)
    labels, rows, metrics = analyse(res, Sigma, theta, parts)
    cfg = dict(preset=name, Sigma_desc=P["desc"], Sigma=Sigma, theta=theta, N=N, omega=omega, wmin=res["wmin"],
               gamma=1.0, lam=1.0, eta=1.0, dt=DT, init="W_IB ~ Exp(0.2), W_II at floor, dense masks",
               seed=seed, mean_field=True, res_tol=RES_TOL, stop_tol=STOP_TOL)
    out = dict(name=name, theta=theta, N=N, omega=omega, seed=seed, config=cfg, clusters=rows, metrics=metrics)
    if film_id:
        out["film"] = film_id
        out["film_bytes"] = export_film(film_id, title, family, res, Sigma, theta, labels, rows, metrics, cfg, F)
    cm.RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cm.RAW / f"cov_{name}_th{theta}_N{N}_om{omega:.4g}_s{seed}.npz", Wii=res["Wii"],
                        Wib=res["Wib"], H=res["H"], labels=labels)
    print(f"cov {name} theta={theta} N={N} omega={omega:.4g} seed={seed}: K={metrics['K']} major={metrics['K_major']} "
          f"dead={metrics['n_dead']} r={res['residual']:.2e} steps={res['steps']} {res['seconds']:.0f}s "
          f"perron_cos={[round(r.get('perron_cos', float('nan')), 3) for r in rows[:6]]}", flush=True)
    return out


JOBS_A = [  # (name, theta, film id, title)
    ("ind3", 0.4, "A_cov_ind_th0.4", "Covariance rule, 3 independent inputs, theta = 0.4 (N = 300)"),
    ("ind3", 0.2, "A_cov_ind_th0.2", "Covariance rule, 3 independent inputs, theta = 0.2 (N = 300)"),
    ("pair3", 0.5, "A_cov_pair_th0.5", "Covariance rule, correlated pair rho12 = 0.6, theta = 0.5 (N = 300)"),
]
THETAS_B = dict(blocks12=(0.2, 0.35, 0.6), hier16=(0.2, 0.35, 0.6), ring16=(0.3, 0.5, 0.7))
HERO_B = {("blocks12", 0.35), ("ring16", 0.5), ("hier16", 0.35)}


def job(args):
    kind, name, theta, omega, seed = args[:5]
    if kind == "A":
        fid, title = args[5], args[6]
        return one(name, theta, 300, omega, seed, film_id=fid if omega == OMEGA and seed == 0 else None,
                   title=title, family="A", F=80)
    fid = f"B_cov_{name}_th{theta}" if (name, theta) in HERO_B and omega == OMEGA and seed == 0 else None
    title = f"Covariance rule, {name}, theta = {theta} (N = 400)"
    return one(name, theta, 400, omega, seed, film_id=fid, title=title, family="B", F=40)


def main(which):
    import multiprocessing as mp
    jobs = []
    if which == "pilot":
        t0 = time.time()
        out = job(("A", "ind3", 0.4, OMEGA, 0, *JOBS_A[0][2:]))
        print("pilot seconds", time.time() - t0, "bytes", out["film_bytes"])
        return
    if which in ("A", "all"):
        for name, th, fid, title in JOBS_A:
            for seed in (0, 1, 2):
                jobs.append(("A", name, th, OMEGA, seed, fid, title))
            jobs.append(("A", name, th, OMEGA / 3, 0, fid, title))
    if which in ("B", "all"):
        for name, ths in THETAS_B.items():
            for th in ths:
                jobs.append(("B", name, th, OMEGA, 0))
                jobs.append(("B", name, th, OMEGA / 3, 0))
    procs = int(os.environ.get("BIG_PROCS", "3"))
    with mp.get_context("fork").Pool(procs) as p:
        outs = p.map(job, jobs, chunksize=1)
    path = cm.OUT / f"cov_{which}.json"
    cm.write_json(path, dict(
        meta=dict(script=SCRIPT, description="Covariance rule at N >> B: per run the cluster table with the Perron "
                  "report and run metrics. See site/data/big/README.md.", omega=OMEGA, res_tol=RES_TOL),
        runs=outs))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
