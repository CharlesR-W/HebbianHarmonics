"""Distance rule at N >> B: unit definitions and the unit runner (shared by the local linear runs
and the Modal rectifier runs, hebb/big/modal_app.py).

Matched protocol (R2): for a given (family, sigma, N, seed) every arm shares the attachments
(stream [seed,0], s = 3 random inputs, Dirichlet shares, total 1), the initial W = 0, the training
bank (S = 256, stream [seed,1]) and evaluation bank (4096, stream [seed,2]) and the b grid; the
arms differ only in the on-site potential:
    linear     klo = khi = 0
    rect       klo = 3, khi = 0, thresholds t_i ~ N(0, 0.7^2) (stream [seed,3])   (oa.RECT)
    rect_homog klo = 3, khi = 0, t_i = 0                                         (control)
Each unit is a cold start at fixed (b, alpha), integrated to stationarity (dist_engine).

Raw output per unit: scratch/out/big/dist/<name>.npz (+ .json): final W, responses on both banks,
and for film units the film arrays (coordinates computed here, from the same states as the
weights and edges).

Run locally (linear units):  uv run python -m hebb.big.dist_runs local [A|B|all]
"""
import json
import os
import resource
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from hebb import onsite_analysis as oa
from hebb.big import common as cm
from hebb.big import dist_engine as de

RAWD = cm.RAW / "dist"
ARMS = dict(linear=oa.LINEAR, rect=oa.RECT, rect_homog=dict(klo=3.0, khi=0.0, sigma_t=0.0))
TMAX = 40000

# b grids (cold-start sweeps) and the three regimes, read off the linear sweeps at seed 0
# (fragmented: most neurons singletons or tiny groups; percolation: largest component
# ~ 1/3-1/2 of N; fused: one component). Same b for every arm (same-b comparison).
GRID = dict(A=np.round(np.geomspace(0.0003, 0.3, 13), 7), B=np.round(np.geomspace(0.001, 1.0, 13), 6))
REGIMES = {("A", "ind3"): dict(fragmented=0.0003, percolation=0.0017, fused=0.01),
           ("B", "blocks"): dict(fragmented=0.003, percolation=0.018, fused=0.1),
           ("B", "power"): dict(fragmented=0.003, percolation=0.045, fused=0.18)}
SIGMA_DESC = dict(ind3="three independent unit-variance inputs",
                  blocks="oa.sigma_blocks(12): 4 blocks of 3, variances 2/1/0.5/0.25, within-block correlation 0.8",
                  power="oa.sigma_power(12): random eigenbasis (seed 12345), eigenvalues ~ 1/a, trace 12")


def sigma_of(name):
    return dict(ind3=np.eye(3), blocks=oa.sigma_blocks(12), power=oa.sigma_power(12))[name]


def unit_name(u):
    return f"{u['fam']}_{u['sigma']}_N{u['N']}_{u['arm']}_b{u['b']:g}_a{u['alpha']:g}_s{u['seed']}"


def units(which="all"):
    out = []

    def add(fam, sigma, N, arm, b, alpha=1.0, seed=0, film=0):
        out.append(dict(fam=fam, sigma=sigma, N=N, arm=arm, b=float(b), alpha=alpha, seed=seed, film=film))

    if which in ("A", "all"):
        reg = REGIMES[("A", "ind3")]
        for arm in ARMS:
            for b in GRID["A"]:
                add("A", "ind3", 300, arm, b)
            for name, b in reg.items():
                for seed in (0, 1, 2):
                    add("A", "ind3", 300, arm, b, seed=seed, film=60 if (seed == 0 and arm != "rect_homog") else 0)
            add("A", "ind3", 300, arm, reg["percolation"], alpha=0.1, film=60 if arm != "rect_homog" else 0)
    if which in ("B", "all"):
        for sg in ("blocks", "power"):
            reg = REGIMES[("B", sg)]
            for arm in ARMS:
                for b in GRID["B"]:
                    add("B", sg, 400, arm, b)
                for name, b in reg.items():
                    for seed in (0, 1):
                        hero = seed == 0 and name == "percolation" and arm != "rect_homog"
                        add("B", sg, 400, arm, b, seed=seed, film=40 if hero else 0)
    # de-duplicate (grid points coinciding with regime points keep the film flag)
    seen = {}
    for u in out:
        k = unit_name(u)
        if k not in seen or u["film"] > seen[k]["film"]:
            seen[k] = u
    return list(seen.values())


def run_unit(u, device=None, out_dir=RAWD):
    """Run one unit, write <name>.npz/.json to out_dir, return the json record."""
    t0 = time.time()
    Sigma = sigma_of(u["sigma"])
    B = Sigma.shape[0]
    N, seed, b, alpha = u["N"], u["seed"], u["b"], u["alpha"]
    G, Ge = oa.banks(Sigma, seed)
    net = oa.make_net(N, B, seed, **ARMS[u["arm"]])
    grid = cm.snapshot_grid(TMAX) if u["film"] else set()
    linear = u["arm"] == "linear"
    if linear:
        r = de.lin_run(net.A, G, b, alpha, Tmax=TMAX, snap_steps=grid)
        U = G @ r["H"].T
        Ue = Ge @ r["H"].T
        einfo = dict(gmax=0.0)
    else:
        r = de.rect_run(net.A, net.t, net.klo, net.khi, G, b, alpha, Tmax=TMAX, snap_steps=grid, device=device)
        U = r["U"]
        Ue, einfo = de.eval_responses(net.A, net.t, net.klo, net.khi, r["W"], Ge, device=device)
    arrays = dict(W=r["W"], A=net.A, t=net.t, U=U, Ue=Ue.astype(np.float32))
    if linear:
        arrays["H"] = r["H"]
    rec = dict(unit=u, name=unit_name(u), steps=r["steps"], t=r["t"], dt=r["dt"], residual=r["residual"],
               converged=r["converged"], run_seconds=r["seconds"], equil_gmax=r.get("equil_gmax", 0.0),
               eval_gmax=einfo["gmax"], newton_iters_mean=r.get("newton_iters_mean"),
               newton_iters_max=r.get("newton_iters_max"), device=r.get("device", "cpu"))
    if u["film"]:
        snaps = r["snaps"]
        steps = cm.pick_frames(snaps.keys(), r["steps"], u["film"])
        S = len(G)
        if linear:
            Hs = np.stack([snaps[s]["H"] for s in steps]).astype(float)
            arrays["film_H"] = Hs.astype(np.float32)
            feats = [G @ h.T for h in Hs]                      # (S, N) responses per frame
        else:
            feats = [snaps[s]["U"].astype(float) for s in steps]
        if linear and B == 3:
            coords = Hs
            rec["space"] = "simplex"
        else:
            Xf = feats[-1].T / np.sqrt(S)
            mean, basis, var = cm.pca_axes(Xf, 3)
            coords = np.stack([(F.T / np.sqrt(S) - mean) @ basis for F in feats])
            rec["space"] = "pca"
            rec["pca_explained_var"] = var[:3].tolist()
            rec["pca_total_var"] = float(var.sum())
        arrays["film_coords"] = coords.astype(np.float32)
        arrays["film_steps"] = np.array(steps)
        arrays["film_edges"] = np.concatenate([snaps[s]["edges"] for s in steps]).astype(np.int32) \
            if any(len(snaps[s]["edges"]) for s in steps) else np.zeros((0, 2), np.int32)
        arrays["film_w"] = np.concatenate([snaps[s]["w"] for s in steps])
        arrays["film_count"] = np.array([len(snaps[s]["edges"]) for s in steps])
        rec["frames"] = [dict(t=s * r["dt"], step=s, residual=snaps[s]["r"], n_live=snaps[s]["n_live"],
                              n_components=snaps[s]["n_comp"]) for s in steps]
    rec["total_seconds"] = time.time() - t0
    rec["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    try:
        import torch
        if torch.cuda.is_available():
            rec["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            rec["gpu_name"] = torch.cuda.get_device_name()
    except ImportError:
        pass
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, rec["name"] + ".npz"), **arrays)
    with open(os.path.join(out_dir, rec["name"] + ".json"), "w") as fh:
        json.dump(cm.clean(rec, 8), fh)
    print(f"{rec['name']}: steps {r['steps']} res {r['residual']:.1e} {rec['total_seconds']:.1f}s", flush=True)
    return rec


def _local(u):
    if (RAWD / (unit_name(u) + ".json")).exists():
        return None
    return run_unit(u)


def main(argv):
    if argv[0] == "local":
        which = argv[1] if len(argv) > 1 else "all"
        us = [u for u in units(which) if u["arm"] == "linear"]
        import multiprocessing as mp
        with mp.get_context("fork").Pool(int(os.environ.get("BIG_PROCS", "3"))) as p:
            p.map(_local, us, chunksize=1)
    elif argv[0] == "list":
        us = units(argv[1] if len(argv) > 1 else "all")
        print(len(us), "units;", sum(u["arm"] != "linear" for u in us), "nonlinear")


if __name__ == "__main__":
    main(sys.argv[1:])
