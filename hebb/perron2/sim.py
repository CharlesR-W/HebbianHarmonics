"""Dynamics helpers for C2/C3: equilibrate the v1 covariance rule (hebb/core.py, mean-field,
dense masks) at a given theta, and analyse an equilibrium against the exact part enumeration.

Conventions as hebb/big/cov_runs.py: w_min = OMEGA / N (OMEGA = 0.009), dt = 0.05, lambda = eta = 1,
cold start core.init_state(N, B, rng, p=1, ii_scale=0, ib_scale=0.2) with rng = default_rng(seed).
Integrate until the projected residual (core.stationarity) < STOP_TOL (checked every CHECK steps)
or max_steps; equilibrium claimed when residual < RES_TOL.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from hebb import core  # noqa: E402
from hebb.big import common as cm  # noqa: E402
from hebb.perron2 import parts as pp  # noqa: E402

OMEGA = 0.009
DT = 0.05
RES_TOL = 1e-6
STOP_TOL = 1e-7
CHECK = 50
MIN_FRAC = 0.01        # a cluster "populates" a part if it holds >= 1% of the neurons


def params(theta, N, gamma):
    return core.Params(theta=float(theta), wmin=OMEGA / N, dt=DT, gamma=float(gamma))


def cold_state(N, B, seed):
    return core.init_state(N, B, np.random.default_rng(seed), p=1.0, ii_scale=0.0, ib_scale=0.2)


def equilibrate(state, Sigma, P, max_steps):
    Wii, Wib, Mii, Mib = state
    n = 0
    r = core.stationarity(Wii, Wib, Mii, Mib, Sigma, P)["r"]
    while n < max_steps and r >= STOP_TOL:
        for _ in range(CHECK):
            Wii, Wib, _ = core.step(Wii, Wib, Mii, Mib, Sigma, P)
        n += CHECK
        r = core.stationarity(Wii, Wib, Mii, Mib, Sigma, P)["r"]
    return (Wii, Wib, Mii, Mib), n, float(r)


def clipped(state, P):
    Wii, Wib, Mii, Mib = state
    Wc = np.maximum(Wii, P.wmin * Mii)
    np.fill_diagonal(Wc, 0)
    return Wc, np.maximum(Wib, P.wmin * Mib)


def analyse(state, Sigma, theta, P, stable=None):
    """Clusters (core.clusters_auto on live neurons, as cov_runs), Perron report per cluster,
    populated stable parts, coverage, lateral weights between major clusters."""
    Wc, Wbc = clipped(state, P)
    N, B = Wbc.shape
    H = core.harmonic(Wc, Wbc)
    if stable is None:
        stable = [p for p in pp.enumerate_fast(Sigma, theta, only_stable=True)]
    thr = P.wmin * (1 + 1e-3)
    dead = (Wbc <= thr).all(1)
    labels = -np.ones(N, int)
    live = ~dead
    if live.sum() > 1:
        ca = core.clusters_auto(H[live])
        labels[live] = ca["labels"]
    elif live.sum() == 1:
        labels[live] = 0
    labels = cm.relabel_by_size(labels)
    rows = cm.perron_report(H, Wbc, P.wmin, labels, Sigma, theta, stable) if live.any() else []
    major = [k for k, r in enumerate(rows) if r["size"] >= max(2, MIN_FRAC * N)]
    st_masks = [p["mask"] for p in stable]
    populated = {}
    for k in major:
        r = rows[k]
        m = sum(1 << b for b in r["support"])
        if r.get("nearest_stable_part_cos", 0) > 0.98 and m in st_masks:
            populated[m] = populated.get(m, 0) + r["size"]
        r["matches_stable_part"] = bool(r.get("nearest_stable_part_cos", 0) > 0.98 and m in st_masks)
    # lateral weights: mean W_ii (above floor counted as is) between / within major clusters
    K = len(major)
    lat = np.zeros((K, K))
    for i, a in enumerate(major):
        ia = labels == a
        for j, b in enumerate(major):
            ib = labels == b
            blk = Wc[np.ix_(ia, ib)]
            if a == b:
                n = ia.sum()
                lat[i, j] = blk.sum() / max(1, n * (n - 1))
            else:
                lat[i, j] = blk.mean()
    offd = lat[~np.eye(K, dtype=bool)] if K > 1 else np.zeros(0)
    n_live_edges = int((np.triu(Wc, 1) > thr).sum())
    summary = dict(
        theta=float(theta), n_dead=int(dead.sum()), K=len(rows), K_major=K,
        sizes=[rows[k]["size"] for k in major],
        supports=[rows[k]["support"] for k in major],
        perron_cos=[rows[k].get("perron_cos") for k in major],
        nearest_cos=[rows[k].get("nearest_stable_part_cos") for k in major],
        matches=[rows[k]["matches_stable_part"] for k in major],
        top_eig_ok=[rows[k].get("top_eig_ok") for k in major],
        offsupport_violation=[rows[k].get("offsupport_violation") for k in major],
        c_top=[rows[k].get("c_top") for k in major],
        stable_parts=[p["S"] for p in stable], stable_c=[p["c"] for p in stable],
        n_stable=len(stable), populated=[pp.mask_list(m) for m in sorted(populated)],
        populated_sizes=[populated[m] for m in sorted(populated)],
        n_populated=len(populated), coverage=len(populated) / max(1, len(stable)),
        frac_on_stable=float(sum(populated.values()) / N),
        lateral_between_max=float(offd.max()) if len(offd) else 0.0,
        lateral_between=lat.round(6).tolist(),
        n_live_lateral=n_live_edges, n_lateral_components=cm.n_components(Wc, thr))
    return summary, labels, H, rows
