"""C2: quasi-static theta ramps on hier16 (N = 300), gamma in {1, 0.5, 0.25}.

Protocol: theta grid = 80 values linspace(0.03, 0.95). Up: cold start (seed 0) at theta = 0.03,
then at each grid theta integrate from the previous equilibrium until the projected residual
< 1e-7 (checked every 50 steps; cap 40000 steps = t 2000; 120000 for the cold start); equilibrium
claimed when < 1e-6. Down: continue from the final up state (theta = 0.95) back down the same grid
(a hysteresis loop). At every new theta all above-floor weights get a relative kick
(1 + 1e-3 xi) (see `kick`) so that symmetric clusters cannot rest on saddles. Per theta: sim.analyse (clusters, populated stable parts, coverage, lateral
weights). Raw states per theta go to scratch/out/perron2/ramp_g<gamma>_<dir>.npz.

Second down ramp ("down2", `run_down2`): cold start at theta = GRID[68] = 0.822 (where cold starts
populate all 16 singleton parts), then down to 0.03.

Run:  uv run python -m hebb.perron2.ramps [run|down2|export g:dir ...]
"""
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from hebb.big import common as cm  # noqa: E402
from hebb.perron2 import films, sim  # noqa: E402
from hebb.perron2 import parts as pp  # noqa: E402

SCRIPT = "hebb/perron2/ramps.py"
N = 300
GAMMAS = [float(g) for g in os.environ.get("P2_GAMMAS", "1,0.5,0.25").split(",")]
GRID = np.round(np.linspace(0.03, 0.95, 80), 6)
MAX_STEPS = 40000
MAX_STEPS_COLD = 120000
KICK = float(os.environ.get("P2_KICK", "1e-3"))   # relative noise on every above-floor weight at each new theta
FILM_EDGES = 300         # stored in the raw npz (strongest first)
SITE_EDGES = 150         # written to the site films
SIGMA = cm.cov_hier((0.8, 0.5, 0.25, 0.05), (2, 2, 2, 2))
_STABLE = {}


def stable_at(th):
    if th not in _STABLE:
        _STABLE[th] = pp.enumerate_fast(SIGMA, th, only_stable=True)
    return _STABLE[th]


def kick(state, P, rng):
    """Multiply every weight above the floor by (1 + KICK xi), xi ~ N(0, 1) i.i.d. (symmetric
    for W_II). Without it the mean-field flow keeps exactly symmetric clone clusters on
    unstable (saddle) parts, where the residual can fall below tolerance."""
    Wii, Wib, Mii, Mib = state
    if KICK <= 0:
        return state
    X = np.triu(rng.standard_normal(Wii.shape), 1)
    X = X + X.T
    Wii = np.where(Wii > P.wmin, Wii * (1 + KICK * X), Wii)
    Wib = np.where(Wib > P.wmin, Wib * (1 + KICK * rng.standard_normal(Wib.shape)), Wib)
    return Wii, Wib, Mii, Mib


def ramp(state, thetas, gamma, tag, first_cap, rng):
    recs, snaps = [], dict(H=[], labels=[], Wib=[], Wii=[], edges=[], edge_w=[])
    tot = 0
    for i, th in enumerate(thetas):
        P = sim.params(th, N, gamma)
        t0 = time.time()
        if i > 0 or first_cap == MAX_STEPS:
            state = kick(state, P, rng)
        state, n, r = sim.equilibrate(state, SIGMA, P, first_cap if i == 0 else MAX_STEPS)
        tot += n
        s, lab, H, rows = sim.analyse(state, SIGMA, th, P, stable_at(float(th)))
        Wc, Wbc = sim.clipped(state, P)
        e, w = cm.top_edges(Wc, P.wmin * (1 + 1e-3), FILM_EDGES)
        s.update(steps=n, steps_cum=tot, residual=r, converged=bool(r < sim.RES_TOL), seconds=time.time() - t0)
        recs.append(s)
        snaps["H"].append(H.astype(np.float32))
        snaps["labels"].append(lab.astype(np.int16))
        snaps["Wib"].append(Wbc.astype(np.float32))
        snaps["Wii"].append(Wc.astype(np.float32))
        snaps["edges"].append(e)
        snaps["edge_w"].append(w)
        print(f"[{tag}] th={th:.4f} steps={n} r={r:.1e} K={s['K_major']} sizes={s['sizes'][:8]} "
              f"pop={s['n_populated']}/{s['n_stable']} dead={s['n_dead']} lat={s['lateral_between_max']:.3g} "
              f"{time.time() - t0:.0f}s", flush=True)
    return state, recs, snaps


def save_raw(tag, thetas, snaps):
    E = max(1, max(len(e) for e in snaps["edges"]))
    ea = np.zeros((len(thetas), E, 2), np.int32)
    wa = np.zeros((len(thetas), E))
    cnt = []
    for f, (e, w) in enumerate(zip(snaps["edges"], snaps["edge_w"])):
        ea[f, :len(e)] = e
        wa[f, :len(e)] = w
        cnt.append(len(e))
    np.savez_compressed(pp.RAW / f"ramp_{tag}.npz", theta=np.asarray(thetas), H=np.stack(snaps["H"]),
                        labels=np.stack(snaps["labels"]), Wib=np.stack(snaps["Wib"]), Wii=np.stack(snaps["Wii"]),
                        edges=ea, edge_w=wa, edge_count=np.array(cnt))


def run_gamma(gamma):
    st = sim.cold_state(N, 16, 0)
    rng = np.random.default_rng(12345)
    st, up, snaps = ramp(st, GRID, gamma, f"g{gamma}_up", MAX_STEPS_COLD, rng)
    save_raw(f"g{gamma}_up", GRID, snaps)
    st, down, snaps = ramp(st, GRID[::-1], gamma, f"g{gamma}_down", MAX_STEPS, rng)
    save_raw(f"g{gamma}_down", GRID[::-1], snaps)
    out = dict(gamma=gamma, up=up, down=down)
    cm.write_json(pp.RAW / f"ramp_g{gamma}.json", out, sig=6)
    return out


DOWN2_START = 68          # GRID[68] = 0.8219: cold starts populate all 16 singleton parts there


def run_down2(gamma):
    """Second down ramp: cold start (seed 0) at GRID[DOWN2_START], then down to 0.03; tests merging
    of many populated children (the first down ramp starts from one cluster)."""
    st = sim.cold_state(N, 16, 0)
    rng = np.random.default_rng(54321)
    thetas = GRID[:DOWN2_START + 1][::-1]
    st, rec, snaps = ramp(st, thetas, gamma, f"g{gamma}_down2", MAX_STEPS_COLD, rng)
    save_raw(f"g{gamma}_down2", thetas, snaps)
    cm.write_json(pp.RAW / f"ramp_g{gamma}_down2.json", dict(gamma=gamma, down2=rec), sig=6)


def run():
    import multiprocessing as mp
    pp.RAW.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(len(GAMMAS)) as p:
        p.map(run_gamma, GAMMAS, chunksize=1)


# ----------------------------------------------------------------------------- export
def export_film(gamma, direction, fid):
    z = np.load(pp.RAW / f"ramp_g{gamma}_{direction}.npz")
    import json
    src = f"ramp_g{gamma}_down2.json" if direction == "down2" else f"ramp_g{gamma}.json"
    rec = json.loads((pp.RAW / src).read_text())[direction]
    H = z["H"].astype(float)
    F = len(H)
    R = cm.psd_sqrt(SIGMA)
    X = H @ R                                         # (F, N, B) features, distance = RMS response distance
    flat = X.reshape(-1, X.shape[-1])
    mean, basis, var = cm.pca_axes(flat, 3)
    coords = np.stack([(x - mean) @ basis for x in X])
    cnt = np.minimum(z["edge_count"], SITE_EDGES)
    edges = [z["edges"][f, :cnt[f]] for f in range(F)]
    ew = [z["edge_w"][f, :cnt[f]] for f in range(F)]
    thr = sim.OMEGA / N * 1.001
    frames = [dict(theta=r["theta"], t=r["steps_cum"] * sim.DT, step=r["steps_cum"], steps_this_theta=r["steps"],
                   residual=r["residual"], converged=r["converged"], n_live=r["n_live_lateral"],
                   n_components=r["n_lateral_components"], K_major=r["K_major"], sizes=r["sizes"],
                   supports=r["supports"], populated=r["populated"], n_stable=r["n_stable"],
                   coverage=r["coverage"], lateral_between_max=r["lateral_between_max"]) for r in rec]
    last = rec[-1]
    th_last = float(z["theta"][-1])
    P = sim.params(th_last, N, gamma)
    labels = z["labels"][-1].astype(int)
    rows = cm.perron_report(H[-1], z["Wib"][-1].astype(float), P.wmin, labels, SIGMA, th_last, stable_at(th_last))
    title = {"down": f"Theta ramp down ({GRID[-1]} to {GRID[0]}, continuing the up ramp)",
             "down2": f"Theta ramp down from a cold start at {GRID[DOWN2_START]} to {GRID[0]}",
             "up": f"Theta ramp up from a cold start at {GRID[0]} to {GRID[-1]}"}[direction]
    title = f"{title}, hier16, gamma = {gamma} (N = {N})"
    cfg = dict(preset="hier16", Sigma=SIGMA, N=N, gamma=gamma, omega=sim.OMEGA, wmin=sim.OMEGA / N, dt=sim.DT,
               lam=1.0, eta=1.0, theta_grid=GRID, direction=direction, seed=0,
               kick=KICK,
               start={"up": "cold start at theta = 0.03", "down": "end of the up ramp (theta = 0.95)",
                      "down2": f"cold start at theta = {GRID[DOWN2_START]}"}[direction],
               protocol="one frame per grid theta; relative kick 1e-3 on all above-floor weights; integrate from the previous equilibrium until projected residual "
                        "< 1e-7 (cap 40000 steps); up starts cold at theta = 0.03; down continues from the end of up",
               init="W_IB ~ Exp(0.2), W_II at floor, dense masks", mean_field=True)
    meta = dict(id=fid, title=title, family="ramp", rule="cov", nonlin="linear", B=16, N=N, seed=0, config=cfg,
                script=SCRIPT, space="pca", space_label="response-function space (PCA)",
                pca=dict(explained_var=var[:3], total_var=var.sum(),
                         features="x_i = Sigma^{1/2} h_i; axes from all frames pooled (not the final frame)"),
                frames=frames, edge_threshold=f"lateral w > w_min (1 + 1e-3) = {thr:.4g}",
                final=dict(labels=labels.tolist(), clusters=rows, metrics=dict(theta=th_last, **{k: last[k] for k in (
                    "K_major", "n_stable", "n_populated", "coverage", "residual", "converged")})),
                network_layout_seed=1000)
    return films.write_film(meta, coords, H, edges, ew, z["labels"],
                            dict(title=title, family="ramp", rule="cov", nonlin="linear", B=16, N=N,
                                 gamma=gamma, direction=direction))


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "run"
    if w == "run":
        run()
    elif w == "down2":
        import multiprocessing as mp
        with mp.get_context("fork").Pool(len(GAMMAS)) as p:
            p.map(run_down2, GAMMAS, chunksize=1)
    elif w == "export":
        for spec in sys.argv[2:]:                    # e.g. 1:up 0.25:down
            g, d = spec.split(":")
            g = float(g)
            print(export_film(g, d, f"ramp_hier16_g{g}_{d}"))
