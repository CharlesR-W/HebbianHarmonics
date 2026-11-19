"""C3: cold starts. hier16 and block12, N = 300, theta on a 30-point subset of the ramp grid
(so C4 can compare with the ramps at identical theta), seeds 0-4, gamma in {1, 0.5}.
Each unit: core.init_state cold start (as hebb/big/cov_runs), integrate to projected residual
< 1e-7 (cap 120000 steps), sim.analyse. Units run on Modal (hebb/perron2/modal_app.py) or locally.

    uv run python -m hebb.perron2.cold units      # writes scratch/out/perron2/cold_cfg.json
    uv run python -m hebb.perron2.cold local      # run all units locally (4 procs)
    uv run python -m hebb.perron2.cold extra      # block345 (unequal blocks), 8 seeds, local
    uv run python -m hebb.perron2.cold merge <dir-with-unit-jsons>
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from hebb.big import common as cm  # noqa: E402
from hebb.perron2 import parts as pp  # noqa: E402
from hebb.perron2 import sim  # noqa: E402

N = 300
MAX_STEPS = 120000
RAMP_GRID = np.round(np.linspace(0.03, 0.95, 80), 6)
GRID = RAMP_GRID[np.round(np.linspace(0, 79, 30)).astype(int)]
SEEDS = range(5)
GAMMAS = (1.0, 0.5)
KICK_TEST = True


def sigmas():
    return dict(hier16=cm.cov_hier((0.8, 0.5, 0.25, 0.05), (2, 2, 2, 2)),
                block12=cm.cov_block([4, 4, 4], 0.5, 0.1),
                block345=cm.cov_block([3, 4, 5], 0.5, 0.1))   # extra: unequal blocks -> unequal c


def unit_name(u):
    return f"{u['sigma']}_th{u['theta']:.6f}_g{u['gamma']}_s{u['seed']}"


def units():
    return [dict(sigma=s, theta=float(th), gamma=g, seed=sd)
            for s in ("hier16", "block12") for th in GRID for g in GAMMAS for sd in SEEDS]


def run_unit(u):
    Sigma = sigmas()[u["sigma"]]
    B = len(Sigma)
    t0 = time.time()
    P = sim.params(u["theta"], N, u["gamma"])
    st, n, r = sim.equilibrate(sim.cold_state(N, B, u["seed"]), Sigma, P, MAX_STEPS)
    stable = pp.enumerate_fast(Sigma, u["theta"], only_stable=True)
    s, lab, H, rows = sim.analyse(st, Sigma, u["theta"], P, stable)
    s.update(u, steps=n, residual=r, converged=bool(r < sim.RES_TOL), name=unit_name(u))
    if KICK_TEST:
        # saddle test: kick all above-floor weights by (1 + 1e-3 xi) and re-equilibrate, up to 3 times,
        # until the cluster supports and sizes stop changing
        from hebb.perron2.ramps import kick
        rng = np.random.default_rng(777 + u["seed"])
        pre = dict(sizes=s["sizes"], supports=s["supports"], populated=s["populated"], n_populated=s["n_populated"])
        nk = 0
        for k in range(3):
            st, n2, r = sim.equilibrate(kick(st, P, rng), Sigma, P, MAX_STEPS)
            n += n2
            s2, lab, H, rows = sim.analyse(st, Sigma, u["theta"], P, stable)
            nk += 1
            same = s2["supports"] == s["supports"] and s2["sizes"] == s["sizes"]
            s = s2
            if same:
                break
        s.update(u, steps=n, residual=r, converged=bool(r < sim.RES_TOL), name=unit_name(u),
                 pre_kick=pre, kicks=nk, changed_by_kick=pre["supports"] != s["supports"] or pre["sizes"] != s["sizes"])
    s["seconds"] = time.time() - t0
    return s


def _local(u):
    out = pp.RAW / "cold_units" / f"{unit_name(u)}.json"
    if out.exists():
        return
    s = run_unit(u)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cm.clean(s, 6)))
    print(s["name"], s["steps"], f"{s['residual']:.1e}", s["sizes"], s["populated"], f"{s['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    w = sys.argv[1]
    if w == "units":
        cfg = dict(units=[dict(name=unit_name(u), item=u) for u in units()])
        pp.RAW.mkdir(parents=True, exist_ok=True)
        (pp.RAW / "cold_cfg.json").write_text(json.dumps(cfg))
        print(len(cfg["units"]), "units")
    elif w == "local":
        import multiprocessing as mp
        with mp.get_context("fork").Pool(int(os.environ.get("P2_PROCS", "4"))) as p:
            p.map(_local, units(), chunksize=1)
    elif w == "extra":
        # which part does a bound population pick when the stable parts have unequal c?
        import multiprocessing as mp
        ex = [dict(sigma="block345", theta=th, gamma=g, seed=sd) for th, g in ((0.15, 1.0), (0.2, 1.0), (0.25, 1.0), (0.1, 0.5))
              for sd in range(8)]
        with mp.get_context("fork").Pool(int(os.environ.get("P2_PROCS", "2"))) as p:
            p.map(_local, ex, chunksize=1)
