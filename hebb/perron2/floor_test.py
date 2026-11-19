"""C2 follow-up: does a population whose part becomes unstable split between the children, or
fall as a body into one, and what decides it (floor w_min = omega / N, kick size, lateral gain)?

Take a ramp equilibrium just below a level (scratch/out/perron2/ramp_g<gamma>_up.npz), jump theta
to just above it, kick, and integrate to equilibrium with variants of omega and kick size.
Writes scratch/out/perron2/floor_test.json.

    uv run python -m hebb.perron2.floor_test
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from hebb.big import common as cm  # noqa: E402
from hebb.perron2 import parts as pp  # noqa: E402
from hebb.perron2 import sim  # noqa: E402
from hebb.perron2.ramps import GRID, N, SIGMA, stable_at  # noqa: E402

CASES = []
for gamma, i_from, i_to in ((0.5, 18, 20), (0.5, 40, 42), (0.1, 1, 2), (0.1, 18, 20), (1.0, 18, 20)):
    for omega in (0.009, 0.0009, 0.00009):
        for kick in (1e-3, 1e-2):
            CASES.append(dict(gamma=gamma, i_from=i_from, i_to=i_to, omega=omega, kick=kick))

# set 2: the ramp's own small step (one grid step: 0.0013 / 0.0075 / 0.0033 past the level)
CASES2 = [dict(gamma=g, i_from=a, i_to=a + 1, omega=om, kick=1e-3)
          for g, a in ((0.5, 18), (0.5, 40), (0.1, 1), (0.1, 18), (0.1, 40)) for om in (0.009, 0.0009, 0.00009)]


def run(c):
    z = np.load(pp.RAW / f"ramp_g{c['gamma']}_up.npz")
    th = float(GRID[c["i_to"]])
    Wii = z["Wii"][c["i_from"]].astype(float)
    Wib = z["Wib"][c["i_from"]].astype(float)
    old_floor = sim.OMEGA / N
    wmin = c["omega"] / N
    Wii = np.where(Wii <= old_floor * 1.001, wmin, Wii)
    np.fill_diagonal(Wii, 0)
    Wib = np.where(Wib <= old_floor * 1.001, wmin, Wib)
    Mii = 1 - np.eye(N)
    Mib = np.ones_like(Wib)
    P = sim.params(th, N, c["gamma"])
    P.wmin = wmin
    rng = np.random.default_rng(5)
    X = np.triu(rng.standard_normal(Wii.shape), 1)
    X = X + X.T
    Wii = np.where(Wii > wmin, Wii * (1 + c["kick"] * X), Wii)
    Wib = np.where(Wib > wmin, Wib * (1 + c["kick"] * rng.standard_normal(Wib.shape)), Wib)
    t0 = time.time()
    st, n, r = sim.equilibrate((Wii, Wib, Mii, Mib), SIGMA, P, 200000)
    s, lab, H, rows = sim.analyse(st, SIGMA, th, P, stable_at(th))
    before = sorted(np.unique(z["labels"][c["i_from"]], return_counts=True)[1].tolist(), reverse=True)
    out = dict(c, theta_from=float(GRID[c["i_from"]]), theta_to=th, sizes_before=before, sizes=s["sizes"],
               supports=s["supports"], populated=s["populated"], n_stable=s["n_stable"], coverage=s["coverage"],
               steps=n, residual=r, converged=bool(r < sim.RES_TOL), seconds=time.time() - t0)
    print(json.dumps(cm.clean(out, 4)), flush=True)
    return out


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "1"
    cases = CASES if which == "1" else CASES2
    with mp.get_context("fork").Pool(int(os.environ.get("P2_PROCS", "2"))) as p:
        res = p.map(run, cases, chunksize=1)
    cm.write_json(pp.RAW / f"floor_test{'' if which == '1' else '2'}.json", dict(cases=res), sig=5)
