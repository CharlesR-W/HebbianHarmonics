"""Q-fusion (SPEC §3): is the fixed point the stated one; the two-neuron calibration; N neurons:
cluster count and diameter vs b and alpha, continuation up/down (hysteresis), nestedness.

Run: uv run python -m hebb.sweeps2.fusion   -> site/data/sweeps2/fusion.json
"""
import hebb.sweeps2  # noqa: F401  (pins BLAS threads)

import numpy as np

from hebb import onsite as o
from hebb import onsite_analysis as oa

# ----------------------------------------------------------------------------- part A: two neurons
# Two neurons, each attached with conductance deg to its own input; inputs uncorrelated with
# variance sigma2 each, so the uncoupled distance is D0^2 = 2 sigma2. Linear case:
#   (deg + 2w)(u1 - u2) = deg (g1 - g2)  =>  D^2(w) = D0^2 / (1 + 2w/deg)^2
#   dw/dt = b - alpha w - D^2(w)/2.
# With c = D0^2/2 and kappa = alpha*deg the saddle-node of the interior branch is at
#   y* = (4c/kappa)^(1/3),  b_sn = 3 * 2^(-4/3) (c kappa^2)^(1/3) - kappa/2   (exists iff y* > 1, i.e. c > kappa/4),
# and w = 0 is stable for b < c (unstable for b > c). Bistable window: b_sn < b < c, non-empty iff c > kappa/4.
# (At fixed b, varying c, hysteresis requires b > kappa/4: see resolution.py.)
G4 = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])   # C_hat = I exactly


def pair_drift(w, b, alpha, D02, deg=1.0):
    return b - alpha * w - 0.5 * D02 / (1 + 2 * w / deg) ** 2


def pair_fixed_points(b, alpha, D02, deg=1.0):
    """All fixed points (w=0 if its projected drift is <= 0, plus interior roots) with stability."""
    from scipy.optimize import brentq
    out = []
    if pair_drift(0.0, b, alpha, D02, deg) <= 0:
        out.append((0.0, True))
    ws = np.linspace(0, b / alpha + 1e-9, 400001)
    v = pair_drift(ws, b, alpha, D02, deg)
    for i in np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]:
        w = brentq(pair_drift, ws[i], ws[i + 1], args=(b, alpha, D02, deg), xtol=1e-14)
        h = 1e-7
        stable = (pair_drift(w + h, b, alpha, D02, deg) - pair_drift(w - h, b, alpha, D02, deg)) < 0
        out.append((w, bool(stable)))
    return out


def b_saddle(alpha, D02, deg=1.0):
    c, k = D02 / 2, alpha * deg
    if 4 * c / k <= 1:
        return c
    return 3 * 2 ** (-4 / 3) * (c * k * k) ** (1 / 3) - k / 2


def pair_net(w0, D02, deg=1.0):
    return o.Net(A=deg * np.eye(2), t=0.0, klo=0.0, khi=0.0, W=np.array([[0.0, w0], [w0, 0.0]])), \
        G4 * np.sqrt(D02 / 2)


def part_pair():
    out = {}
    # SPEC example, both readings of D0^2
    ex = []
    for D02, b, alpha in [(2.0, 0.2, 0.1), (1.0, 0.2, 0.1), (2.0, 0.4, 0.2)]:
        fps = pair_fixed_points(b, alpha, D02)
        sims = []
        for w0 in (0.0, 0.5, 3.0):
            net, G = pair_net(w0, D02)
            r = oa.solve(net, G, b, alpha, dt=0.05, T=200000)
            sims.append(dict(w0=w0, w=net.W[0, 1], residual=r["residual"], converged=r["converged"]))
        ex.append(dict(D02=D02, b=b, alpha=alpha, fixed_points=[[w, s] for w, s in fps],
                       b_sn=b_saddle(alpha, D02), b_on=D02 / 2, sims=sims))
    out["spec_example"] = ex

    # grid check: numerical end states from w0 = 0 and w0 = b/alpha vs analytic stable points
    grid = []
    maxerr = 0.0
    for D02 in (1.0, 2.0):
        for alpha in (0.05, 0.1, 0.2, 0.5, 1.0):
            for b in np.round(np.geomspace(0.05, 1.5, 12), 4):
                stable = [w for w, s in pair_fixed_points(b, alpha, D02) if s]
                ends = []
                for w0 in (0.0, b / alpha):
                    net, G = pair_net(w0, D02)
                    r = oa.solve(net, G, b, alpha, dt=min(0.05, 0.5 / alpha), T=400000)
                    w = net.W[0, 1]
                    err = min(abs(w - s) for s in stable)
                    maxerr = max(maxerr, err)
                    ends.append([w, r["residual"]])
                grid.append([D02, alpha, float(b), len(stable), ends[0][0], ends[1][0],
                             max(ends[0][1], ends[1][1])])
    out["grid"] = dict(columns=["D02", "alpha", "b", "n_stable_analytic", "w_end_from_0", "w_end_from_top",
                                "max_residual"], rows=grid, max_err_vs_analytic=maxerr)

    # analytic phase diagram (explorer): b_sn(alpha) and b_on for D0^2 = 1, 2
    al = np.geomspace(0.01, 4, 60)
    out["phase"] = dict(alpha=al, b_sn_D02_2=[b_saddle(a, 2.0) for a in al],
                        b_sn_D02_1=[b_saddle(a, 1.0) for a in al], b_on_D02_2=1.0, b_on_D02_1=0.5)

    # numerical hysteresis loop (continuation in b), D0^2 = 2, alpha = 0.2
    alpha, D02 = 0.2, 2.0
    bs = np.round(np.linspace(0.1, 1.3, 61), 4)
    net, G = pair_net(0.0, D02)
    up = []
    for b in bs:
        oa.solve(net, G, b, alpha, dt=0.05, T=400000)
        up.append(net.W[0, 1])
    down = []
    for b in bs[::-1]:
        oa.solve(net, G, b, alpha, dt=0.05, T=400000)
        down.append(net.W[0, 1])
    down = down[::-1]
    up, down = np.array(up), np.array(down)
    out["loop"] = dict(alpha=alpha, D02=D02, b=bs, w_up=up, w_down=down,
                       b_switch_up=float(bs[np.argmax(up > 1e-6)]),
                       b_switch_down=float(bs[np.argmax(down > 1e-6)]),
                       b_on=D02 / 2, b_sn=b_saddle(alpha, D02),
                       grid_step=float(bs[1] - bs[0]))
    # drift curves for the explorer
    wgrid = np.linspace(0, 4, 81)
    out["drift_curves"] = dict(w=wgrid, curves=[dict(D02=D02, b=b, alpha=a, drift=pair_drift(wgrid, b, a, D02))
                                                 for D02, b, a in [(2.0, 0.2, 0.1), (1.0, 0.2, 0.1),
                                                                   (2.0, 0.4, 0.2)]])
    return out


# ----------------------------------------------------------------------------- part B: N neurons
N, B, S_ATT = 40, 12, 3
BGRID = np.round(np.geomspace(0.005, 2.0, 15), 5)
SIGMAS = dict(blocks=oa.sigma_blocks(B), power=oa.sigma_power(B))
POTS = dict(linear=oa.LINEAR, rect=oa.RECT)
ALPHAS = (0.1, 1.0)
SEEDS = dict(linear=[0, 1, 2, 3, 4], rect=[0, 1, 2])


def J_value(net, U, G, b, alpha):
    iu = np.triu_indices(net.N, 1)
    w = net.W[iu]
    return o.energy(net, U, G) + (0.5 * alpha * w ** 2 - b * w).sum()


def record(net, r, G, Ge, b, alpha):
    U = r["U"]
    cs = oa.cluster_summary(net.W, U, b)
    Ue = o.equilibrate(net, Ge)
    cse = o.cluster_stats(Ue, cs["lab"])
    D2t, D2e = o.dist2(U), o.dist2(Ue)
    fp = oa.fixed_point_check(net.W, U, b, alpha)
    return dict(
        K=cs["K"], K_multi=cs["K_multi"], largest=cs["largest"], diam_max=cs["diam_max"],
        diam_mean=cs["diam_mean"], span_max=cs["span_max"], span_over_range=cs["span_over_range"],
        diam_max_eval=float(max(c["diam_max"] for c in cse)),
        span_over_range_eval=float(max(np.sqrt(np.clip(D2e[np.ix_(c["members"], c["members"])], 0, None)).max()
                                       for c in cse) / np.sqrt(2 * b)),
        d2_sampling=float(np.abs(D2t - D2e).max() / D2e.max()),
        n_live=fp["n_live"], dev_live=fp["dev_live"], slack_dead=fp["slack_dead"],
        residual=r["residual"], converged=r["converged"], steps=r["steps"],
        J=J_value(net, U, G, b, alpha), lab=cs["lab"])


def envelope_check(net, G, b, alpha):
    """Central-difference dE*/dw_ij vs D^2/2 on two live and two dead pairs at a fixed point."""
    U = o.equilibrate(net, G)
    D2 = o.dist2(U)
    iu = np.triu_indices(net.N, 1)
    live = np.nonzero(net.W[iu] > 1e-9)[0][:2]
    dead = np.nonzero(net.W[iu] <= 1e-9)[0][:2]
    errs = []
    for k in list(live) + list(dead):
        i, j = iu[0][k], iu[1][k]
        h = 1e-5
        es = []
        for sgn in (1, -1):
            net.W[i, j] += sgn * h; net.W[j, i] += sgn * h
            es.append(o.energy(net, o.equilibrate(net, G), G))
            net.W[i, j] -= sgn * h; net.W[j, i] -= sgn * h
        errs.append(abs((es[0] - es[1]) / (2 * h) - 0.5 * D2[i, j]))
    return float(max(errs))


def unit(args):
    sig, pot, alpha, seed = args
    Sigma = SIGMAS[sig]
    G, Ge = oa.banks(Sigma, seed)
    proto = lambda: oa.make_net(N, B, seed, s=S_ATT, **POTS[pot])  # noqa: E731
    cold, up, down = [], [], []
    for b in BGRID:
        net = proto()
        r = oa.solve(net, G, b, alpha)
        cold.append(record(net, r, G, Ge, b, alpha))
        if b == BGRID[len(BGRID) // 2]:
            env = envelope_check(net, G, b, alpha)
    net = proto()
    for b in BGRID:
        r = oa.solve(net, G, b, alpha)
        up.append(record(net, r, G, Ge, b, alpha))
    net = proto()
    for b in BGRID[::-1]:
        r = oa.solve(net, G, b, alpha)
        down.append(record(net, r, G, Ge, b, alpha))
    down = down[::-1]
    nest_cold = [oa.pair_violation(cold[k]["lab"], cold[k + 1]["lab"]) for k in range(len(BGRID) - 1)]
    nest_down = [oa.pair_violation(down[k]["lab"], down[k + 1]["lab"]) for k in range(len(BGRID) - 1)]
    hyst = [oa.pair_violation(u["lab"], d["lab"]) + oa.pair_violation(d["lab"], u["lab"]) for u, d in zip(up, down)]
    print(f"done {sig} {pot} alpha={alpha} seed={seed}", flush=True)
    return dict(sig=sig, pot=pot, alpha=alpha, seed=seed, cold=cold, up=up, down=down,
                nest_cold=nest_cold, nest_down=nest_down, part_diff_up_down=hyst, envelope_err=env)


SCALARS = ["K", "K_multi", "largest", "diam_max", "diam_mean", "span_max", "span_over_range", "diam_max_eval",
           "span_over_range_eval",
           "d2_sampling", "n_live", "dev_live", "slack_dead", "residual", "converged", "steps", "J"]


def main():
    pair = part_pair()
    print("pair done", pair["loop"]["b_switch_up"], pair["loop"]["b_switch_down"], pair["grid"]["max_err_vs_analytic"])
    jobs = [(sg, pt, a, s) for sg in SIGMAS for pt in POTS for a in ALPHAS for s in SEEDS[pt]]
    # put the slow rectifying small-alpha jobs first for load balance
    jobs.sort(key=lambda j: (j[1] != "rect", -1 / j[2]))
    res = oa.pool_map(unit, jobs)
    runs = []
    for u in res:
        row = dict(sigma=u["sig"], pot=u["pot"], alpha=u["alpha"], seed=u["seed"],
                   nest_cold=u["nest_cold"], nest_down=u["nest_down"], part_diff_up_down=u["part_diff_up_down"],
                   envelope_err=u["envelope_err"])
        for proto in ("cold", "up", "down"):
            row[proto] = {k: [rec[k] for rec in u[proto]] for k in SCALARS}
            if u["seed"] == 0:
                row[proto]["lab"] = [rec["lab"] for rec in u[proto]]
        runs.append(row)
    allres = [x for u in res for p in ("cold", "up", "down") for x in u[p]]
    summary = dict(
        n_runs=len(allres), n_converged=int(sum(x["converged"] for x in allres)),
        max_residual=float(max(x["residual"] for x in allres)),
        max_dev_live_converged=float(max(x["dev_live"] for x in allres if x["converged"])),
        min_slack_dead_converged=float(min(x["slack_dead"] for x in allres if x["converged"])),
        max_envelope_err=float(max(u["envelope_err"] for u in res)),
    )
    meta = dict(
        script="hebb/sweeps2/fusion.py",
        description="Q-fusion: two-neuron calibration (analytic vs simulated, SPEC example), and N=40 neuron "
                    "networks swept in b from W=0 (cold) and by continuation up/down, with cluster count, "
                    "diameters, fixed-point identity checks, nestedness and hysteresis.",
        config=dict(N=N, B=B, s=S_ATT, deg=1.0, S_train=oa.S_TRAIN, S_eval=oa.S_EVAL, b_grid=BGRID,
                    alphas=ALPHAS, sigmas=dict(blocks="sigma_blocks(B=12, var=(2,1,.5,.25), rho=.8)",
                                               power="sigma_power(B=12, exponent=1, trace B)"),
                    pots=POTS, dt="min(0.5, 1/alpha)", run_tol=oa.RUN_TOL, max_steps=40000,
                    pair="two neurons, A = I (deg 1), bank = 4 rows (+-1,+-1) * sqrt(D0^2/2) so C_hat = sigma2*I exactly"),
        seeds=SEEDS,
        fields={
            "pair.spec_example": "SPEC example (b=.2, alpha=.1) for D0^2 = 2 (unit-variance inputs) and 1; fixed_points [w, stable]; sims from w0 in {0,.5,3}",
            "pair.grid.rows": "[D0^2, alpha, b, #stable analytic fixed points, w end from 0, w end from b/alpha, max residual] (w in conductance units, same as deg)",
            "pair.phase": "analytic bistability window: b_sn(alpha) < b < b_on = D0^2/2 (b in D^2 units), non-empty iff D0^2/2 > alpha/4",
            "pair.loop": "continuation in b (step grid_step) up then down; w_up/w_down end states; switch points vs analytic b_on and b_sn",
            "runs[].cold|up|down.<field>[k]": "value at b_grid[k]; cold = from W=0, up/down = continuation carrying W",
            "K": "number of connected components of the live graph (w > 1e-9), singletons included",
            "K_multi": "components with >= 2 members",
            "largest": "size of the largest component (neurons)",
            "diam_max": "max over components of max RMS distance D of a member to the prototype (response units, training bank)",
            "diam_mean": "mean over multi-member components of the mean member-to-prototype D (training bank)",
            "span_max": "max pairwise D inside any component (training bank)",
            "span_over_range": "span_max / sqrt(2b): > 1 means a component is wider than one edge's range (a chain)",
            "diam_max_eval": "diam_max recomputed on the evaluation bank (sampling check)",
            "span_over_range_eval": "span_over_range with D from the evaluation bank (independent population geometry)",
            "d2_sampling": "max |D2_train - D2_eval| / max D2_eval over pairs (sampling error)",
            "n_live": "live edges (w > 1e-9)",
            "dev_live": "max over live edges |alpha w - (b - D^2/2)| (fixed-point identity error, D^2 units)",
            "slack_dead": "min over dead pairs of D^2 - 2b (>= -residual expected)",
            "residual": "projected residual max|dw/dt| at the end",
            "converged": "residual < res_tol",
            "J": "J(W) = mean energy over the training bank + sum_{i<j} (alpha/2 w^2 - b w)",
            "lab": "component labels per neuron (seed 0 only)",
            "nest_cold": "pair_violation(lab at b_k, lab at b_{k+1}) for cold runs: fraction of pairs together at the smaller b but apart at the larger b (0 = nested refinement)",
            "nest_down": "same along the down-continuation",
            "part_diff_up_down": "pair_violation(up,down)+pair_violation(down,up) at each b (0 = same partition, no hysteresis)",
            "envelope_err": "max |central-difference dE*/dw - D^2/2| on 2 live + 2 dead pairs at the mid-grid cold fixed point",
        })
    oa.dump("fusion", meta, dict(pair=pair, summary=summary, runs=runs), sig=4)
    print(summary)


if __name__ == "__main__":
    main()
