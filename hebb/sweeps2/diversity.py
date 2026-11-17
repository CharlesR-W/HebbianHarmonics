"""Q-diversity (T1, SPEC §3). Decisive controlled experiment: rank-one (scalar-drive) input
g = z v, matched attachments, three arms that differ ONLY in the on-site potential:
  hetero : klo=3, khi=0, thresholds t_i ~ N(0, 0.7^2)   (heterogeneous leaky rectifiers)
  homo   : klo=3, khi=0, t_i = 0                        (homogeneous)
  linear : klo=khi=0
Measures on the evaluation bank (independent of learning), split into fit/test halves:
  affine_err  held-out relative MSE of an affine fit of each response from the FULL input g
              (0 = the neuron is an affine function of the input; measures nonlinearity)
  spline_err  held-out relative MSE of a flexible 1-D fit from the planted scalar z
              (0 = single-valued function of the shared mode)
  descriptive, per multi-member component: d_eff of member covariance, proto_err (held-out
  relative MSE of member ~ a + c * prototype: 0 = members are affine copies of the prototype),
  numerical rank of the centred member responses (singular values > 1e-3 of the largest).
Then the same on multi-mode planted-block inputs, and the dependence on b, alpha.

Run: uv run python -m hebb.sweeps2.diversity   -> site/data/sweeps2/diversity.json
"""
import hebb.sweeps2  # noqa: F401

import numpy as np

from hebb import onsite as o
from hebb import onsite_analysis as oa

N, B, S_ATT = 40, 12, 3
ARMS = dict(hetero=oa.RECT, homo=dict(klo=3.0, khi=0.0, sigma_t=0.0), linear=oa.LINEAR)
V1 = np.linspace(0.5, 1.5, B)
V1 = V1 / np.sqrt((V1 ** 2).mean())            # positive drive vector, mean-square 1
BGRID = np.round(np.geomspace(0.003, 0.3, 7), 5)
ALPHAS = (0.1, 1.0, 10.0)
SEEDS = [0, 1, 2, 3, 4]
DEFAULT = dict(b=float(BGRID[3]), alpha=1.0)
ZGRID = np.linspace(-2.5, 2.5, 41)


# ----------------------------------------------------------------------------- banks
def rank1_banks(seed):
    z = oa.rng_for(seed, 1).standard_normal(oa.S_TRAIN)
    ze = oa.rng_for(seed, 2).standard_normal(oa.S_EVAL)
    return z[:, None] * V1, z, ze[:, None] * V1, ze


NBLK = 4
VAR = np.array([2.0, 1.0, 0.5, 0.25])


def block_banks(seed, rho):
    """Planted block inputs: block k shares scalar z_k (variance rho*var_k per input) plus private
    noise (1-rho)*var_k. Covariance = oa.sigma_blocks(B, VAR, rho)."""
    m = B // NBLK
    out = []
    for stream, S in ((1, oa.S_TRAIN), (2, oa.S_EVAL)):
        rng = oa.rng_for(seed, stream)
        Z = rng.standard_normal((S, NBLK))
        E = rng.standard_normal((S, B))
        sd_s = np.repeat(np.sqrt(rho * VAR), m)
        sd_p = np.repeat(np.sqrt((1 - rho) * VAR), m)
        out += [np.repeat(Z, m, axis=1) * sd_s + E * sd_p, Z]
    return out


# ----------------------------------------------------------------------------- measures
def neuron_measures(Ue, Ge, Ze):
    """Ze: (S_eval, k) planted scalars. Returns affine_err (N,), best single-scalar spline_err (N,)."""
    h = len(Ue) // 2
    aff = oa.affine_fit_err(Ue[:h], Ge[:h], Ue[h:], Ge[h:])
    spl = np.min([oa.spline_fit_err(Ue[:h], Ze[:h, k], Ue[h:], Ze[h:, k]) for k in range(Ze.shape[1])], axis=0)
    return aff, spl


def cluster_measures(Ue, lab):
    h = len(Ue) // 2
    rows = []
    for k in range(lab.max() + 1):
        idx = np.nonzero(lab == k)[0]
        if len(idx) < 2:
            continue
        X = Ue[:, idx]
        proto = X.mean(1)
        Xc = X - X.mean(0)
        C = Xc.T @ Xc / len(X)
        sv = np.linalg.svd(Xc, compute_uv=False)
        rank = int((sv > 1e-3 * sv[0]).sum()) if sv[0] > 0 else 0
        perr = oa.affine_fit_err(X[:h], proto[:h, None], X[h:], proto[h:, None])
        # residual of each member after its affine regression on the prototype (all of the bank)
        P1 = np.stack([np.ones(len(X)), proto], 1)
        R = X - P1 @ np.linalg.lstsq(P1, X, rcond=None)[0]
        Cr = R.T @ R / len(X)
        D = np.sqrt(((X - proto[:, None]) ** 2).mean(0))
        rows.append(dict(size=len(idx), deff=oa.deff(C), rank=rank, proto_err=float(perr.mean()), deff_resid=oa.deff(Cr),
                         diam_max=float(D.max()), diam_mean=float(D.mean())))
    return rows


def summarise(net, r, b, G, Ge, Ze):
    Ue = o.equilibrate(net, Ge)
    aff, spl = neuron_measures(Ue, Ge, Ze)
    cs = oa.cluster_summary(net.W, r["U"], b)
    cl = cluster_measures(Ue, cs["lab"])
    n_in = sum(c["size"] for c in cl)
    wavg = (lambda key: float(sum(c[key] * c["size"] for c in cl) / n_in)) if cl else (lambda key: float("nan"))
    return dict(K=cs["K"], K_multi=cs["K_multi"], frac_in_multi=n_in / net.N,
                affine_err_mean=float(aff.mean()), affine_err_max=float(aff.max()),
                spline_err_mean=float(spl.mean()), spline_err_max=float(spl.max()),
                deff_w=wavg("deff"), rank_w=wavg("rank"), proto_err_w=wavg("proto_err"), deff_resid_w=wavg("deff_resid"),
                diam_mean_w=wavg("diam_mean"), residual=r["residual"], converged=r["converged"]), Ue, cs["lab"]


# ----------------------------------------------------------------------------- units
def rank1_unit(args):
    arm, alpha, seed = args
    G, z, Ge, ze = rank1_banks(seed)
    A = o.make_attachments(N, B, S_ATT, oa.rng_for(seed, 0))
    out = []
    for b in BGRID:
        net = oa.make_net(N, B, seed, A=A, **ARMS[arm])
        r = oa.solve(net, G, b, alpha)
        s, Ue, lab = summarise(net, r, b, G, Ge, ze[:, None])
        out.append(s)
        if seed == 0 and b == DEFAULT["b"] and alpha == DEFAULT["alpha"]:
            # response curves u_i(z) on a grid for member-vs-mean plots
            curves = o.equilibrate(net, ZGRID[:, None] * V1).T
            preset = dict(lab=lab, curves=curves, t=net.t, W_live=int((net.W > 1e-9).sum() // 2))
    # uncoupled (W = 0) reference for the same arm: diversity before any fusion
    net0 = oa.make_net(N, B, seed, A=A, **ARMS[arm])
    U0e = o.equilibrate(net0, Ge)
    aff0, spl0 = neuron_measures(U0e, Ge, ze[:, None])
    res = dict(arm=arm, alpha=alpha, seed=seed, runs=out,
               uncoupled=dict(affine_err_mean=float(aff0.mean()), spline_err_max=float(spl0.max())))
    if seed == 0 and alpha == DEFAULT["alpha"]:
        res["preset"] = preset
    print(f"rank1 {arm} alpha={alpha} seed={seed}", flush=True)
    return res


MM_B = np.round(np.geomspace(0.01, 0.3, 5), 5)


def multi_unit(args):
    arm, rho, attach, seed = args
    G, _, Ge, Ze = block_banks(seed, rho)
    if attach == "random":
        A = o.make_attachments(N, B, S_ATT, oa.rng_for(seed, 0))
    else:
        A, _ = oa.block_attachments(N, B, NBLK, S_ATT, seed)
    out = []
    for b in MM_B:
        net = oa.make_net(N, B, seed, A=A, **ARMS[arm])
        r = oa.solve(net, G, b, 1.0)
        out.append(summarise(net, r, b, G, Ge, Ze)[0])
    print(f"multi {arm} rho={rho} {attach} seed={seed}", flush=True)
    return dict(arm=arm, rho=rho, attach=attach, seed=seed, runs=out)


KEYS = ["K", "K_multi", "frac_in_multi", "affine_err_mean", "affine_err_max", "spline_err_mean", "spline_err_max",
        "deff_w", "rank_w", "proto_err_w", "deff_resid_w", "diam_mean_w", "residual", "converged"]

STRENGTH = [(1.0, 0.0), (3.0, 0.0), (10.0, 0.0), (30.0, 0.0), (0.0, 3.0)]
TSPREAD = [0.35, 0.7, 1.4]


def strength_unit(args):
    """Rank-one drive at the default (b, alpha): does a stronger or differently shaped potential, or a
    wider threshold spread, give more within-cluster diversity?"""
    klo, khi, st, seed = args
    G, z, Ge, ze = rank1_banks(seed)
    A = o.make_attachments(N, B, S_ATT, oa.rng_for(seed, 0))
    net = oa.make_net(N, B, seed, A=A, klo=klo, khi=khi, sigma_t=st)
    r = oa.solve(net, G, DEFAULT["b"], DEFAULT["alpha"])
    s = summarise(net, r, DEFAULT["b"], G, Ge, ze[:, None])[0]
    return dict(klo=klo, khi=khi, sigma_t=st, seed=seed, **s)


def main():
    jobs = [(arm, al, s) for arm in ARMS for al in ALPHAS for s in SEEDS]
    jobs.sort(key=lambda j: (j[0] == "linear", j[1]))
    r1 = oa.pool_map(rank1_unit, jobs)
    mjobs = [(arm, rho, att, s) for arm in ARMS for rho in (1.0, 0.8) for att in ("random", "block") for s in SEEDS[:3]]
    mm = oa.pool_map(multi_unit, mjobs)
    sjobs = [(kl, kh, st, s) for kl, kh in STRENGTH for st in TSPREAD for s in SEEDS]
    strength = oa.pool_map(strength_unit, sjobs)
    rank1 = [dict(arm=u["arm"], alpha=u["alpha"], seed=u["seed"], uncoupled=u["uncoupled"],
                  **{k: [x[k] for x in u["runs"]] for k in KEYS}) for u in r1]
    presets = {u["arm"]: u["preset"] for u in r1 if "preset" in u}
    multi = [dict(arm=u["arm"], rho=u["rho"], attach=u["attach"], seed=u["seed"],
                  **{k: [x[k] for x in u["runs"]] for k in KEYS}) for u in mm]
    allr = [x for u in r1 + mm for x in u["runs"]] + strength
    meta = dict(
        script="hebb/sweeps2/diversity.py",
        description="Q-diversity: rank-one scalar drive, three arms with matched attachments (hetero/homo nonlinear, "
                    "linear), held-out affine and 1-D fits, within-cluster descriptors, vs b and alpha; then "
                    "planted multi-block inputs.",
        config=dict(N=N, B=B, s=S_ATT, deg=1.0, v="linspace(0.5,1.5,12), mean-square 1", arms=ARMS, b_grid=BGRID,
                    alphas=ALPHAS, default=DEFAULT, S_train=oa.S_TRAIN, S_eval=oa.S_EVAL,
                    eval_split="first half fits, second half scores", spline="piecewise linear, 24 knots at quantiles",
                    multi=dict(blocks=NBLK, var=VAR, rho=[1.0, 0.8], attach=["random s=3", "block: s=3 inside one block"],
                               b_grid=MM_B, alpha=1.0, planted="z_k block scalars; spline fit uses the best single z_k")),
        seeds=dict(rank1=SEEDS, multi=SEEDS[:3]),
        n_runs=len(allr), n_converged=int(sum(x["converged"] for x in allr)),
        fields={
            "rank1[].<key>[k]": "value at b_grid[k] for (arm, alpha, seed)",
            "K, K_multi": "components / components with >= 2 members",
            "frac_in_multi": "fraction of neurons in multi-member components",
            "affine_err_mean/max": "held-out relative MSE (dimensionless, MSE / variance) of the affine fit from the full input, mean/max over neurons",
            "spline_err_mean/max": "held-out relative MSE of the 1-D fit from the planted scalar (multi: best single z_k)",
            "deff_w, rank_w, proto_err_w, deff_resid_w, diam_mean_w": "per multi-member component, averaged weighted by size: d_eff (dimensionless), numerical rank (sv > 1e-3 sv_max), held-out relative MSE of member ~ a + c*prototype, d_eff of the members' residuals after that regression, mean RMS distance to prototype (response units)",
            "strength[]": "rank-one drive at default (b, alpha), potentials (klo, khi) in %s x threshold spread sigma_t in %s: same measures" % (STRENGTH, TSPREAD),
            "uncoupled": "same neuron measures at W = 0 (before learning)",
            "presets.<arm>": "seed 0 at default (b, alpha): component labels, response curves u_i(z) on z_grid (rows = neurons), thresholds t",
            "residual, converged": "projected residual and residual < res_tol",
        })
    oa.dump("diversity", meta, dict(z_grid=ZGRID, rank1=rank1, presets=presets, multi=multi, strength=strength), sig=4)


if __name__ == "__main__":
    main()
