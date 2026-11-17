"""Synthetic Gaussian checks of the theory (docs/nonlin.md §2.4), mean-field, numpy.

1. Threshold remap.  Ring covariance (B=24 unit-variance inputs on a circle, squared-
   exponential correlation, length 2).  Linear rule at theta = rho*; rectified rule (gain
   c^2 = 1/k_tau(1)) at theta = k~_tau(rho*) = k_tau(rho*)/k_tau(1), the same correlation
   threshold for unit-variance pairs.  If Gaussian rectification were only a remap of
   theta, K(rho*) and the parts would coincide.  Records K, part width (participation
   ratio), cluster sizes and energies.
2. Identifiability.  For each converged net, readout with theta known and theta fitted.

    PYTHONPATH=. python -m hebb.nonlin.synth  -> site/data/nonlin/synth_ring.json
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def ring_cov(B=24, ell=2.0):
    d = np.abs(np.arange(B)[:, None] - np.arange(B)[None, :]); d = np.minimum(d, B - d)
    return np.exp(-d ** 2 / (2 * ell ** 2))


def one(args):
    import torch
    from hebb import core
    from hebb.nonlin import dual, meanfield as MF, readout as NR
    from hebb.mnist.readout import Scorer
    rule, tau, rho_star, N, seed, T = args
    S = ring_cov()
    if rule == "linear":
        theta, c = rho_star, 1.0
    else:
        theta, c = float(dual.k(rho_star, tau) / dual.k(1.0, tau)), float(1 / np.sqrt(dual.relu_var(tau)))
    P = core.Params(theta=theta, dt=0.05, wmin=1e-3)
    t0 = time.time()
    out = MF.run(S, N, P, tau=tau, c=c, T=T, seed=seed, linear=(rule == "linear"))
    H = out["H"]
    cl = core.clusters_auto(H)
    lab = cl["labels"]; st = core.cluster_stats(H, lab, S)
    width = 1.0 / (st["parts"] ** 2).sum(1)
    big = st["sizes"] >= 2
    sc = Scorer(torch.as_tensor(S))
    Ht = torch.as_tensor(H); Wii = torch.as_tensor(out["Wii"]); Wib = torch.as_tensor(out["Wib"])
    if rule == "linear":
        from hebb.mnist import readout as SR
        eq = SR.equations(Wii, Wib, theta, 1, 1, 1, P.wmin)
        C, _ = SR.ls_readout(Ht, *eq, "both", iters=800)
        ro = sc.score(C); ro_fit = None
    else:
        eq = NR.rect_equations(Wii, Wib, theta, 1, 1, 1, P.wmin, c)
        C, C0, info = NR.gauss_readout(Ht, *eq, tau=tau, iters=400)
        ro = sc.score(C)
        eq0 = NR.rect_equations(Wii, Wib, 0.0, 1, 1, 1, P.wmin, c)
        Cf, _, inf = NR.gauss_readout(Ht, *eq0, tau=tau, iters=400, fit_theta=True, a_ii=1 / c ** 2, a_ib=1 / c ** 2)
        ro_fit = dict(theta_true=theta, theta_fit=inf["theta_fit"], **{k: v for k, v in sc.score(Cf).items() if k in ("known_relfro", "contrast_cos")})
    return dict(rule=rule, tau=tau, rho_star=rho_star, theta=theta, c=c, seed=seed, residual=out["residual"],
                K=int(cl["K"]), K_major=int(big.sum()), sizes=st["sizes"].tolist(),
                width_mean=float(np.average(width, weights=st["sizes"])),
                width_major=float(np.average(width[big], weights=st["sizes"][big])) if big.any() else None,
                energy=st["energy"].tolist(), parts=np.round(st["parts"], 4).tolist(),
                readout=dict(known_relfro=ro["known_relfro"], contrast_cos=ro["contrast_cos"]), readout_fit_theta=ro_fit,
                seconds=time.time() - t0)


def main():
    rhos = [0.3, 0.45, 0.6, 0.7, 0.8, 0.88, 0.94]
    jobs = []
    for seed in (0, 1):
        for r in rhos:
            jobs.append(("linear", 0.0, r, 96, seed, 4000))
            for tau in (0.0, 1.0):
                jobs.append(("rect", tau, r, 96, seed, 4000))
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as ex:
        res = list(ex.map(one, jobs))
    from hebb.nonlin import dual
    rr = np.linspace(-1, 1, 201)
    curves = {f"{t:g}": (dual.k(rr, t) / dual.k(1.0, t)).round(6).tolist() for t in (0.0, 0.5, 1.0, 2.0)}
    out = dict(description=__doc__, cov="ring B=24 ell=2 unit variance", N=96, T=4000, dt=0.05, wmin=1e-3,
               rho_grid=rr.round(4).tolist(), ktilde=curves, runs=res)
    p = ROOT / "site/data/nonlin/synth_ring.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    for r in res:
        print(r["rule"], r["tau"], r["rho_star"], r["seed"], "K", r["K"], r["K_major"], "w", round(r["width_mean"], 2),
              "res", f"{r['residual']:.1e}", "ro", round(r["readout"]["contrast_cos"], 3),
              "" if r["readout_fit_theta"] is None else f"theta {r['theta']:.3f} fit {r['readout_fit_theta']['theta_fit']:.3f}")


if __name__ == "__main__":
    main()
