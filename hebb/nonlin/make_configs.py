"""Write the Modal grid configs.  theta at fixed quantiles q of the off-diagonal BOUNDARY
covariance that the rule sees: Sigma_g for the linear rule, the (gain-normalised) pixel-rate
covariance c^2 Cov(s_b, s_c) for the rectified rule at its tau.  The Gaussian surrogate (B)
reuses A's c and theta (identical rule; the input differs only beyond second moments);
B' (gauss_own) uses theta at the same quantile of the surrogate's own boundary rate cov."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hebb.nonlin import dual
from hebb.nonlin.model import Data

QS = [0.92, 0.95, 0.97, 0.98, 0.99]
OUT = Path(__file__).resolve().parent / "configs"


def offq(S, qs):
    iu = np.triu_indices(len(S), 1)
    return [float(np.quantile(S[iu], q)) for q in qs]


def tables(res=14):
    D = Data(res)
    g = D.Gtr.double().numpy()
    tab = {"res": res, "q": QS, "linear": {"theta": offq(D.Sigma, QS), "c": 1.0}}
    for tau in (0.0, 0.5, 1.0):
        s = np.maximum(g - D.mu - tau * D.sd, 0)
        Ss = np.cov(s.T, bias=True)
        c = float(1 / np.sqrt(np.diag(Ss).mean()))
        sd = D.sd; den = np.outer(sd, sd)
        rho = np.divide(D.Sigma, den, out=np.zeros_like(den), where=den > 0)
        Sg = den * dual.k(rho, tau)                       # Gaussian surrogate's boundary rate cov
        tab[f"rect_t{tau:g}"] = {"theta": offq(c * c * Ss, QS), "c": c,
                                 "gauss_own_theta": offq(c * c * Sg, QS),
                                 "gauss_mean_rate_var": float(c * c * np.diag(Sg).mean())}
    return tab


def main():
    OUT.mkdir(exist_ok=True)
    t14 = tables(14)
    (OUT / "theta_table14.json").write_text(json.dumps(t14, indent=1))
    units = []
    for i, q in enumerate(QS):
        thl = round(t14["linear"]["theta"][i], 4)
        for seed in (0, 1):
            units.append(dict(rule="linear", input="pixels", theta=thl, seed=seed, extra=dict(q=q, exp="A_lin")))
        units.append(dict(rule="linear", input="gauss", theta=thl, seed=0, extra=dict(q=q, exp="B_lin")))
        for tau in (0.0, 0.5, 1.0):
            T = t14[f"rect_t{tau:g}"]; th = round(T["theta"][i], 4); c = T["c"]
            for seed in ((0, 1) if tau == 0.0 else (0,)):
                units.append(dict(rule="rect", tau=tau, input="pixels", theta=th, c=c, seed=seed, extra=dict(q=q, exp="A_rect")))
            if tau in (0.0, 1.0):
                units.append(dict(rule="rect", tau=tau, input="gauss", theta=th, c=c, seed=0, extra=dict(q=q, exp="B_rect")))
            if tau == 0.0:
                units.append(dict(rule="rect", tau=tau, input="gauss", theta=round(T["gauss_own_theta"][i], 4), c=c, seed=0,
                                  extra=dict(q=q, exp="Bown_rect")))
    cfg = {"defaults": dict(N=980, steps=3000, n_snap=24, n_sub=256, nls_iters=300), "units": units}
    (OUT / "main14.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(t14, indent=1)); print(len(units), "units")


if __name__ == "__main__":
    main()
