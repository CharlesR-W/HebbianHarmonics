"""Reading the weights of the covariance rule with pruning (page `readout`, figure "It depends on the rule").

The covariance rule of `hebb/core.py` (every synapse plastic, input synapses included):
    dw/dt = eta * (Cov(u_i, u_j) - theta) - lam * w,   clipped at the floor w_min,
trained in mean field on the same 12-input block covariance as the distance-rule figures
(oa.sigma_blocks(12): four blocks of three inputs, variances 2 / 1 / 0.5 / 0.25, within-block correlation 0.8).

At its fixed point every synapse above the floor gives one linear equation on the input covariance C:
    lateral   gamma * h_i^T C h_j = theta + (lam/eta) w_ij
    input     (H C)_ib            = theta + (lam/eta) w_ib
with H computed from the weights. We solve them exactly as `core.readout_ls` does (masked ridge least squares,
theta, lam, eta, gamma known, direct solve, PSD projection) and compare the contrast part P C P with P Sigma P,
the same comparison the page makes for the distance rule. We also report the rank of the measurement operator on
the 78-dimensional space of symmetric 12 x 12 matrices (core.measurement_spectrum).

Run: uv run python -m hebb.readout_cov_rule   -> site/data/readout/cov_rule.json   (a few seconds on one CPU)
"""
import json
import os
import pathlib

import numpy as np

from hebb import core
from hebb.onsite_analysis import sigma_blocks

OUT = pathlib.Path(__file__).resolve().parents[1] / "site" / "data" / "readout" / "cov_rule.json"
B, N = 12, 80
SEEDS = (0, 1, 2, 3, 4)
PARAMS = dict(theta=0.3, lam=1.0, eta=1.0, gamma=1.0, wmin=1e-3, dt=0.05)
RES_TOL = 1e-9


def train(Sigma, seed):
    P = core.Params(**PARAMS)
    Wii, Wib, Mii, Mib = core.init_state(N, B, np.random.default_rng(seed))
    r = np.inf
    for t in range(1, 200001):
        Wii, Wib, _ = core.step(Wii, Wib, Mii, Mib, Sigma, P)
        if t % 100 == 0:
            r = core.stationarity(Wii, Wib, Mii, Mib, Sigma, P)["r"]
            if r < RES_TOL:
                break
    return P, Wii, Wib, Mii, Mib, t, r


def unit(Sigma, seed):
    P, Wii, Wib, Mii, Mib, steps, r = train(Sigma, seed)
    Wc, Wbc = np.maximum(Wii, P.wmin * Mii), np.maximum(Wib, P.wmin * Mib)
    H = core.harmonic(Wc, Wbc)
    ro = core.readout_state(Wii, Wib, Mii, Mib, P, method="direct", psd=True)
    eq = core.readout_equations(Wii, Wib, Mii, Mib, P.theta, P.lam, P.eta, P.gamma, P.wmin)
    ms = core.measurement_spectrum(H, eq)
    C = ro["C"]
    PC, PS = core.project1(C), core.project1(Sigma)
    rel = float(np.linalg.norm(PC - PS) / np.linalg.norm(PS))
    cos = float((PC * PS).sum() / (np.linalg.norm(PC) * np.linalg.norm(PS)))
    K = core.clusters_auto(H)["K"]
    return dict(seed=seed, steps=steps, residual=float(r), converged=bool(r < RES_TOL), K=int(K),
                m_ii=int(ro["m_ii"]), m_ib=int(ro["m_ib"]), rank=int(ms["both"]["rank"]), dim=int(ms["p"]),
                relerr_PCP=rel, cos_PCP=cos, C=C, PCP=PC)


def main():
    Sigma = sigma_blocks(B)
    runs = [unit(Sigma, s) for s in SEEDS]
    for u in runs:
        print(f"seed {u['seed']}: steps {u['steps']} r {u['residual']:.1e} K {u['K']} rank {u['rank']}/{u['dim']} "
              f"rel {u['relerr_PCP']:.3f} cos {u['cos_PCP']:.3f}")
    r0 = runs[0]
    rnd = lambda M: [[round(float(v), 6) for v in row] for row in M]  # noqa: E731
    out = dict(
        meta=dict(script="hebb/readout_cov_rule.py",
                  description="Covariance rule with pruning (hebb/core.py, mean field, every synapse plastic): "
                              "fixed-point readout of the input covariance (core.readout_state, method direct, "
                              "theta known, PSD) compared on the contrast space with P Sigma P.",
                  config=dict(B=B, N=N, seeds=list(SEEDS), **PARAMS, res_tol=RES_TOL, init="core.init_state defaults",
                              Sigma="oa.sigma_blocks(12): 4 blocks of 3, variances 2/1/0.5/0.25, within 0.8")),
        Sigma=rnd(Sigma), PSigmaP=rnd(core.project1(Sigma)),
        example=dict(seed=r0["seed"], C=rnd(r0["C"]), PCP=rnd(r0["PCP"])),
        runs=[{k: v for k, v in u.items() if k not in ("C", "PCP")} for u in runs],
        summary=dict(rank=[min(u["rank"] for u in runs), max(u["rank"] for u in runs)],
                     relerr_PCP=[min(u["relerr_PCP"] for u in runs), max(u["relerr_PCP"] for u in runs)],
                     K=[min(u["K"] for u in runs), max(u["K"] for u in runs)],
                     all_converged=all(u["converged"] for u in runs)))
    os.makedirs(OUT.parent, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
