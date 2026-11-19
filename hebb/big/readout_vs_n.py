"""Family C: readout of P C_hat P from the weights vs N at fixed B = 12 (linear distance rule).

N in {40, 80, 160, 320, 500}, seeds 0-2, b in {0.01, 0.036, 0.1}, alpha = 1, Sigma = oa.sigma_blocks(12),
cold start, s = 3 random attachments (stream [seed,0]), training bank S = 256 ([seed,1]).
Each converged network is analysed by hebb/sweeps2/readout.analyse unchanged: rank and
conditioning of the live-edge measurement operator on the 66-dim contrast space, recovery
cosines vs P C_hat P and P Sigma P, and the three errors separately (fixed point, sampling,
inverse problem on the identifiable subspace).

Statement being tested (R5): exact identifiability of the training-bank contrast second moment
P C_hat P, for stationary noiseless weights, known attachments and rule parameters, and a
full-rank, well-conditioned live-edge operator.

Run: uv run python -m hebb.big.readout_vs_n   -> site/data/big/readout_vs_N.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from hebb import onsite as o  # noqa: E402
from hebb import onsite_analysis as oa  # noqa: E402
from hebb.big import common as cm  # noqa: E402
from hebb.big import dist_engine as de  # noqa: E402
from hebb.sweeps2 import readout as ro  # noqa: E402

NS = (40, 80, 160, 320, 500)
BS = (0.01, 0.036, 0.1)
SEEDS = (0, 1, 2)
B = 12
KEEP = ("n_live", "dim", "rank_1e8", "rank_1e4", "sv_max", "sv_min_nonzero", "err_fixed_point", "err_sampling",
        "err_inverse_identifiable", "identifiable_frac", "K", "K_multi", "largest", "diam_max", "span_over_range")


def unit(args):
    N, seed, b = args
    Sigma = oa.sigma_blocks(B)
    G, _ = oa.banks(Sigma, seed)
    Chat = G.T @ G / len(G)
    net = oa.make_net(N, B, seed)
    r = de.lin_run(net.A, G, b, 1.0, Tmax=40000)
    net.W = r["W"]
    rr = dict(U=G @ r["H"].T, residual=r["residual"], converged=r["converged"])
    a = ro.analyse(net, rr, b, Sigma, Chat, "0", seed)
    out = dict(N=N, seed=seed, b=b, residual=r["residual"], converged=r["converged"], steps=r["steps"])
    out.update({k: a[k] for k in KEEP})
    out["condition"] = a["sv_max"] / a["sv_min_nonzero"] if a["sv_min_nonzero"] > 0 else None
    for m in ("ridge_psd", "hinge_psd", "base_identity", "base_diag", "base_untrained"):
        for t in ("Chat", "Sigma"):
            k = f"{m}_vs_{t}"
            if k in a:
                out[f"{m}_cos_{t}"] = a[k]["cos"]
    out["noise1e-4_cos"] = a.get("noise0.0001_ridge_psd_vs_Chat_cos")
    out["noise1e-2_cos"] = a.get("noise0.01_ridge_psd_vs_Chat_cos")
    print(f"N={N} seed={seed} b={b}: live={a['n_live']} rank={a['rank_1e8']}/{a['dim']} "
          f"cos={out['ridge_psd_cos_Chat']:.4f} cond={out['condition']} res={r['residual']:.1e}", flush=True)
    return out


def main():
    jobs = [(N, s, b) for N in NS for s in SEEDS for b in BS]
    rows = oa.pool_map(unit, jobs[::-1], procs=int(os.environ.get("BIG_PROCS", "3")))
    rows.sort(key=lambda r: (r["b"], r["N"], r["seed"]))
    summary = []
    for b in BS:
        for N in NS:
            g = [r for r in rows if r["b"] == b and r["N"] == N]
            f = lambda k: [float(np.mean([r[k] for r in g])), float(np.std([r[k] for r in g]))]  # noqa: E731
            summary.append(dict(b=b, N=N, n_live=f("n_live"), rank=f("rank_1e8"), rank_1e4=f("rank_1e4"),
                                cos_Chat=f("ridge_psd_cos_Chat"), cos_Sigma=f("ridge_psd_cos_Sigma"),
                                log10_condition=[float(np.mean([np.log10(r["condition"]) for r in g if r["condition"]] or [np.nan])),
                                                 float(np.std([np.log10(r["condition"]) for r in g if r["condition"]] or [np.nan]))],
                                err_fixed_point_max=float(max(r["err_fixed_point"] for r in g)),
                                err_sampling=f("err_sampling"), err_inverse=f("err_inverse_identifiable"),
                                K=f("K"), largest=f("largest"), all_converged=all(r["converged"] for r in g)))
    cm.write_json(cm.OUT / "readout_vs_N.json", dict(
        meta=dict(script="hebb/big/readout_vs_n.py", analysis="hebb/sweeps2/readout.analyse (unchanged)",
                  description="Readout of P C_hat P from the weights vs N at B = 12 (blocks), linear distance rule, "
                              "alpha = 1, 3 seeds x 3 b. summary[]: mean and sd over seeds.",
                  config=dict(B=B, Ns=NS, bs=BS, seeds=SEEDS, alpha=1.0, s=3, S_train=oa.S_TRAIN,
                              Sigma="oa.sigma_blocks(12): 4 blocks of 3, variances 2/1/0.5/0.25, within 0.8",
                              contrast_dim=66, res_tol=1e-6),
                  fields={"rank": "rank of the live-edge operator at relative singular-value tolerance 1e-8 (of 66)",
                          "rank_1e4": "rank at tolerance 1e-4",
                          "condition": "sv_max / smallest singular value above 1e-8 sv_max",
                          "cos_Chat / cos_Sigma": "Frobenius cosine of the PSD-projected min-norm ridge recovery with "
                                                  "P C_hat P (training bank, the exact target) / P Sigma P",
                          "err_fixed_point": "max over live edges |2(b - alpha w) - d^T C_hat d| (D^2 units)",
                          "err_sampling": "||P C_hat P - P Sigma P|| / ||P Sigma P||",
                          "err_inverse": "relative error of the recovery on the identifiable subspace (sv > 1e-4 sv_max)",
                          "noise1e-4_cos / noise1e-2_cos": "cosine vs P C_hat P with multiplicative weight noise of that size",
                          "base_*_cos_*": "baselines from hebb/sweeps2/readout (identity / diagonal fitted to the same "
                                          "measurements; untrained random network)"}),
        summary=summary, runs=rows))


if __name__ == "__main__":
    main()
