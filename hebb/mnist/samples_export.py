"""Export item 6: moment-based sample synthesis (see hebb/mnist/sampling.py for the
generative assumptions).  One PNG sheet per run, rows = generators, same seeds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hebb.mnist import sampling as SM

N_SHOW = 16
N_METRIC_PIX = 1000
N_METRIC_RFF = 400


def _sheet(rows_x: list[np.ndarray], site: Path, path: Path):
    from hebb.mnist.export import sprite, save_png
    tiles = np.concatenate([np.clip(r[:N_SHOW], 0, 1).reshape(-1, 14, 14) for r in rows_x], 0)
    img, lay = sprite(tiles, N_SHOW, "gray_r", 0, 1, upscale=3)
    return save_png(path, img), lay


def export_samples(names, found, ctxs, x_tr, x_te, x28_te, site: Path):
    from hebb.mnist.export import lift_key, rnd
    out = []
    rng = np.random.default_rng(0)
    te_idx = rng.permutation(len(x_te))
    x_show = x_te[te_idx[:N_SHOW]]
    x_ref = x_te[te_idx[N_SHOW:N_SHOW + 1000]]            # held-out digits for MMD
    x_trs = x_tr[np.random.default_rng(1).choice(len(x_tr), 1000, replace=False)]
    sig_mmd = float(np.median(np.linalg.norm(x_ref[:500, None] - x_ref[None, :500], axis=-1)))
    mmd_real = SM.mmd2(x_trs, x_ref, sig_mmd)
    for name in names:
        rd = found[name]
        r = json.loads((rd / f"{name}.json").read_text()); cfg = r["cfg"]
        z = np.load(rd / f"{name}.npz")
        ctx = ctxs[lift_key(cfg)]
        lift = ctx.lift
        Sig = ctx.Sigma
        C_hat = z["C_ls_both"].astype(np.float64)
        C_init = z["C_init_ls_both"].astype(np.float64)
        gens = [("true-moment generator", "Sigma_g (exact lifted covariance of the training set)", Sig),
                ("weight-reconstructed generator", "C_hat: masked ridge LS readout (I-I + I-B), parameters known", C_hat),
                ("baseline generator: diag", "diag(Sigma_g)", np.diag(np.diag(Sig))),
                ("baseline generator: untrained net", "LS readout applied to the untrained (init) weights; equilibrium assumption violated", C_init)]
        B = cfg["B"]
        rows_x, rows_meta = [x_show], [dict(label="real data", detail="held-out MNIST test digits (14x14)")]
        if cfg["lift"] == "pixels":
            n = N_METRIC_PIX
            Z = np.random.default_rng(123).standard_normal((n, B))
            s = lift.scale
            for label, detail, C in gens:
                Cx = C / s ** 2
                X, neg, Cpsd = SM.gaussian_samples(ctx.xbar.astype(np.float64), Cx, Z)
                Xc = np.clip(X, 0, 1)
                Sx = Sig / s ** 2
                meta = dict(label=label, detail=detail, psd_negative_mass_clipped=neg,
                            mmd2_to_heldout=SM.mmd2(Xc, x_ref, sig_mmd),
                            moment_relerr_vs_target_unclipped=SM.rel_fro(np.cov(X.T), Cpsd),
                            moment_relerr_vs_true_unclipped=SM.rel_fro(np.cov(X.T), Sx),
                            moment_relerr_vs_true_clipped=SM.rel_fro(np.cov(Xc.T), Sx),
                            target_trace_over_true_trace=float(np.trace(Cpsd) / np.trace(Sx)))
                rows_x.append(X); rows_meta.append(meta)
            gen = dict(kind="gaussian-pixels", mean="training-data mean (NOT in the weights; outside information)",
                       scale="C_hat read out with theta and lambda/eta known (absolute units), divided by lift.scale^2 to return to pixel units",
                       completion="ridge minimum-norm: unconstrained directions get ~0 variance",
                       psd_repair="negative eigenvalues clipped to 0 (mass reported per row)",
                       display="samples clipped to [0,1] for display and for MMD; moment checks also reported unclipped",
                       n_metric=n, seed=123)
        else:
            n = N_METRIC_RFF
            Z = np.random.default_rng(123).standard_normal((n, B))
            steps, lr = 300, 0.02
            # round-trip reference: pre-image of real digits' features
            Xr, resr = SM.preimage(lift, lift.raw(x_show).astype(np.float32), ctx.xbar, steps, lr)
            rows_x.append(Xr)
            rows_meta.append(dict(label="real data, pre-image round trip", detail="phi(x) of the real digits above, inverted by the same optimiser",
                                  rel_residual_median=float(np.median(resr / np.linalg.norm(lift.raw(x_show) - ctx.mu, axis=1)))))
            for label, detail, C in gens:
                G, neg, Cpsd = SM.gaussian_samples(ctx.mu, C, Z)
                X, res = SM.preimage(lift, G.astype(np.float32), ctx.xbar, steps, lr)
                gnorm = np.linalg.norm(G - ctx.mu, axis=1)
                Gx = lift.raw(X).astype(np.float64)
                meta = dict(label=label, detail=detail, psd_negative_mass_clipped=neg,
                            mmd2_to_heldout=SM.mmd2(X, x_ref, sig_mmd),
                            preimage_rel_residual_median=float(np.median(res / np.maximum(gnorm, 1e-12))),
                            moment_relerr_target_samples_vs_target=SM.rel_fro(np.cov(G.T), Cpsd),
                            moment_relerr_preimage_features_vs_target=SM.rel_fro(np.cov(Gx.T), Cpsd),
                            moment_relerr_preimage_features_vs_true=SM.rel_fro(np.cov(Gx.T), Sig))
                rows_x.append(X); rows_meta.append(meta)
            gen = dict(kind="rff-gaussian-preimage", mean="mu_g = training mean of the features (outside information)",
                       preimage=f"projected Adam on ||phi(x) - g||^2, x in [0,1]^196, init x0 = training-mean image, {steps} steps, lr {lr}, identical for all rows",
                       scale="C read out with parameters known (absolute lifted units)",
                       psd_repair="negative eigenvalues clipped to 0 (mass reported per row)",
                       caveat="Gaussian feature vectors are generally off the image of phi; the optimiser, init and box constraint shape the result (exploratory)",
                       n_metric=n, seed=123)
        png, lay = _sheet(rows_x, site, site / "samples" / f"{name}.png")
        out.append(dict(run=name, png=png, layout=lay, rows=rnd(rows_meta), generator=gen,
                        mmd=dict(kernel="gaussian on 14x14 pixel vectors", bandwidth=sig_mmd,
                                 reference_real_train_vs_heldout=mmd_real,
                                 note="unbiased squared MMD between generated samples and 1000 held-out test digits; lower is better")))
        print("samples", name, [round(m.get("mmd2_to_heldout", float("nan")), 4) for m in rows_meta[1:]], flush=True)
    return out
