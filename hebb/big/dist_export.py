"""Analyse the distance-rule units (raw .npz/.json from dist_runs, local and Modal) and write the
site outputs: films for the film units, and the summaries dist_A.json and dist_B.json.

Per unit (all arms): components of the live graph (w > 1e-9) with diameters on the training
bank (oa.cluster_summary), the fixed-point check (oa.fixed_point_check), edge density, and
within-cluster diversity on the evaluation bank (hebb/sweeps2/diversity.cluster_measures:
proto_err, d_eff, numerical rank, per multi-member component; size-weighted means) plus the
per-neuron affine error from the full input (nonlinearity).
Linear arm, family B (alpha = 1): covariance readout, hebb/sweeps2/readout.analyse unchanged
(rank and conditioning of the live-edge operator, cosine to P C_hat P and P Sigma P, the three
error types). Rectifier arms: distance readout only (R1): weights-implied distances on live
edges vs measured (training and evaluation bank), dead-pair lower bounds, fraction of pairs
identified, shortest-path completion and its MDS vs the measured-distance MDS.
Matched comparisons (R2): same b, and matched edge density / matched largest component, read
off the seed-0 b grids by log-interpolation of the linear arm.

Run: uv run python -m hebb.big.dist_export   (needs scratch/out/big/dist and .../modal)
"""
import glob
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402
from scipy.spatial import procrustes  # noqa: E402

from hebb import onsite as o  # noqa: E402
from hebb import onsite_analysis as oa  # noqa: E402
from hebb.big import common as cm  # noqa: E402
from hebb.big import dist_runs as dr  # noqa: E402
from hebb.sweeps2 import diversity as dv  # noqa: E402
from hebb.sweeps2 import readout as ro  # noqa: E402

SCRIPT = "hebb/big/dist_export.py"
SRC_DIRS = [cm.RAW / "dist", cm.RAW / "modal"]
EPS = 1e-9


def load_all():
    recs = {}
    for d in SRC_DIRS:
        for p in glob.glob(str(d / "*.json")):
            if os.path.basename(p).startswith(("validate", "MANIFEST", "chunk")):
                continue
            r = json.load(open(p))
            if "unit" in r and os.path.exists(p[:-5] + ".npz"):
                r["_npz"] = p[:-5] + ".npz"
                recs[r["name"]] = r
    return recs


def cmds(D, dim=3):
    n = len(D)
    J = np.eye(n) - 1.0 / n
    Bm = -0.5 * J @ (D ** 2) @ J
    lam, V = np.linalg.eigh(Bm)
    lam, V = lam[::-1][:dim], V[:, ::-1][:, :dim]
    return V * np.sqrt(np.clip(lam, 0, None))


def distance_readout(W, U, Ue, b, alpha):
    """R1 readout for the rectifier: what the weights say about response distances."""
    N = len(W)
    iu = np.triu_indices(N, 1)
    Dt = np.sqrt(np.clip(o.dist2(U), 0, None))
    De = np.sqrt(np.clip(o.dist2(Ue.astype(float)), 0, None))
    live = W > EPS
    Lw = np.where(live, np.sqrt(np.clip(2 * (b - alpha * W), 0, None)), 0.0)
    np.fill_diagonal(Lw, 0)
    lab = o.components(W, EPS)
    same = lab[:, None] == lab[None]
    inner = same & ~live & ~np.eye(N, dtype=bool)
    cross = ~same
    out = dict(frac_exact=float(live[iu].mean()), frac_upper_bounded=float(inner[iu].mean()),
               frac_lower_only=float(cross[iu].mean()))
    if live.any():
        out["edge_abs_err_train_max"] = float(np.abs(Lw[live] - Dt[live]).max())
        rel = np.abs(Lw[live] - De[live]) / np.maximum(De[live], 1e-300)
        out["edge_rel_err_eval_median"] = float(np.median(rel))
        out["edge_rel_err_eval_max"] = float(rel.max())
    if cross.any():
        out["dead_lower_bound_holds_train"] = float((Dt[cross] >= np.sqrt(2 * b) - 1e-6).mean())
        out["dead_lower_bound_holds_eval"] = float((De[cross] >= np.sqrt(2 * b)).mean())
    if inner.any():
        Dsp = shortest_path(np.where(live, Lw, 0), method="D", directed=False)
        ratio = Dsp[inner] / np.maximum(De[inner], 1e-300)
        out["sp_over_eval_median"] = float(np.median(ratio))
        out["sp_over_eval_max"] = float(ratio.max())
        k = np.bincount(lab).argmax()
        idx = np.nonzero(lab == k)[0]
        if len(idx) >= 4:
            X1, X2 = cmds(Dsp[np.ix_(idx, idx)]), cmds(De[np.ix_(idx, idx)])
            if np.linalg.norm(X1) > 0 and np.linalg.norm(X2) > 0:
                out["mds_completed_vs_measured_eval_disparity"] = float(procrustes(X2, X1)[2])
                out["mds_component_size"] = int(len(idx))
    return out


COMPACT_SPAN = 2.0   # a component is "compact" if its span (largest pairwise D) is <= 2 edge ranges sqrt(2b)


def diversity(Ue, Ge, lab, D=None, b=None):
    """Per-neuron nonlinearity and within-component diversity (evaluation bank). With D (training-bank
    distances) and b, also the size-weighted proto_err over compact components only (span <= 2 sqrt(2b)):
    in a chain proto_err measures the chain's width, not diversity."""
    h = len(Ue) // 2
    aff = oa.affine_fit_err(Ue[:h], Ge[:h], Ue[h:], Ge[h:])
    cl = dv.cluster_measures(Ue, lab)
    if D is not None:
        multi = [k for k in range(lab.max() + 1) if (lab == k).sum() >= 2]   # same order as cluster_measures
        for c, k in zip(cl, multi):
            idx = np.nonzero(lab == k)[0]
            c["span_over_range"] = float(D[np.ix_(idx, idx)].max() / np.sqrt(2 * b))
    n_in = sum(c["size"] for c in cl)
    w = (lambda key: float(sum(c[key] * c["size"] for c in cl) / n_in)) if cl else (lambda key: float("nan"))
    top = sorted(cl, key=lambda c: -c["size"])[:10]
    out = dict(affine_err_mean=float(aff.mean()), affine_err_max=float(aff.max()), frac_in_multi=n_in / len(lab),
               proto_err_w=w("proto_err"), deff_w=w("deff"), rank_w=w("rank"), deff_resid_w=w("deff_resid"),
               proto_err_max=max([c["proto_err"] for c in cl] + [0.0]), top_clusters=top)
    if D is not None:
        cc = [c for c in cl if c["span_over_range"] <= COMPACT_SPAN]
        n_c = sum(c["size"] for c in cc)
        out["proto_err_w_compact"] = float(sum(c["proto_err"] * c["size"] for c in cc) / n_c) if cc else None
        out["frac_in_compact"] = n_c / len(lab)
        out["n_compact"] = len(cc)
        out["n_multi_noncompact"] = len(cl) - len(cc)
    return out


def analyse(rec):
    u = rec["unit"]
    z = np.load(rec["_npz"])
    W, U, Ue = z["W"], z["U"], z["Ue"].astype(float)
    b, alpha = u["b"], u["alpha"]
    Sigma = dr.sigma_of(u["sigma"])
    G, Ge = oa.banks(Sigma, u["seed"])
    N = len(W)
    cs = oa.cluster_summary(W, U, b)
    fp = oa.fixed_point_check(W, U, b, alpha)
    lab = cm.components_sorted(W, EPS)
    sizes = np.bincount(lab)
    D = np.sqrt(np.clip(o.dist2(U), 0, None))
    comps = []
    for k in range(min(len(sizes), 30)):
        idx = np.nonzero(lab == k)[0]
        proto = U[:, idx].mean(1)
        dd = np.sqrt(((U[:, idx] - proto[:, None]) ** 2).mean(0))
        c = dict(size=int(len(idx)), diam_max=float(dd.max()), diam_mean=float(dd.mean()),
                 span=float(D[np.ix_(idx, idx)].max()))
        if u["arm"] == "linear":
            c["centroid"] = z["H"][idx].mean(0)
        comps.append(c)
    out = dict(name=rec["name"], fam=u["fam"], sigma=u["sigma"], N=N, arm=u["arm"], b=b, alpha=alpha, seed=u["seed"],
               converged=rec["converged"], residual=rec["residual"], steps=rec["steps"], t=rec["t"],
               seconds=rec["total_seconds"], device=rec.get("device"), equil_gmax=rec.get("equil_gmax"),
               K=cs["K"], K_multi=cs["K_multi"], largest=cs["largest"], largest_frac=cs["largest"] / N,
               n_live=fp["n_live"], edge_density=fp["n_live"] / (N * (N - 1) / 2),
               diam_max=cs["diam_max"], diam_mean=cs["diam_mean"], span_over_range=cs["span_over_range"],
               fp_dev_live=fp["dev_live"], fp_slack_dead=fp["slack_dead"], components=comps)
    out["span_largest_over_range"] = comps[0]["span"] / np.sqrt(2 * b)   # largest component (span_over_range: max over components)
    out["diversity"] = diversity(Ue, Ge, cs["lab"], D, b)
    if u["arm"] == "linear" and u["fam"] == "B" and alpha == 1.0:
        net = o.Net(A=z["A"], t=0.0, klo=0.0, khi=0.0, W=W)
        Chat = G.T @ G / len(G)
        r = dict(U=U, residual=rec["residual"], converged=rec["converged"])
        ra = ro.analyse(net, r, b, Sigma, Chat, "0", u["seed"])
        keep = ("n_live", "dim", "rank_1e8", "rank_1e4", "sv_max", "sv_min_nonzero", "err_fixed_point", "err_sampling",
                "err_inverse_identifiable", "identifiable_frac", "identifiable_frac_1e8", "ridge_psd_vs_Chat",
                "ridge_psd_vs_Sigma", "hinge_psd_vs_Chat", "hinge_psd_vs_Sigma", "base_identity_vs_Chat",
                "base_diag_vs_Chat", "base_shuffled_vs_Chat", "base_untrained_vs_Chat", "noise0.01_ridge_psd_vs_Chat_cos",
                "noise0.0001_ridge_psd_vs_Chat_cos")
        out["readout"] = {k: ra[k] for k in keep if k in ra}
        sv_min = ra["sv_min_nonzero"]
        out["readout"]["condition"] = ra["sv_max"] / sv_min if sv_min > 0 else None
    if u["arm"] != "linear":
        out["distance_readout"] = distance_readout(W, U, Ue, b, alpha)
    return out


# ----------------------------------------------------------------------------- matched comparisons
def matched(rows, fam, sigma):
    grid = {arm: sorted([r for r in rows if r["fam"] == fam and r["sigma"] == sigma and r["arm"] == arm
                         and r["seed"] == 0 and r["alpha"] == 1.0 and r["b"] in set(dr.GRID[fam].tolist())],
                        key=lambda r: r["b"]) for arm in dr.ARMS}
    lin = grid["linear"]
    out = dict(grid={arm: [dict(b=r["b"], K=r["K"], K_multi=r["K_multi"], largest_frac=r["largest_frac"],
                                edge_density=r["edge_density"], diam_max=r["diam_max"], span_over_range=r["span_over_range"],
                                span_largest_over_range=r["span_largest_over_range"],
                                proto_err_w=r["diversity"]["proto_err_w"], deff_w=r["diversity"]["deff_w"],
                                proto_err_w_compact=r["diversity"]["proto_err_w_compact"],
                                affine_err_mean=r["diversity"]["affine_err_mean"], converged=r["converged"])
                           for r in g] for arm, g in grid.items()})
    lb = np.log([r["b"] for r in lin])

    def b_at(key, val):
        ys = np.array([r[key] for r in lin], float)
        ok = ys > 0
        if not ok.any() or val <= 0:
            return None
        ly = np.log(np.maximum.accumulate(ys[ok]) + 1e-12 * np.arange(ok.sum()))   # monotone envelope
        return float(np.exp(np.interp(np.log(val), ly, lb[ok]))) if ly[0] <= np.log(val) <= ly[-1] else None

    pairs = []
    keys = ("K", "largest_frac", "edge_density", "diam_max", "span_over_range")
    dkeys = ("proto_err_w", "deff_w", "affine_err_mean")
    for arm in ("rect", "rect_homog"):
        for r in grid[arm]:
            p = dict(arm=arm, b=r["b"], edge_density=r["edge_density"], largest_frac=r["largest_frac"],
                     linear_b_same_density=b_at("edge_density", r["edge_density"]),
                     linear_b_same_largest=b_at("largest_frac", r["largest_frac"]))
            if r["edge_density"] > 0:
                # nearest linear grid point in log edge density (a matched-density pair from the grid)
                j = int(np.argmin([abs(np.log(max(q["edge_density"], 1e-12)) - np.log(r["edge_density"])) for q in lin]))
                q = lin[j]
                p["nearest_linear_by_density"] = dict(b=q["b"], **{k: q[k] for k in keys},
                                                      **{k: q["diversity"][k] for k in dkeys})
                p["this_arm"] = dict(**{k: r[k] for k in keys}, **{k: r["diversity"][k] for k in dkeys})
            pairs.append(p)
    out["matched"] = pairs
    # the regime runs themselves (seed 0, alpha = 1): the linear b with the same edge density, interpolated on
    # the linear grid at this run's own density (not at the nearest grid point)
    reg = []
    for regime, bb in dr.REGIMES[(fam, sigma)].items():
        for arm in ("rect", "rect_homog"):
            r = next((r for r in rows if r["fam"] == fam and r["sigma"] == sigma and r["arm"] == arm and r["seed"] == 0
                      and r["alpha"] == 1.0 and abs(r["b"] - bb) < 1e-12), None)
            if r is None:
                continue
            lb_ = b_at("edge_density", r["edge_density"]) if r["edge_density"] > 0 else None
            reg.append(dict(regime=regime, arm=arm, b=bb, edge_density=r["edge_density"], linear_b_same_density=lb_,
                            ratio=bb / lb_ if lb_ else None))
    out["regime_match"] = reg
    return out


# ----------------------------------------------------------------------------- films
def export_film(rec, row):
    u = rec["unit"]
    z = np.load(rec["_npz"])
    arm = u["arm"]
    fam = u["fam"]
    regime = next((k for k, v in dr.REGIMES[(fam, u["sigma"])].items() if v == u["b"]), "b")
    fid = f"{fam}_dist_{u['sigma']}_{'lin' if arm == 'linear' else arm}_{regime}" + ("" if u["alpha"] == 1 else f"_a{u['alpha']:g}")
    nice = dict(linear="linear", rect="rectifier", rect_homog="homogeneous rectifier")[arm]
    short = dict(ind3="3 independent inputs", blocks="12 inputs in 4 blocks", power="12 inputs, power-law spectrum")
    title = (f"Distance rule, {nice}, {short[u['sigma']]}, b = {u['b']:g} ({regime}), "
             f"alpha = {u['alpha']:g} (N = {u['N']})")
    counts = z["film_count"]
    offs = np.concatenate([[0], np.cumsum(counts)])
    edges = [z["film_edges"][offs[f]:offs[f + 1]] for f in range(len(counts))]
    ws = [z["film_w"][offs[f]:offs[f + 1]] for f in range(len(counts))]
    B = z["A"].shape[1]
    space = rec["space"]
    lab = cm.components_sorted(z["W"], EPS)
    pot = dr.ARMS[arm]
    cfg = dict(sigma=u["sigma"], Sigma_desc=dr.SIGMA_DESC[u["sigma"]], Sigma=dr.sigma_of(u["sigma"]), N=u["N"], B=B,
               b=u["b"], alpha=u["alpha"], regime=regime, dt=rec["dt"], s=3, deg=1.0, S_train=oa.S_TRAIN,
               S_eval=oa.S_EVAL, klo=pot["klo"], khi=pot["khi"], sigma_t=pot["sigma_t"], W0="zero (cold start)",
               run_tol=1e-9, res_tol=1e-6,
               seed_streams="attachments [seed,0], training bank [seed,1], evaluation bank [seed,2], thresholds [seed,3]",
               matched_protocol="arms share seed, attachments, initial W, banks and b (R2)")
    meta = dict(id=fid, title=title, family=fam, rule="dist", nonlin=dict(linear="linear", rect="rect", rect_homog="rect_homog")[arm],
                B=B, N=u["N"], seed=u["seed"], config=cfg, script="hebb/big/dist_runs.py (run), hebb/big/dist_export.py (export)",
                space=space,
                space_label="input space (harmonic embedding)" if space == "simplex" else "response-function space (PCA)",
                pca=None if space == "simplex" else dict(
                    explained_var=rec.get("pca_explained_var"), total_var=rec.get("pca_total_var"),
                    features="x_i = F_i / sqrt(S) on the training bank (S = 256): Euclidean distance = RMS response "
                             "distance D_ij; axes from the final frame; equal to classical MDS of the measured D"),
                frames=rec["frames"], edge_threshold="w > 1e-9 (live edge)",
                final=dict(labels=lab.tolist(), clusters=row["components"],
                           metrics={k: row[k] for k in ("K", "K_multi", "largest", "n_live", "edge_density", "diam_max",
                                                        "span_over_range", "converged", "residual", "steps", "t",
                                                        "fp_dev_live", "fp_slack_dead")}
                           | dict(diversity={k: v for k, v in row["diversity"].items() if k != "top_clusters"})),
                network_layout_seed=1000 + u["seed"])
    H = z["film_H"].astype(float) if "film_H" in z else None
    return fid, cm.write_film(meta, z["film_coords"].astype(float), H=H, edges=edges, edge_w=ws, in_w=z["A"][None],
                              index_entry=dict(title=title, family=fam, rule="dist", nonlin=meta["nonlin"], B=B, N=u["N"]))


def main():
    recs = load_all()
    print(len(recs), "units")
    import multiprocessing as mp
    names = sorted(recs)
    with mp.get_context("fork").Pool(int(os.environ.get("BIG_PROCS", "3"))) as p:
        rows = p.map(analyse, [recs[n] for n in names], chunksize=1)
    byname = dict(zip(names, rows))
    films = {}
    for n in names:
        if recs[n]["unit"]["film"]:
            fid, nbytes = export_film(recs[n], byname[n])
            films[n] = dict(id=fid, bytes=nbytes)
    expected = {dr.unit_name(u) for u in dr.units("all")}
    missing = sorted(expected - set(names))
    for fam, sigmas in (("A", ("ind3",)), ("B", ("blocks", "power"))):
        fr = [r for r in rows if r["fam"] == fam]
        cm.write_json(cm.OUT / f"dist_{fam}.json", dict(
            meta=dict(script=SCRIPT, run_script="hebb/big/dist_runs.py", engine="hebb/big/dist_engine.py (validated "
                      "against hebb/onsite.py by hebb/big/validate.py)", grid=dr.GRID[fam], regimes={s: dr.REGIMES[(fam, s)] for s in sigmas},
                      arms=dr.ARMS, sigma_desc={s: dr.SIGMA_DESC[s] for s in sigmas}, res_tol=1e-6,
                      missing_units=[m for m in missing if m.startswith(fam)],
                      description="Distance rule at N >> B, matched linear / rectifier / homogeneous-rectifier arms. "
                                  "See site/data/big/README.md for every field."),
            films={n: f for n, f in films.items() if n.startswith(fam)},
            matched={s: matched(fr, fam, s) for s in sigmas},
            runs=fr), sig=5)


if __name__ == "__main__":
    main()
