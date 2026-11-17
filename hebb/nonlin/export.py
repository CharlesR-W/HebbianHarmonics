"""Export the rectified-output experiments to site/data/nonlin/ (format: site/data/nonlin/README.md).

    PYTHONPATH=. python -m hebb.nonlin.export /var/tmp/hebb-nonlin/runs/<main14 id> [/var/tmp/hebb-nonlin/runs/<main28 id>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from hebb.nonlin import dual
from hebb.nonlin.model import Data, thresholds

SITE = Path(__file__).resolve().parents[2] / "site" / "data" / "nonlin"
SNAP_PICK = 10
N_SHOW = 256
FEATURED = {  # page key -> (experiment key, q).  q = 0.98 is the sibling's theta = 0.6 for the linear rule.
    "A_lin": ("A_lin", 0.98), "A_rect_t0": ("A_rect_t0", 0.98), "A_rect_t1": ("A_rect_t1", 0.98),
    "Bown_rect_t0": ("Bown_rect_t0", 0.98), "B_rect_t0_q0.946": ("B_rect_t0", 0.946), "A_rect_t0_q0.946": ("A_rect_t0", 0.946)}


# ---------------------------------------------------------------- helpers
def wbin(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(arr).tofile(path)
    return dict(file=str(path.relative_to(SITE)), dtype=str(arr.dtype), shape=list(arr.shape))


def _jdef(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def wjson(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), default=_jdef))
    return str(path.relative_to(SITE))


def rnd(x, n=4):
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{x:.{n}g}") if np.isfinite(x) else None
    return x


def sprite(tiles, ncol, cmap, vmin, vmax, upscale=2, pad=1):
    """tiles (n, h, w) -> RGB uint8 sprite, white gutters (same layout as site/data/mnist)."""
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    n, h, w = tiles.shape
    nrow = int(np.ceil(n / ncol))
    th, tw = h * upscale, w * upscale
    img = np.full((nrow * (th + pad) + pad, ncol * (tw + pad) + pad, 3), 255, np.uint8)
    for i in range(n):
        lo = float(vmin[i] if np.ndim(vmin) else vmin); hi = float(vmax[i] if np.ndim(vmax) else vmax)
        t = np.clip((tiles[i] - lo) / max(hi - lo, 1e-12), 0, 1)
        rgb = (cm(t)[..., :3] * 255).astype(np.uint8).repeat(upscale, 0).repeat(upscale, 1)
        r, c = divmod(i, ncol)
        y, x = pad + r * (th + pad), pad + c * (tw + pad)
        img[y:y + th, x:x + tw] = rgb
    return img, dict(tile_w=tw, tile_h=th, pad=pad, ncol=ncol, nrow=nrow, n=n, cmap=cmap)


def save_png(path: Path, img):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, optimize=True)
    return str(path.relative_to(SITE))


def umap2(H, seed=0):
    """Hellinger UMAP of probability vectors (same metric/settings as site/data/mnist)."""
    import umap
    import warnings
    warnings.filterwarnings("ignore")
    H = np.clip(H.astype(np.float64), 0, None); H = H / H.sum(1, keepdims=True)
    H = H + np.random.default_rng(seed).normal(0, 1e-9, H.shape).clip(0)
    Y = umap.UMAP(n_neighbors=15, min_dist=0.15, metric="hellinger", random_state=seed, n_epochs=200).fit_transform(H)
    return (Y - Y.mean(0)).astype(np.float32)


def slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    b = np.polyfit(x, y, 1)[0]
    return float(b)


# ---------------------------------------------------------------- per-run structure
def structure(r, z):
    """Dead pool, live clusters, allocation, diversity (computed from the saved arrays)."""
    cfg = r["cfg"]; N = cfg["N"]; wmin = cfg["wmin"]
    Wii = z["Wii_avg"].astype(np.float32); Wib = z["Wib_avg"].astype(np.float32)
    np.fill_diagonal(Wii, 0)
    dead = ((Wii <= 2 * wmin).sum(1) == N) & ((Wib <= 2 * wmin).all(1))   # diag counts as <= 2wmin
    lab = z["labels"]; H = z["H_final"]; cent = z["cent"]
    K = int(lab.max() + 1)
    sizes = np.bincount(lab, minlength=K)
    dead_frac = np.bincount(lab, weights=dead.astype(float), minlength=K) / np.maximum(sizes, 1)
    dead_cl = dead_frac > 0.5
    live = ~dead_cl
    major = live & (sizes >= max(2, 0.01 * N))
    Ev = np.asarray(r["cluster_energy_volt"]); Er = np.asarray(r["cluster_energy_rate"])
    # within-cluster diversity among live neurons in live clusters of size >= 2
    d = np.abs(H - cent[lab]).sum(1)
    m = live[lab] & (sizes[lab] >= 2) & ~dead
    l1 = float(d[m].mean()) if m.any() else None
    idx = np.where(major)[0]
    alloc = dict(sizes=sizes[idx].tolist(), energy_volt=Ev[idx].tolist(), energy_rate=Er[idx].tolist(),
                 part_width=(1.0 / (cent[idx] ** 2).sum(1)).tolist(),
                 beta_volt=slope(Ev[idx], sizes[idx]) if len(idx) >= 3 else None,
                 beta_rate=slope(Er[idx], sizes[idx]) if len(idx) >= 3 else None)
    return dict(n_dead=int(dead.sum()), n_live_clusters=int(live.sum()), n_live_major=int(major.sum()),
                n_singletons_live=int((live & (sizes == 1)).sum()), within_l1_live=l1,
                live_neurons=int((~dead).sum())), alloc, dict(dead=dead, dead_cl=dead_cl, sizes=sizes, major=major)


PROBE_KEYS = ["pixels", "trained_volt_all", "trained_volt_sub", "trained_rate_t0_all", "trained_rate_t0_sub",
              "trained_rate_t0.5_all", "trained_rate_t0.5_sub", "trained_rate_t1_all", "trained_rate_t1_sub",
              "init_volt_all", "init_rate_t0_all", "init_rate_t0_sub", "init_rate_t1_all", "init_rate_t1_sub",
              "randrelu_t0_all", "randrelu_t0_sub", "randrelu_t1_all", "randrelu_t1_sub"]
RO_KEYS = ["known_relfro", "known_bw", "scale_cos", "scale_bw", "contrast_cos", "contrast_bw", "overlap_k5",
           "overlap_k10", "overlap_kclust", "neg_mass_full", "neg_mass_contrast"]


def run_info(r, struct, alloc):
    cfg = r["cfg"]; ex = cfg.get("extra", {})
    info = dict(name=r["name"], exp=ex.get("exp"), q=ex.get("q"), rule=cfg["rule"], tau=cfg["tau"] if cfg["rule"] == "rect" else None,
                input=cfg["input"], res=cfg["res"], B=cfg["B"], N=cfg["N"], theta=cfg["theta"], c=r["c"], seed=cfg["seed"],
                wmin=cfg["wmin"], gamma=cfg["gamma"], steps=cfg["steps"], dt=cfg["dt"], batch=cfg["batch"])
    f = r["final"]
    info["final"] = rnd({k: f[k] for k in ("n_clusters", "n_clusters_tol05", "n_clusters_tol20", "n_major", "n_singletons",
                                           "n_active_components", "within_l1", "eff_rank", "floor_ii", "floor_ib", "res_max",
                                           "res_mean", "res_active_mean", "res_rel", "res_max_strict", "w_max",
                                           "averaged_over_steps") if k in f})
    info["final"].update(rnd(struct))
    info["allocation"] = rnd(alloc)
    info["rule_cov_quantiles"] = rnd(r["final_cov_quantiles"])
    g = r["gap"]
    info["gap"] = rnd({fam: {k: v for k, v in g[fam].items() if k not in ("sample", "emp_quantiles")} for fam in ("ii", "ib", "bb")})
    for fam in ("ii", "ib", "bb"):
        info["gap"][fam]["emp_quantiles"] = rnd(g[fam]["emp_quantiles"])
    info["gap"]["var_ratio_interior_q10_50_90"] = rnd(g["var_ratio_interior_quantiles"])
    info["gap"]["var_ratio_pixel_q10_50_90"] = rnd(g["var_ratio_pixel_quantiles"])
    gi = r["gap_init"]
    info["gap_init"] = rnd({fam: {k: gi[fam][k] for k in ("rel_fro_gap", "corr_emp_pred", "rate_corr_rel_gap")} for fam in ("ii", "ib")})
    if "readout" in r:
        info["readout"] = {k: dict(**rnd({m: v["score"].get(m) for m in RO_KEYS}),
                                   **({"info": rnd({kk: vv for kk, vv in v["info"].items() if not isinstance(vv, (list, dict))})} if "info" in v else {}))
                           for k, v in r["readout"].items()}
        info["readout_baselines"] = {k: rnd({m: v.get(m) for m in RO_KEYS}) for k, v in r["readout_baselines"].items()}
    if "probes" in r:
        info["probes"] = {k: rnd(r["probes"][k]["test"]) for k in PROBE_KEYS if k in r["probes"]}
        info["probes_frac_active"] = {k: rnd(v["frac_active"]) for k, v in r["probes"].items() if "frac_active" in v}
    info["timing"] = rnd(r["timing"])
    return info


# ---------------------------------------------------------------- featured runs
def featured(r, z, D: Data, st, out: Path):
    cfg = r["cfg"]; B = cfg["B"]; side = cfg["res"]
    tau = cfg["tau"] if cfg["rule"] == "rect" else 0.0
    c = r["c"] if cfg["rule"] == "rect" else D.pixel_rate_gain(0.0)
    files = {}
    sub = z["sub"]; Hsub = z["H_sub"].astype(np.float32)               # (S, n, B)
    S = Hsub.shape[0]
    pick = np.unique(np.round(np.linspace(0, S - 1, SNAP_PICK)).astype(int))
    steps = [r["snaps"][i] for i in pick]
    lab = z["labels"]; sizes = st["sizes"]; dead = st["dead"]
    rank = np.empty(len(sizes), int); order_cl = np.lexsort((-sizes, st["dead_cl"])); rank[order_cl] = np.arange(len(sizes))
    ls = lab[sub]
    Hfin = z["H_final"]; cent = z["cent"]
    dist = np.abs(Hfin[sub] - cent[ls]).sum(1)
    order = np.lexsort((dist, rank[ls]))
    ids = sub[order]
    Hs = Hsub[pick][:, order]                                          # (S', n, B)
    Sig = torch.as_tensor(D.Sigma)
    sims_v, sims_r = [], []
    for s in range(len(pick)):
        Ht = torch.as_tensor(Hs[s], dtype=torch.float64)
        sims_v.append((Ht @ Sig @ Ht.T).numpy())
        bi, _ = thresholds(Ht.float(), D, tau)
        R = torch.relu(D.Gtr @ Ht.float().T - bi).double()
        sims_r.append((c * c * torch.cov(R.T, correction=0)).numpy())
    for nm, sims in (("similarity_volt", sims_v), ("similarity_rate", sims_r)):
        A = np.stack(sims); a = float(np.abs(A).max())
        q = np.clip(np.round((A + a) / (2 * a) * 255), 0, 255).astype(np.uint8)
        files[nm] = dict(**wbin(out / f"{nm}.u8", q), a=a)
    files["similarity_volt"]["note"] = "H Sigma_g H^T (voltage covariance) of the shown neurons; value = -a + 2a*byte/255"
    files["similarity_rate"]["note"] = (f"rate covariance c^2 Cov(relu(u_i - b_i), relu(u_j - b_j)) over the 60k training digits, tau = {tau:g}, "
                                        f"c = {c:.4g}" + (" (post hoc: the linear rule does not use rates)" if cfg["rule"] == "linear" else ""))
    # cluster parts over snapshots (final clusters with members among the shown neurons)
    shown_cl = [k for k in order_cl if (not st["dead_cl"][k]) and (ls == k).any()][:10]
    tiles = []
    for k in shown_cl:
        m = ls[order] == k
        for s in range(len(pick)):
            tiles.append(Hs[s][m].mean(0).reshape(side, side))
    if tiles:
        T = np.stack(tiles); mx = T.reshape(len(T), -1).max(1)
        img, lay = sprite(T, len(pick), "magma_r", 0, mx, upscale=3 if side == 14 else 1)
        files["parts_over_time"] = dict(png=save_png(out / "parts_over_time.png", img), layout=lay,
                                        clusters=[int(k) for k in shown_cl], cluster_rank=[int(rank[k]) for k in shown_cl],
                                        steps=steps, note="rows = final live clusters (largest first, members among the shown neurons), "
                                        "cols = snapshots; tile = mean h_i of those members at that snapshot as an image, own max")
    # 16 sample neurons over snapshots, stratified over clusters
    rng = np.random.default_rng(0)
    cand = []
    for k in shown_cl:
        pos = np.where(ls[order] == k)[0]
        cand += list(rng.choice(pos, min(len(pos), 2), replace=False))
    rest = [p for p in range(len(order)) if p not in cand]
    cand += list(rng.choice(rest, max(0, 16 - len(cand)), replace=False)) if rest else []
    cand = sorted(cand[:16])
    T = np.stack([Hs[s][p].reshape(side, side) for p in cand for s in range(len(pick))])
    mx = T.reshape(len(T), -1).max(1)
    img, lay = sprite(T, len(pick), "magma_r", 0, mx, upscale=3 if side == 14 else 1)
    files["neurons_over_time"] = dict(png=save_png(out / "neurons_over_time.png", img), layout=lay,
                                      positions=[int(p) for p in cand], neuron_ids=[int(ids[p]) for p in cand], steps=steps,
                                      note="rows = 16 shown neurons (positions in display order), cols = snapshots, h_i as an image, own max")
    # UMAP of all final embeddings + labels + preferred digit (on the rule's own signal)
    Y = umap2(Hfin)
    files["umap"] = dict(**wbin(out / "umap.f32", Y), note="(N, 2) Hellinger-UMAP of all N final embeddings (neuron id order)")
    cr = np.where(dead, 255, np.minimum(rank[lab], 250)).astype(np.uint8)
    files["umap_cluster_rank"] = dict(**wbin(out / "umap_cluster_rank.u8", cr), note="cluster rank by size among live clusters (0 = largest; 250 = rank >= 250), 255 = dead")
    Ht = torch.as_tensor(Hfin).float()
    U = D.Gte @ Ht.T
    if cfg["rule"] == "rect":
        bi, _ = thresholds(Ht, D, tau); U = torch.relu(U - bi)
    U = (U - U.mean(0)) / U.std(0).clamp_min(1e-9)
    yte = torch.as_tensor(D.yte)
    tun = torch.stack([U[yte == d].mean(0) for d in range(10)], 1).numpy()   # (N, 10)
    files["umap_pref_digit"] = dict(**wbin(out / "umap_pref_digit.u8", tun.argmax(1).astype(np.uint8)),
                                    note="argmax over digit classes of the mean z-scored test response (rates for rect, voltages for linear)")
    # readout matrices
    for k in ("C_gauss_both", "C_chord_both", "C_gauss_expected_cov", "C_ls_both"):
        if k in z.files:
            files[f"chat_{k[2:]}"] = wbin(out / f"chat_{k[2:]}.f32", z[k].astype(np.float32))
    # curves
    keys = ["step", "time", "n_clusters", "n_major", "n_singletons", "n_active_components", "within_l1", "eff_rank",
            "floor_ii", "floor_ib", "res_max", "res_mean", "res_active_mean", "res_rel", "res_max_strict",
            "cov_q10", "cov_q50", "cov_q90", "cov_q99"]
    curves = {k: [rnd(row.get(k)) for row in r["snap_rows"]] for k in keys}
    curves["dense_cols"] = r["dense_cols"]; curves["dense"] = rnd(z["dense"].tolist())
    files["curves"] = wjson(out / "curves.json", curves)
    return dict(files=files, snapshot_steps=steps, n_shown=int(len(ids)), shown_ids=ids.tolist(),
                shown_cluster_rank=[int(rank[k]) for k in ls[order]], shown_dead=dead[ids].astype(int).tolist())


def part_stats(r, z, st, D: Data, lin_parts: dict):
    """Per major live cluster (largest first, at most 8): shape and higher-order statistics of its voltage
    u = c_k . g over the 60k training digits, its rate variance vs the Gaussian prediction, its digit tuning,
    a 1-feature probe, the cosine to the paired linear-rule part, and the part energy c^T C_hat c recovered by
    each readout variant relative to the true c^T Sigma c (parameters known)."""
    cfg = r["cfg"]; tau = cfg["tau"] if cfg["rule"] == "rect" else 0.0
    idx = [k for k in np.argsort(-st["sizes"]) if st["major"][k]][:8]
    if not idx:
        return []
    G = D.Gtr.double(); Gte = D.Gte.double()
    out = []
    key = (cfg["gamma"], cfg["seed"], round(cfg["extra"].get("q", -1), 4))
    hl = lin_parts.get(key)
    for k in idx:
        h = torch.as_tensor(z["cent"][k], dtype=torch.float64)
        u = G @ h; uc = u - u.mean(); sd = float(uc.std())
        sk = float((uc ** 3).mean() / sd ** 3); ku = float((uc ** 4).mean() / sd ** 4 - 3)
        var = float(h @ torch.as_tensor(D.Sigma) @ h)
        b = float(u.mean()) + tau * np.sqrt(var)
        rr = torch.relu(u - b); rv = float(rr.var(unbiased=False))
        ute = Gte @ h; rte = torch.relu(ute - b) if cfg["rule"] == "rect" else ute
        z_ = (rte - rte.mean()) / rte.std().clamp_min(1e-12)
        yte = torch.as_tensor(D.yte)
        tun = [float(z_[yte == d].mean()) for d in range(10)]
        from hebb.nonlin.analysis import logistic_probe
        ftr = torch.relu(u - b) if cfg["rule"] == "rect" else u
        p1 = logistic_probe(ftr[:, None].float(), D.ytr, rte[:, None].float(), D.yte, iters=60)["test"]
        d = dict(cluster=int(k), size=int(st["sizes"][k]), width=float(1 / (z["cent"][k] ** 2).sum()), volt_var=var, skew=sk,
                 excess_kurtosis=ku, rate_var=rv, rate_var_over_gauss=rv / (dual.k(1.0, tau) * var) if var > 0 else None,
                 frac_active=float((rr > 0).double().mean()), tuning=tun, pref_digit=int(np.argmax(tun)), probe_1feature=p1)
        if hl is not None:
            d["cos_to_linear_part"] = float(np.dot(z["cent"][k], hl) / np.linalg.norm(z["cent"][k]) / np.linalg.norm(hl))
        for nm in ("C_gauss_both", "C_chord_both", "C_gauss_expected_cov", "C_ls_both"):
            if nm in z.files:
                C = torch.as_tensor(z[nm], dtype=torch.float64)
                d[f"energy_ratio_{nm[2:]}"] = float(h @ C @ h) / var if var > 0 else None
        out.append(rnd(d))
    return out


def parts_final(r, z, st, out: Path):
    side = r["cfg"]["res"]
    sizes = st["sizes"]; idx = [k for k in np.argsort(-sizes) if not st["dead_cl"][k]][:20]
    if not idx:
        return None
    T = z["cent"][idx].reshape(-1, side, side); mx = T.reshape(len(T), -1).max(1)
    img, lay = sprite(T, 10, "magma_r", 0, mx, upscale=2 if side == 14 else 1)
    return dict(png=save_png(out / "parts_final.png", img), layout=lay, sizes=sizes[idx].tolist(),
                note="final live cluster parts (centroid h of members), largest first, own max")


def samples_sheet(feat_runs, D: Data, out: Path):
    """Export item D: Gaussian samples in pixel space x = (mu + C_+^{1/2} z)/scale, clipped to [0,1].
    Same z for every row. mu = data mean (ASSUMPTION: the weights do not determine the mean)."""
    rows = []
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((16, D.Sigma.shape[0]))
    def gen(C):
        ev, V = np.linalg.eigh((C + C.T) / 2); neg = float(-ev[ev < 0].sum() / np.abs(ev).sum())
        X = D.mu + Z @ (V * np.sqrt(np.clip(ev, 0, None))).T
        return np.clip(X / D.scale, 0, 1), neg
    real = (D.Gte[:16].numpy() / D.scale)
    rows.append(("real test digits", real, None))
    rows.append(("true moments: N(mu, Sigma_g)", *gen(D.Sigma)))
    for label, (name, key) in feat_runs.items():
        rows.append((label, *gen(np.asarray(name[key], float))))
    tiles = np.concatenate([x.reshape(16, 14, 14) for _, x, _ in rows])
    img, lay = sprite(tiles, 16, "gray_r", 0, 1, upscale=2)
    # second-moment fidelity of each generator, a cheap check (contrast cosine vs Sigma)
    return dict(png=save_png(out / "samples.png", img), layout=lay,
                rows=[dict(label=l, psd_negative_mass_clipped=None if n is None else rnd(n)) for l, _, n in rows],
                generator="x = clip((mu + C_+^{1/2} z) / pixel_scale, 0, 1); z ~ N(0, I) shared across rows; mu = training-data mean "
                          "(assumption, not read from the weights); C_+ = C_hat with negative eigenvalues clipped; absolute units "
                          "(parameters known)")


# ---------------------------------------------------------------- main
def main(run_dirs):
    SITE.mkdir(parents=True, exist_ok=True)
    runs = {}
    for d in run_dirs:
        for f in sorted(Path(d).glob("*.json")):
            if f.name == "MANIFEST.json":
                continue
            r = json.loads(f.read_text()); runs[r["name"]] = (r, f.with_suffix(".npz"))
    D14 = Data(14)
    index = dict(version=1, description="Rectified-output Hebb rule on raw MNIST pixels (docs/nonlin.md)", runs=[], featured={})
    for t in ("14", "28"):
        p = Path(__file__).parent / "configs" / f"theta_table{t}.json"
        if p.exists():
            index[f"theta_table{t}"] = rnd(json.loads(p.read_text()))
    loaded = {}
    for name, (r, npz) in sorted(runs.items()):
        z = np.load(npz)
        st_sum, alloc, st = structure(r, z)
        info = run_info(r, st_sum, alloc)
        pf = parts_final(r, z, st, SITE / "runs" / name)
        if pf:
            info["parts_final"] = pf
        loaded[name] = (r, z, st, info)
    # part statistics (paired with the linear-rule part of the same gamma, seed and q)
    lin_parts = {}
    for name, (r, z, st, info) in loaded.items():
        cfg = r["cfg"]
        if cfg["rule"] == "linear" and cfg["input"] == "pixels" and cfg["res"] == 14:
            k = int(np.argmax(st["sizes"] * ~st["dead_cl"])) if (~st["dead_cl"]).any() else None
            if k is not None:
                lin_parts[(cfg["gamma"], cfg["seed"], round(cfg["extra"].get("q", -1), 4))] = z["cent"][k]
    for name, (r, z, st, info) in loaded.items():
        if r["cfg"]["res"] == 14:
            info["parts"] = part_stats(r, z, st, D14, lin_parts)
    # featured runs (14x14, gamma = 1, seed 0)
    feat = {}
    for name, (r, z, st, info) in loaded.items():
        cfg = r["cfg"]; ex = cfg.get("extra", {})
        if cfg["res"] != 14 or cfg["seed"] != 0 or cfg["gamma"] != 1.0:
            continue
        key = ex.get("exp", "") + ("" if cfg["rule"] == "linear" else f"_t{cfg['tau']:g}")
        q = round(ex.get("q", 0), 3)
        for fk, (want_key, want_q) in FEATURED.items():
            if key == want_key and q == want_q:
                print("featured", fk, name, flush=True)
                info.update(featured(r, z, D14, st, SITE / "runs" / name))
                feat[fk] = name
    index["featured"] = feat
    index["runs"] = [loaded[n][3] for n in sorted(loaded)]
    # dual activation curves + empirical scatter
    rr = np.linspace(-1, 1, 201)
    dualj = dict(rho=rnd(rr.tolist()), ktilde={f"{t:g}": rnd((dual.k(rr, t) / dual.k(1.0, t)).tolist()) for t in (0.0, 0.5, 1.0, 2.0)},
                 dk={f"{t:g}": rnd((dual.dk(rr, t)).tolist()) for t in (0.0, 0.5, 1.0, 2.0)},
                 kraw={f"{t:g}": rnd((dual.k(rr, t)).tolist()) for t in (0.0, 0.5, 1.0, 2.0)}, scatter={})
    for key, name in feat.items():
        r = loaded[name][0]
        tau = r["cfg"]["tau"] if r["cfg"]["rule"] == "rect" else 0.0
        dualj["scatter"][key] = dict(run=name, tau=tau, **{fam: dict(rho=r["gap"][fam]["sample"]["rho"][:1500],
                                                                     rate_corr=r["gap"][fam]["sample"]["rate_corr"][:1500])
                                                           for fam in ("ii", "ib", "bb")})
    index["files"] = dict(dual=wjson(SITE / "dual.json", dualj), sigma=wbin(SITE / "sigma14.f32", D14.Sigma.astype(np.float32)),
                          mu=wbin(SITE / "mu14.f32", D14.mu.astype(np.float32)))
    # sample sheet (D)
    rows = {}
    for lab_, key, var in (("linear rule, LS readout", "A_lin", "C_ls_both"),
                           ("rectified tau=0, Gaussian inversion of weights", "A_rect_t0", "C_gauss_both"),
                           ("rectified tau=0, chord (nonlinearity ignored)", "A_rect_t0", "C_chord_both"),
                           ("rectified tau=0, Gaussian inversion of exact rate cov", "A_rect_t0", "C_gauss_expected_cov"),
                           ("Gaussian-surrogate-trained rectified tau=0 (own-quantile theta), Gaussian inversion", "Bown_rect_t0", "C_gauss_both")):
        if key in feat:
            rows[lab_ + f" [{feat[key]}]"] = (loaded[feat[key]][1], var)
    index["samples"] = samples_sheet(rows, D14, SITE)
    synth = SITE / "synth_ring.json"
    if synth.exists():
        index["files"]["synth_ring"] = str(synth.relative_to(SITE))
    wjson(SITE / "index.json", index)
    tot = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"exported {len(loaded)} runs, featured {feat}, total {tot / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1:])
