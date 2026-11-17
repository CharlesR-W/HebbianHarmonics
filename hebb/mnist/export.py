"""Build the page data under site/data/mnist/ from fetched run artifacts.

    PYTHONPATH=. python -m hebb.mnist.export --runs /var/tmp/hebb-mnist/runs/<run_id> [...] \
        --featured name1,name2 --headline name1

Layout and every file format: site/data/mnist/README.md (written by write_readme below).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from hebb.mnist.data import load_mnist, make_lift, lifted_stats, offdiag_quantiles
from hebb.mnist import sampling as SM

SITE = Path(__file__).resolve().parents[2] / "site" / "data" / "mnist"
N_SHOW = 256          # neurons per similarity matrix / map / graph
N_EDGES = 3000        # strongest edges per graph snapshot
_rr, _cc = np.divmod(np.arange(196), 14)
PIX_XY = np.stack([_cc, _rr], 1).astype(np.float32)   # boundary pixel b -> (x = column, y = row)


# ---------------------------------------------------------------- helpers
def lift_key(cfg):
    if cfg["lift"] == "pixels":
        return "pixels"
    return f"{cfg['lift']}{cfg['B']}" + ("" if cfg["bw_mult"] == 1 else f"bw{cfg['bw_mult']:g}")


def init_key(cfg):
    return "generic" if cfg["ib_shape"] == 1.0 else f"heavy{cfg['ib_shape']:g}"


def wbin(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    return dict(file=str(path.relative_to(SITE)), dtype=str(arr.dtype), shape=list(arr.shape))


def wjson(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), default=_jdef))
    return str(path.relative_to(SITE))


def _jdef(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def rnd(x, n=4):
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{x:.{n}g}")
    return x


def sprite(tiles: np.ndarray, ncol: int, cmap: str, vmin, vmax, upscale: int = 2, pad: int = 1):
    """tiles (n, h, w) -> RGB uint8 sprite with `pad` px white gutters; returns (img, layout)."""
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    n, h, w = tiles.shape
    nrow = int(np.ceil(n / ncol))
    th, tw = h * upscale, w * upscale
    img = np.full((nrow * (th + pad) + pad, ncol * (tw + pad) + pad, 3), 255, np.uint8)
    for i in range(n):
        v = np.asarray(vmin[i] if np.ndim(vmin) else vmin), np.asarray(vmax[i] if np.ndim(vmax) else vmax)
        t = np.clip((tiles[i] - v[0]) / max(float(v[1] - v[0]), 1e-12), 0, 1)
        rgb = (cm(t)[..., :3] * 255).astype(np.uint8)
        rgb = rgb.repeat(upscale, 0).repeat(upscale, 1)
        r, c = divmod(i, ncol)
        y, x = pad + r * (th + pad), pad + c * (tw + pad)
        img[y:y + th, x:x + tw] = rgb
    return img, dict(tile_w=tw, tile_h=th, pad=pad, ncol=ncol, nrow=nrow, n=n, cmap=cmap)


def save_png(path: Path, img: np.ndarray):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, optimize=True)
    return str(path.relative_to(SITE))


def procrustes(A, B):
    """Rotate/reflect+translate A (n,2) onto B; scale preserved (fit uniform scale too)."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, s, Vt = np.linalg.svd(A0.T @ B0)
    R = U @ Vt
    sc = s.sum() / max((A0 ** 2).sum(), 1e-12)
    return sc * A0 @ R + mb


def umap_series(Hs: np.ndarray, seed=0):
    """Hs (S, n, B) probability vectors -> (S, n, 2): Hellinger UMAP at the last snapshot,
    then backwards, each frame initialised at the next (later) frame's layout and
    Procrustes-aligned to it."""
    import umap
    import warnings
    warnings.filterwarnings("ignore")
    S, n, _ = Hs.shape
    out = np.zeros((S, n, 2), np.float32)
    prev = None
    rng = np.random.default_rng(seed)
    for s in range(S - 1, -1, -1):
        H = np.clip(Hs[s].astype(np.float64), 0, None)
        H = H / H.sum(1, keepdims=True)
        H = H + rng.normal(0, 1e-9, H.shape).clip(0)  # break exact ties among identical rows
        kw = dict(n_neighbors=15, min_dist=0.15, metric="hellinger", random_state=seed, n_epochs=200)
        if prev is not None:
            kw["init"] = prev
        Y = umap.UMAP(**kw).fit_transform(H)
        if prev is not None:
            Y = procrustes(Y, prev)
        Y = Y - Y.mean(0)
        out[s] = Y
        prev = Y
    return out


# ---------------------------------------------------------------- per-lift statistics
class LiftCtx:
    def __init__(self, cfg, x_tr, x_te, y_te):
        self.key = lift_key(cfg)
        self.lift = make_lift(cfg["lift"], cfg["B"], cfg["lift_seed"], cfg["bw_mult"], x_tr)
        self.mu, self.Sigma = lifted_stats(self.lift, x_tr)
        self.G_te = self.lift.raw(x_te).astype(np.float64) - self.mu      # centred test features
        self.x_te, self.y_te = x_te, y_te
        self.xbar = x_tr.mean(0)
        P = np.eye(cfg["B"]) - 1.0 / cfg["B"]
        ev, V = np.linalg.eigh(P @ self.Sigma @ P)
        self.ev, self.V = ev[::-1], V[:, ::-1]

    def responses(self, Hrows):
        """r(x) = h^T (g(x) - mu_g) over the test set: (n_test, k)."""
        return self.G_te @ Hrows.T

    def sta(self, Hrows):
        """response-weighted average stimulus E[r (x - xbar)] / sd(r): (k, 196), pixel units."""
        R = self.responses(Hrows)
        R = R / np.maximum(R.std(0, keepdims=True), 1e-12)
        return (R.T @ (self.x_te - self.x_te.mean(0))) / len(R)

    def tuning(self, Hrows):
        """mean z-scored response per digit class: (k, 10)."""
        R = self.responses(Hrows)
        R = (R - R.mean(0)) / np.maximum(R.std(0), 1e-12)
        return np.stack([R[self.y_te == c].mean(0) for c in range(10)], 1)

    def top_digits(self, Hrows, k=8):
        R = self.responses(Hrows)
        return np.argsort(-R, 0)[:k].T   # (rows, k) indices into the test set


def export_lift(ctx: LiftCtx, x28_te):
    d = SITE / "lifts" / ctx.key
    B = ctx.lift.B
    info = dict(key=ctx.key, **ctx.lift.describe(), sigma_stats=offdiag_quantiles(ctx.Sigma))
    info["sigma_stats"]["diag"] = rnd(np.quantile(np.diag(ctx.Sigma), [0, .25, .5, .75, 1]).tolist())
    iu = np.triu_indices(B, 1)
    hist, edges = np.histogram(ctx.Sigma[iu], bins=60)
    info["offdiag_hist"] = dict(counts=hist.tolist(), edges=rnd(edges.tolist()))
    info["files"] = {}
    if B <= 256:
        info["files"]["sigma"] = wbin(d / "sigma.f32", ctx.Sigma.astype(np.float32))
    info["files"]["sigma_contrast_spectrum"] = wbin(d / "sigma_spectrum.f32", ctx.ev.astype(np.float32))
    # top eigenvectors rendered like the clusters
    k = 16
    V = ctx.V[:, :k].T.copy()
    for i in range(k):  # sign: make the response-weighted average point "into" the digit mass
        if (ctx.sta(V[i:i + 1]) * (ctx.xbar - 0.5)).sum() < 0 and ctx.key != "pixels":
            V[i] *= -1
    if ctx.key == "pixels":
        tiles = V.reshape(k, 14, 14)
        a = np.abs(tiles).max((1, 2))
        img, lay = sprite(tiles, 8, "RdBu_r", -a, a, upscale=4)
        info["files"]["eigen_images"] = dict(png=save_png(d / "eigen_images.png", img), layout=lay,
                                             note="top-16 eigenvectors of P Sigma P as 14x14 images, each on its own symmetric scale (red +, blue -)")
    sta = ctx.sta(V).reshape(k, 14, 14)
    a = np.abs(sta).max((1, 2))
    img, lay = sprite(sta, 8, "RdBu_r", -a, a, upscale=4)
    info["files"]["eigen_sta"] = dict(png=save_png(d / "eigen_sta.png", img), layout=lay,
                                      note="response-weighted average stimulus (centred) of the top-16 eigen-directions v: E[r(x)(x - xbar)]/sd(r), r = v^T(g - mu_g)")
    info["eigen_tuning"] = rnd(ctx.tuning(V).tolist(), 3)
    td = ctx.top_digits(V, 8)
    img, lay = sprite(x28_te[td.reshape(-1)], 8, "gray_r", 0, 1, upscale=1)
    info["files"]["eigen_top_digits"] = dict(png=save_png(d / "eigen_top_digits.png", img), layout=lay,
                                             note="row i = 8 test digits with the largest response to eigen-direction i")
    return info


# ---------------------------------------------------------------- per-run export
def choose_neurons(lab_sub: np.ndarray, H_fin_sub: np.ndarray, dead_set: set, n_show=N_SHOW, seed=0):
    """Stratified subsample of the stored neurons: each cluster gets min(size, 4) members,
    the rest proportional to cluster size.  Returns positions into the stored subset, ordered
    by cluster (largest live cluster first, dead pool last), then by L1 distance to the
    cluster centroid."""
    rng = np.random.default_rng(seed)
    n = len(lab_sub)
    if n <= n_show:
        pick = np.arange(n)
    else:
        ks, counts = np.unique(lab_sub, return_counts=True)
        guar = np.minimum(counts, 4)
        rest = n_show - guar.sum()
        extra = np.floor(rest * (counts - guar) / max((counts - guar).sum(), 1)).astype(int)
        while guar.sum() + extra.sum() < n_show:
            extra[np.argmax(counts - guar - extra)] += 1
        pick = []
        for k, g_, e_ in zip(ks, guar, extra):
            m = np.where(lab_sub == k)[0]
            pick += rng.choice(m, min(len(m), g_ + e_), replace=False).tolist()
        pick = np.array(pick)
    lab = lab_sub[pick]
    sizes = np.bincount(lab_sub)
    key = []
    for p, k in zip(pick, lab):
        cent = H_fin_sub[lab_sub == k].mean(0)
        key.append((1 if k in dead_set else 0, -sizes[k], k, np.abs(H_fin_sub[p] - cent).sum()))
    order = sorted(range(len(pick)), key=lambda i: key[i])
    return pick[order]


def boundary_order(cent_live: np.ndarray, ev_V: np.ndarray):
    """Order boundary nodes: by the live cluster whose part puts most mass on them (clusters
    in size order), within by that mass; unsupported nodes last, by leading eigenvector."""
    B = ev_V.shape[0]
    if len(cent_live):
        owner = cent_live.argmax(0)
        mass = cent_live.max(0)
        supported = mass > 2.0 / B
    else:
        owner = np.zeros(B, int); mass = np.zeros(B); supported = np.zeros(B, bool)
    a = [b for b in range(B) if supported[b]]
    a.sort(key=lambda b: (owner[b], -mass[b]))
    rest = [b for b in range(B) if not supported[b]]
    rest.sort(key=lambda b: -ev_V[b, 0])
    return np.array(a + rest), int(len(a))


def support_metrics(z, ctx, dead_set):
    """How good is C_hat where the network actually has equations?  Support = boundary
    nodes carrying > 2/B of some live cluster part's mass."""
    cent = z["cent"].astype(np.float64)
    B = cent.shape[1]
    live = [k for k in range(len(cent)) if k not in dead_set]
    if not live:
        return dict(n_supported=0)
    sup = (cent[live] > 2.0 / B).any(0)
    S = ctx.Sigma
    res = dict(n_supported=int(sup.sum()), frac_supported=float(sup.mean()))
    P = np.eye(B) - 1.0 / B
    Sp = P @ S @ P
    blk = np.ix_(sup, sup)
    res["sigma_contrast_mass_on_support_block"] = float((Sp[blk] ** 2).sum() / (Sp ** 2).sum())
    for v in ("ls_both", "ls_ib", "ls_ii"):
        C = z[f"C_{v}"].astype(np.float64)
        Cs, Ss = C[blk], S[blk]
        res[v] = dict(cos_on_support=float((Cs * Ss).sum() / max(np.linalg.norm(Cs) * np.linalg.norm(Ss), 1e-300)),
                      relerr_on_support_known=float(np.linalg.norm(Cs - Ss) / np.linalg.norm(Ss)),
                      frac_C_mass_on_support_block=float((C[blk] ** 2).sum() / max((C ** 2).sum(), 1e-300)))
        parts = cent[live]
        q_true = np.einsum("kb,bc,kc->k", parts, S, parts)
        q_hat = np.einsum("kb,bc,kc->k", parts, C, parts)
        res[v]["part_energy_relerr_median"] = float(np.median(np.abs(q_hat - q_true) / np.abs(q_true)))
    return rnd(res)


def export_run(name: str, run_dir: Path, ctx: LiftCtx, x28_te, featured: bool):
    r = json.loads((run_dir / f"{name}.json").read_text())
    cfg = r["cfg"]
    out = dict(name=name, lift=ctx.key, B=cfg["B"], N=cfg["N"], theta=cfg["theta"], wmin=cfg["wmin"],
               gamma=cfg["gamma"], init=init_key(cfg), ib_shape=cfg["ib_shape"], ii_scale=cfg["ii_scale"],
               mode=cfg["mode"], seed=cfg["seed"], steps=cfg["steps"], dt=cfg["dt"], batch=cfg["batch"],
               featured=featured)
    iu = np.triu_indices(cfg["B"], 1)
    out["theta_percentile"] = float((ctx.Sigma[iu] < cfg["theta"]).mean())
    f = r["final"]
    out["final"] = rnd({k: v for k, v in f.items() if k != "dead_clusters"})
    out["cluster_sizes"] = r["cluster_sizes"]
    out["cluster_energy"] = rnd(r["cluster_energy"])
    out["dead_clusters"] = f.get("dead_clusters", [])
    out["readout"] = rnd(r["readout"]); out["baselines"] = rnd(r["baselines"])
    out["readout_info"] = rnd(r["readout_info"]); out["measurement_rank"] = r["measurement_rank"]
    out["timing"] = rnd(r["timing"])
    z = np.load(run_dir / f"{name}.npz")
    out["support"] = support_metrics(z, ctx, set(f.get("dead_clusters", [])))
    if not featured:
        return out

    d = SITE / "runs" / name
    if d.exists():
        shutil.rmtree(d)
    files = {}
    snaps = np.array(r["snaps"])
    rows = r["snap_rows"]
    keep = ["step", "time", "n_clusters", "n_clusters_tol05", "n_clusters_tol20", "n_major", "n_leader",
            "n_live_clusters", "n_live_major", "n_dead", "n_active_components", "within_l1", "eff_rank",
            "floor_ii", "floor_ib", "mean_wii", "mean_wib", "max_w", "res_max", "res_mean", "res_active_mean"]
    ro_keys = [k for k in rows[0] if k.startswith("ro_")]
    curves = {k: [rw.get(k) for rw in rows] for k in keep + ro_keys}
    curves = rnd(curves)
    dense = z["dense"]
    curves["dense"] = dict(cols=r["dense_cols"], rows=rnd(dense.tolist()))
    curves["hist_edges"] = rnd(z["hist_edges"].tolist())
    curves["hist_ii"] = z["hist_ii"].tolist(); curves["hist_ib"] = z["hist_ib"].tolist()
    curves["cluster_sizes_snap"] = [s[s > 0].tolist() for s in z["sizes_snap"]]
    curves["cluster_energy_snap"] = rnd(r["cluster_energy_snap"], 3)
    curves["equilibrium_note"] = ("readout metrics at intermediate snapshots use instantaneous weights before "
                                  "convergence: 'equilibrium assumption violated' control; residual res_* is the "
                                  "projected mean-field drift max|dw/dt| at the snapshot weights")
    files["curves"] = wjson(d / "curves.json", curves)

    # ---- neuron subsample and fixed order
    sub = z["sub"]; lab_all = z["labels"]; H_sub = z["H_sub"]          # (S, n_sub, B) f16
    H_fin = z["H_final"].astype(np.float32)
    lab_sub = lab_all[sub]
    dead_set = set(out["dead_clusters"])
    pos = choose_neurons(lab_sub, H_fin[sub], dead_set)
    ids = sub[pos]
    Hs = H_sub[:, pos].astype(np.float32)                               # (S, n, B)
    S_, n, B = Hs.shape
    Sig = ctx.Sigma.astype(np.float32)
    sims = np.einsum("snb,bc,smc->snm", Hs, Sig, Hs)
    lo, hi = float(np.quantile(sims, 0.001)), float(sims.max())
    a = max(abs(lo), abs(hi))
    q = np.clip(np.round((sims + a) / (2 * a) * 255), 0, 255).astype(np.uint8)
    files["similarity"] = dict(**wbin(d / "similarity.u8", q), decode="value = -a + 2a*byte/255", a=a,
                               note="H Sigma_g H^T for the shown neurons at every snapshot, fixed neuron order; symmetric colour range [-a, a]")
    labels_show = lab_all[ids]
    sizes = np.bincount(lab_all)
    # cluster info (live clusters ordered by size; dead clusters flagged)
    K = len(sizes)
    cent = z["cent"].astype(np.float64)
    order_k = sorted(range(K), key=lambda k: (k in dead_set, -sizes[k]))
    clusters = []
    sta = ctx.sta(cent) if K else np.zeros((0, 196))
    tun = ctx.tuning(cent)
    for rank_, k in enumerate(order_k):
        clusters.append(dict(id=int(k), rank=rank_, size=int(sizes[k]), dead=bool(k in dead_set),
                             energy=float(r["cluster_energy"][k]),
                             support=int((cent[k] > 2.0 / B).sum()), entropy_eff=float(np.exp(-(cent[k][cent[k] > 0] * np.log(cent[k][cent[k] > 0])).sum())),
                             tuning=rnd(tun[k].tolist(), 3), preferred_digit=int(np.argmax(tun[k]))))
    # per-neuron preferred digit (final embedding)
    tn = ctx.tuning(H_fin[ids].astype(np.float64))
    neurons = dict(ids=ids.tolist(), cluster=labels_show.tolist(),
                   cluster_rank=[order_k.index(int(k)) for k in labels_show],
                   dead=[bool(k in dead_set) for k in labels_show],
                   preferred_digit=np.argmax(tn, 1).tolist(),
                   note="ids = interior-neuron indices; order = fixed display order (cluster, then distance to centroid); "
                        "sample is stratified (each cluster >= min(size,4) members, rest proportional to size): use cluster sizes, not counts in the sample")
    # ---- cluster images
    nshow_k = min(K, 40)
    tiles_k = [k for k in order_k][:nshow_k]
    if ctx.key == "pixels":
        P = cent[tiles_k].reshape(-1, 14, 14)
        img, lay = sprite(P, 10, "magma_r", 0, P.max((1, 2)), upscale=4)
        files["cluster_parts"] = dict(png=save_png(d / "cluster_parts.png", img), layout=lay, clusters=[int(k) for k in tiles_k],
                                      note="cluster part c_k (mean harmonic embedding, a probability vector over the 196 pixels), each tile scaled to its own max")
    files["cluster_part_arrays"] = wbin(d / "cluster_parts.f32", cent[order_k].astype(np.float32))
    T = sta[tiles_k].reshape(-1, 14, 14)
    a_ = np.abs(T).max((1, 2))
    img, lay = sprite(T, 10, "RdBu_r", -a_, a_, upscale=4)
    files["cluster_sta"] = dict(png=save_png(d / "cluster_sta.png", img), layout=lay, clusters=[int(k) for k in tiles_k],
                                note="response-weighted average stimulus (centred, pixel units) of each cluster part, own symmetric scale")
    td = ctx.top_digits(cent[tiles_k], 8)
    img, lay = sprite(x28_te[td.reshape(-1)], 8, "gray_r", 0, 1, upscale=1)
    files["cluster_top_digits"] = dict(png=save_png(d / "cluster_top_digits.png", img), layout=lay, clusters=[int(k) for k in tiles_k],
                                       note="row = cluster (same order as cluster_sta), 8 test digits with the largest response")
    if ctx.key == "pixels":
        # individual neurons' h_i over training: 16 neurons (spread over clusters) x S snapshots
        pick16 = np.linspace(0, n - 1, 16).astype(int)
        T = Hs[:, pick16].transpose(1, 0, 2).reshape(-1, 14, 14)
        mx = np.repeat(Hs[:, pick16].max(2).T.reshape(-1), 1)
        img, lay = sprite(T, S_, "magma_r", 0, mx, upscale=2)
        files["neuron_tiles"] = dict(png=save_png(d / "neuron_tiles.png", img), layout=lay, neurons=pick16.tolist(),
                                     note="row = one shown neuron (index into neurons.ids), column = snapshot; h_i as 14x14, each tile on its own max")
    # ---- UMAP over snapshots
    U = umap_series(Hs)
    files["umap"] = dict(**wbin(d / "umap.f32", U), note="(snapshot, neuron, xy) Hellinger-UMAP of the shown neurons' embeddings, warm-started backwards from the final snapshot and Procrustes-aligned")
    # ---- graph at the 6 graph snapshots
    g_steps = z["g_steps"].tolist()
    gpos = [int(np.where(snaps == s)[0][0]) for s in g_steps]
    Wii_g = z["Wii_g"][:, pos][:, :, pos].astype(np.float32)
    Wib_g = z["Wib_g"][:, pos].astype(np.float32)
    edges, layouts = [], []
    with_b = ctx.key == "pixels"
    for gi, (s, sp) in enumerate(zip(g_steps, gpos)):
        Wi = np.triu(Wii_g[gi], 1)
        r_, c_ = np.nonzero(Wi > 2 * cfg["wmin"])
        e = [np.stack([r_, c_, Wi[r_, c_]], 1)]
        if with_b:
            r2, c2 = np.nonzero(Wib_g[gi] > 2 * cfg["wmin"])
            e.append(np.stack([r2, n + c2, Wib_g[gi][r2, c2]], 1))
        e = np.concatenate(e, 0) if len(e) else np.zeros((0, 3))
        e = e[np.argsort(-e[:, 2])][:N_EDGES].astype(np.float32)
        edges.append(e)
        lay = dict(step=int(s), umap_snapshot=sp, n_edges=int(len(e)))
        if with_b:
            xy = Hs[sp] @ PIX_XY
            xy = xy + np.random.default_rng(gi).normal(0, 0.12, xy.shape)
            lay["tutte_xy"] = rnd(xy.tolist(), 4)
        layouts.append(lay)
    ecat = np.concatenate(edges, 0) if edges else np.zeros((0, 3), np.float32)
    files["graph"] = dict(**wbin(d / "graph_edges.f32", ecat), snapshots=layouts,
                          n_neurons=int(n), n_boundary=int(B if with_b else 0),
                          boundary_xy=PIX_XY.tolist() if with_b else None,
                          note="edges (src, dst, weight) concatenated over graph snapshots (n_edges each, strongest first); "
                               "nodes 0..n-1 = shown neurons (neurons.ids order), n..n+B-1 = boundary pixels; only active edges (w > 2 w_min); "
                               "layout: neurons at UMAP coords of snapshot umap_snapshot, or (pixels) Tutte/harmonic position H @ pixel_xy plus jitter sd 0.12 px")
    # ---- readout
    live_k = [k for k in order_k if k not in dead_set]
    border, nsup = boundary_order(cent[live_k] if live_k else np.zeros((0, B)), ctx.V)
    ro = dict(order=border.tolist(), n_supported=nsup, spectra={})
    for kname in [k for k in z.files if k.startswith("spec_")]:
        ro["spectra"][kname[5:]] = rnd(z[kname].astype(np.float64).tolist(), 4)
    if B <= 256:
        for v in ["ls_both", "ls_ii", "ls_ib", "naive_ib", "init_ls_both"]:
            files[f"chat_{v}"] = wbin(d / f"chat_{v}.f32", z[f"C_{v}"].astype(np.float32))
    files["readout"] = wjson(d / "readout.json", ro)
    files["neurons"] = wjson(d / "neurons.json", dict(neurons=neurons, clusters=clusters,
                                                     snapshot_steps=snaps.tolist()))
    out["files"] = files
    out["n_shown"] = int(n)
    out["snapshot_steps"] = snaps.tolist()
    return out


# ---------------------------------------------------------------- filmstrip
def filmstrip(name: str, run_out: dict, n_frames=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = SITE / "runs" / name
    fs = run_out["files"]["similarity"]
    q = np.fromfile(SITE / fs["file"], np.uint8).reshape(fs["shape"])
    a = fs["a"]
    steps = run_out["snapshot_steps"]
    S = len(steps)
    idx = np.unique(np.round(np.geomspace(1, S - 1, n_frames)).astype(int))
    idx = np.concatenate([[0], idx])[-n_frames:] if len(idx) < n_frames else idx
    fig, axs = plt.subplots(1, len(idx), figsize=(2.3 * len(idx), 2.7), dpi=110)
    for ax, i in zip(axs, idx):
        v = -a + 2 * a * q[i].astype(np.float32) / 255
        ax.imshow(v, cmap="RdBu_r", vmin=-a, vmax=a, interpolation="nearest")
        ax.set_title(f"step {steps[i]}  (t = {steps[i] * run_out['dt']:g})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"H Σ H^T over training, {name} (fixed neuron order, colour range ±{a:.2f})", fontsize=9)
    fig.tight_layout()
    p = SITE / f"filmstrip_{name}.png"
    fig.savefig(p); plt.close(fig)
    return str(p.relative_to(SITE)), [int(steps[i]) for i in idx]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--featured", default="")
    ap.add_argument("--headline", default="")
    ap.add_argument("--samples", default="", help="comma list of run names to synthesise samples from")
    ap.add_argument("--only-index", action="store_true")
    a = ap.parse_args()
    featured = set(filter(None, a.featured.split(",")))
    x_tr, _ = load_mnist("train")
    x_te, y_te = load_mnist("test")
    from hebb.mnist.data import _read_idx, MNIST_DIR
    x28_te = _read_idx(MNIST_DIR / "t10k-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    runs, lifts, ctxs = [], {}, {}
    found = {}
    for rd in a.runs:
        for p in sorted(Path(rd).glob("*.json")):
            if p.name == "MANIFEST.json":
                continue
            found[p.stem] = Path(rd)
    for name, rd in sorted(found.items()):
        cfg = json.loads((rd / f"{name}.json").read_text())["cfg"]
        key = lift_key(cfg)
        if key not in ctxs:
            ctxs[key] = LiftCtx(cfg, x_tr, x_te, y_te)
            lifts[key] = export_lift(ctxs[key], x28_te)
            print("lift", key, flush=True)
        is_f = name in featured and not a.only_index and (rd / f"{name}.npz").exists()
        runs.append(export_run(name, rd, ctxs[key], x28_te, is_f))
        print("run", name, "featured" if is_f else "", flush=True)
    index = dict(
        version=1, generated_by="hebb/mnist/export.py",
        description="Hebbian resistor network trained on kernel-lifted MNIST; see README.md for every file",
        headline=a.headline, lifts=lifts, runs=runs)
    if a.headline:
        h = next(r for r in runs if r["name"] == a.headline)
        if h.get("files"):
            p, st = filmstrip(a.headline, h)
            index["filmstrip"] = dict(png=p, steps=st, run=a.headline)
    if a.samples:
        from hebb.mnist.samples_export import export_samples
        index["samples"] = export_samples([s for s in a.samples.split(",") if s], found, ctxs, x_tr, x_te, x28_te, SITE)
    wjson(SITE / "index.json", index)
    tot = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file())
    print(f"site/data/mnist total {tot / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
