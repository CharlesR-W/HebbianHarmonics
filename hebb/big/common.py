"""Shared pieces for the big runs: paths, film encoding (site/data/big format), snapshot
selection, function-space PCA, covariance presets, part enumeration and the Perron analysis.

Film format (contract with the browser player, docs/PLAN_curation.md "Film data format"):
<id>.json holds metadata and small arrays, <id>.bin little-endian typed arrays at the byte
offsets listed in json["arrays"]. Documented in full in site/data/big/README.md.
"""
import itertools
import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "data" / "big"
RAW = ROOT / "scratch" / "out" / "big"        # gitignored: full weights, raw unit outputs
MAX_EDGES = 1500
ALIGN = 8


# ----------------------------------------------------------------------------- json helpers
def clean(x, sig=5):
    if isinstance(x, dict):
        return {str(k): clean(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v, sig) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist(), sig)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not np.isfinite(x):
            return None if np.isnan(x) else ("inf" if x > 0 else "-inf")
        return float(f"{x:.{sig}g}")
    return x


def write_json(path, obj, sig=5):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj, sig), separators=(",", ":")))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} kB)", flush=True)
    return path


# ----------------------------------------------------------------------------- snapshots
def snapshot_grid(Tmax, n=220):
    """Steps at which runs keep a snapshot: 0..10 and a log grid to Tmax."""
    g = np.unique(np.round(np.geomspace(1, Tmax, n)).astype(int))
    return set(range(11)) | set(g.tolist())


def pick_frames(steps, n_final, F):
    """Choose <= F snapshot steps, roughly log-spaced from 0 to the final step n_final
    (which must be in steps)."""
    steps = np.array(sorted(s for s in steps if s <= n_final))
    want = np.concatenate([[0], np.geomspace(1, max(n_final, 1), F - 1)])
    chosen = []
    for w in want:
        k = steps[np.argmin(np.abs(np.log1p(steps) - np.log1p(w)))]
        if k not in chosen:
            chosen.append(int(k))
    if n_final not in chosen:
        chosen.append(int(n_final))
    return sorted(chosen)


def top_edges(W, thr, kmax=MAX_EDGES):
    """Strongest <= kmax edges with w > thr: (E,2) int, (E,) float. W symmetric."""
    iu = np.triu_indices(len(W), 1)
    w = W[iu]
    live = np.nonzero(w > thr)[0]
    if len(live) > kmax:
        live = live[np.argpartition(-w[live], kmax - 1)[:kmax]]
    live = live[np.argsort(-w[live])]
    return np.stack([iu[0][live], iu[1][live]], 1).astype(np.int64), w[live].astype(float)


def n_components(W, thr):
    from scipy.sparse.csgraph import connected_components
    return int(connected_components((W > thr).astype(np.int8), directed=False)[0])


def components_sorted(W, thr):
    """Connected components of the graph w > thr, labelled 0.. by decreasing size."""
    from scipy.sparse.csgraph import connected_components
    _, lab = connected_components((W > thr).astype(np.int8), directed=False)
    return relabel_by_size(lab)


def relabel_by_size(lab):
    lab = np.asarray(lab)
    ids, counts = np.unique(lab[lab >= 0], return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    m = {int(k): i for i, k in enumerate(order)}
    return np.array([m[int(x)] if x >= 0 else -1 for x in lab], int)


# ----------------------------------------------------------------------------- function space
def pca_axes(X_final, D=3):
    """Function-space PCA fixed by the final frame. X_final: (N, P) feature rows whose Euclidean
    distances are the response distances D_ij. Returns (mean, basis (P, D)); coords of any frame
    are (X - mean) @ basis. This is classical MDS of D at the final frame."""
    mean = X_final.mean(0)
    Xc = X_final - mean
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    basis = Vt[:D].T
    for k in range(basis.shape[1]):          # sign: largest-|.| neuron coordinate positive
        c = Xc @ basis[:, k]
        if c[np.argmax(np.abs(c))] < 0:
            basis[:, k] = -basis[:, k]
    var = s ** 2 / len(Xc)
    return mean, basis, var


def psd_sqrt(M):
    lam, V = np.linalg.eigh((M + M.T) / 2)
    return (V * np.sqrt(np.clip(lam, 0, None))) @ V.T


# ----------------------------------------------------------------------------- film writer
def _quant_coords(C):
    lo = C.reshape(-1, C.shape[-1]).min(0)
    hi = C.reshape(-1, C.shape[-1]).max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    q = np.round((C - lo) / span * 65535).clip(0, 65535).astype("<u2")
    return q, lo, hi


def write_film(meta, coords, H=None, edges=None, edge_w=None, in_w=None, index_entry=None):
    """meta: dict with id, title, rule, nonlin, B, N, seed, config, script, space, space_label,
    frames (list of dicts), final, network_layout_seed (and anything else).
    coords: (F, N, D) float. H: (F, N, B) in [0,1] or None. edges: list of (E_f, 2) int arrays;
    edge_w: list of (E_f,) floats. in_w: optional (F_in, N, B) input-synapse conductances.
    Writes <id>.json and <id>.bin; updates index.json if index_entry is given."""
    fid = meta["id"]
    chunks, arrays, off = [], {}, 0

    def add(name, arr, extra):
        nonlocal off
        pad = (-off) % ALIGN
        if pad:
            chunks.append(b"\0" * pad)
            off += pad
        b = np.ascontiguousarray(arr).tobytes()
        arrays[name] = dict(offset=off, dtype=str(arr.dtype.name), shape=list(arr.shape), **extra)
        chunks.append(b)
        off += len(b)

    q, lo, hi = _quant_coords(np.asarray(coords, float))
    add("coords", q, dict(lo=lo.tolist(), hi=hi.tolist(),
                          decode="x = lo + (hi - lo) * v / 65535"))
    if H is not None:
        add("H", np.round(np.clip(H, 0, 1) * 255).astype(np.uint8), dict(decode="h = v / 255"))
    if edges is not None:
        F = len(edges)
        E = max(1, max(len(e) for e in edges))
        ea = np.zeros((F, E, 2), "<u2")
        wa = np.zeros((F, E), np.uint8)
        wmax = max([float(w.max()) for w in edge_w if len(w)] + [1e-300])
        for f, (e, w) in enumerate(zip(edges, edge_w)):
            ea[f, :len(e)] = e
            wa[f, :len(e)] = np.round(255 * np.sqrt(np.clip(w / wmax, 0, 1)))
        add("edges", ea, dict(count=[int(len(e)) for e in edges],
                              note="first count[f] rows of frame f are valid, strongest first"))
        add("edge_w", wa, dict(wmax=wmax, decode="w = wmax * (v / 255)^2"))
    if in_w is not None:
        in_w = np.asarray(in_w, float)
        imax = float(in_w.max()) if in_w.size else 1.0
        add("in_w", np.round(255 * np.sqrt(np.clip(in_w / imax, 0, 1))).astype(np.uint8),
            dict(wmax=imax, decode="w = wmax * (v / 255)^2"))
    blob = b"".join(chunks)
    meta = dict(meta)
    meta["bin"] = f"{fid}.bin"
    meta["arrays"] = arrays
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{fid}.bin").write_bytes(blob)
    write_json(OUT / f"{fid}.json", meta, sig=6)
    total = (OUT / f"{fid}.bin").stat().st_size + (OUT / f"{fid}.json").stat().st_size
    if index_entry is not None:
        update_index(dict(index_entry, id=fid, file=f"{fid}.json", bytes=total))
    return total


def update_index(entry):
    p = OUT / "index.json"
    idx = json.loads(p.read_text()) if p.exists() else []
    idx = [e for e in idx if e["id"] != entry["id"]] + [entry]
    order = {"A": 0, "B": 1, "C": 2}
    idx.sort(key=lambda e: (order.get(e.get("family", "Z")[:1], 9), e["id"]))
    p.write_text(json.dumps(idx, indent=1))


# ----------------------------------------------------------------------------- covariance presets (v1, copied from
# hebb/sweeps/common.py at commit f613d68 so the many-input cases match docs/v1/findings.md F1)
def cov_block(sizes, rho_w=0.5, rho_a=0.1):
    sizes = list(sizes)
    B = sum(sizes)
    C = np.full((B, B), float(rho_a))
    off = 0
    for m in sizes:
        C[off:off + m, off:off + m] = rho_w
        off += m
    np.fill_diagonal(C, 1.0)
    return C


def cov_hier(levels, branching):
    """Ultrametric correlation: leaves whose lowest common ancestor is at depth d (0 = siblings)
    have correlation levels[d]."""
    B = int(np.prod(branching))
    idx = np.array(list(itertools.product(*[range(b) for b in reversed(branching)])))[:, ::-1]
    C = np.eye(B)
    for i in range(B):
        for j in range(B):
            if i != j:
                C[i, j] = levels[np.nonzero(idx[i] != idx[j])[0].max()]
    return C


def cov_ring(B=16, ell=1.5):
    d = np.abs(np.subtract.outer(np.arange(B), np.arange(B)))
    d = np.minimum(d, B - d)
    return np.exp(-d ** 2 / (2 * ell ** 2))


def enumerate_parts(Sigma, theta, tol=1e-9):
    """All parts of A = Sigma - theta 11^T (B <= 16): h_S a positive eigenvector of A_SS with
    eigenvalue c > 0 and (A h)_b <= 0 off S. Stable (sufficient test) iff c is strictly the top
    eigenvalue of A_SS and the off-support inequality is strict. Same algorithm as v1
    hebb/sweeps/common.enumerate_parts."""
    B = Sigma.shape[0]
    out = []
    for r in range(1, B + 1):
        for S in itertools.combinations(range(B), r):
            S = list(S)
            ev, V = np.linalg.eigh(Sigma[np.ix_(S, S)] - theta)
            for k in range(r):
                v = V[:, k] * (1 if V[:, k].sum() >= 0 else -1)
                if v.min() <= tol or ev[k] <= tol:
                    continue
                h = np.zeros(B)
                h[S] = v / v.sum()
                off = np.setdiff1d(np.arange(B), S)
                margin = theta - (Sigma[off] @ h).max() if len(off) else np.inf
                if margin < 0:
                    continue
                gap = ev[k] - (np.delete(ev, k).max() if r > 1 else -np.inf)
                out.append(dict(S=S, h=h, c=float(ev[k]), margin=float(margin), gap=float(gap),
                                stable=bool(k == r - 1 and gap > tol and margin > tol)))
    return out


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def perron_report(H, Wib, wmin, labels, Sigma, theta, parts=None, sup_eps=1e-3):
    """Per cluster (labels >= 0): size, support (inputs whose input synapse is above the floor,
    w > wmin (1 + sup_eps), majority of members), centroid, member spread (max and mean L1 to
    the centroid), Perron fit on the support (cosine of the centroid with the Perron vector of
    A_SS, embedded in R^B), top-eigenvalue test, off-support violation max_{b not in S} (A h)_b
    (positive = violated), best |cosine| with an eigenvector of Sigma, and (if parts given)
    cosine to the nearest enumerated stable part."""
    B = Sigma.shape[0]
    A = Sigma - theta
    lamS, VS = np.linalg.eigh(Sigma)
    rows = []
    for k in range(labels.max() + 1):
        idx = np.nonzero(labels == k)[0]
        if not len(idx):
            continue
        h = H[idx].mean(0)
        above = Wib[idx] > wmin * (1 + sup_eps)
        S = np.nonzero(above.mean(0) > 0.5)[0]
        agree = float((above == (above.mean(0) > 0.5)).all(1).mean())
        l1 = np.abs(H[idx] - h).sum(1)
        row = dict(size=int(len(idx)), support=S.tolist(), support_agree=agree, centroid=h.tolist(),
                   spread=float(l1.max()), spread_mean=float(l1.mean()),
                   c_input_conductance=float(Wib[idx].sum(1).mean()))
        if len(S):
            ev, V = np.linalg.eigh(A[np.ix_(S, S)])
            v = V[:, -1] * (1 if V[:, -1].sum() >= 0 else -1)
            vfull = np.zeros(B)
            vfull[S] = v
            cosk = [abs(_cos(h[S], V[:, j])) for j in range(len(S))]
            off = np.setdiff1d(np.arange(B), S)
            Ah = A @ h
            rq = float(h[S] @ A[np.ix_(S, S)] @ h[S] / (h[S] @ h[S]))
            row.update(perron_cos=_cos(h, vfull), perron_cos_on_support=_cos(h[S], v),
                       perron_positive=bool(v.min() > 0),
                       top_eig_ok=bool(int(np.argmax(cosk)) == len(S) - 1 and v.min() > 0 and rq > 0
                                       and (len(S) == 1 or ev[-1] - ev[-2] > 1e-9)),
                       eig_gap=float(ev[-1] - ev[-2]) if len(S) > 1 else None,
                       c_rayleigh=rq, c_top=float(ev[-1]),
                       offsupport_violation=float(Ah[off].max()) if len(off) else None,
                       offsupport_mass=float(h[off].sum()) if len(off) else 0.0)
        # best eigenvector of Sigma; degenerate eigenvalues are grouped and the cosine is with the
        # projection onto the whole eigenspace (otherwise "best eigenvector" depends on the basis)
        ce, ranks = [], []
        j = 0
        while j < B:
            g = [j]
            while g[-1] + 1 < B and abs(lamS[g[-1] + 1] - lamS[j]) < 1e-9 * max(1.0, abs(lamS[j])):
                g.append(g[-1] + 1)
            pr = VS[:, g] @ (VS[:, g].T @ h)
            ce.append(float(np.linalg.norm(pr) / (np.linalg.norm(h) + 1e-300)))
            ranks.append((int(B - 1 - g[-1]), len(g)))
            j = g[-1] + 1
        m = int(np.argmax(ce))
        row.update(best_eig_cos=ce[m], best_eig_rank=ranks[m][0], best_eig_multiplicity=ranks[m][1])
        if parts:
            st = [p for p in parts if p["stable"]]
            if st:
                cs = [_cos(h, p["h"]) for p in st]
                m = int(np.argmax(cs))
                row.update(nearest_stable_part_cos=cs[m], nearest_stable_part_support=st[m]["S"])
        rows.append(row)
    return rows
