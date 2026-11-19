"""Static part theory (claim C1): exact, vectorised enumeration of the Perron parts of
A = Sigma - theta 11^T (B <= 16), part diagrams on a theta grid, and the comparison with the
threshold graph G_theta = {a != b : Sigma_ab > theta}.

Part (same definition as hebb/big/common.enumerate_parts, checked against it in `selftest`):
S a support, h_S > 0 an eigenvector of A_SS with eigenvalue c > 0, (A h)_b <= 0 for b not in S
(h normalised to sum 1, so (A h)_b = Sigma_bS h_S - theta; margin = theta - max_b Sigma_bS h_S).
Stable: c strictly the top eigenvalue of A_SS (gap > TOL) and margin > TOL.

Run:  uv run python -m hebb.perron2.parts [selftest|diagrams|edges|random|all]
"""
import itertools
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from hebb import core  # noqa: E402
from hebb.big import common as cm  # noqa: E402

ROOT = cm.ROOT
OUT = ROOT / "site" / "data" / "perron2"
RAW = ROOT / "scratch" / "out" / "perron2"
TOL = 1e-9
SCRIPT = "hebb/perron2/parts.py"


# ----------------------------------------------------------------------------- covariances
def ultra_tree(tree):
    """Ultrametric matrix from a nested tree. Node = (corr, [children]); leaf = None or
    ("leaf", variance). Leaves whose lowest common ancestor is node n get Sigma = corr(n).
    Unit variance unless given. Returns (Sigma, clusters) with clusters = [(corr, leaves)]
    for every internal node."""
    leaves, clusters, var = [], [], []

    def walk(node):
        if node is None or (isinstance(node, tuple) and node[0] == "leaf"):
            leaves.append(len(leaves))
            var.append(1.0 if node is None else node[1])
            return [leaves[-1]]
        corr, ch = node
        groups = [walk(c) for c in ch]
        allv = sum(groups, [])
        clusters.append((corr, allv, groups))
        return allv

    walk(tree)
    B = len(leaves)
    S = np.zeros((B, B))
    for corr, _, groups in clusters:
        for g1, g2 in itertools.combinations(groups, 2):
            S[np.ix_(g1, g2)] = corr
            S[np.ix_(g2, g1)] = corr
    S[np.diag_indices(B)] = var
    return S, [(c, sorted(a)) for c, a, _ in clusters]


def random_ultra(rng, B, unit=True):
    """Random ultrametric correlation: random binary-or-wider merge tree, merge heights
    decreasing toward the root (uniform random in (0, 0.95))."""
    heights = np.sort(rng.uniform(0.0, 0.95, B))[::-1]   # heights used in merge order
    k = 0
    items = [dict(corr=1.0, node=None) for _ in range(B)]
    while len(items) > 1:
        m = int(min(len(items), rng.choice([2, 2, 2, 3, 4])))
        pick = sorted(rng.choice(len(items), m, replace=False).tolist(), reverse=True)
        ch = [items.pop(i) for i in pick]
        h = min([heights[k]] + [c["corr"] - 1e-3 for c in ch])
        k += 1
        items.append(dict(corr=h, node=(h, [c["node"] for c in ch])))
    S, _ = ultra_tree(items[0]["node"])
    return S


def sigma_families():
    """name -> dict(Sigma, desc, kind). kind: 'ultra' (unit-variance ultrametric), 'ultra_var'
    (ultrametric matrix, unequal diagonal), 'drd' (D R D, R ultrametric), 'nonultra'."""
    F = {}
    hier = cm.cov_hier((0.8, 0.5, 0.25, 0.05), (2, 2, 2, 2))
    F["hier16"] = dict(Sigma=hier, kind="ultra", desc="binary hierarchy depth 4, 0.8/0.5/0.25/0.05 (cov_runs hier16)")
    F["hier12_322"] = dict(Sigma=cm.cov_hier((0.7, 0.4, 0.1), (2, 2, 3)), kind="ultra",
                           desc="branching (3 at the root, then 2, 2), levels 0.7/0.4/0.1")
    L = None
    tree = (0.08, [(0.45, [(0.85, [L, L, L]), (0.6, [L, L])]),
                   (0.3, [L, (0.7, [L, L, L, L]), (0.55, [L, L])]),
                   (0.9, [L, L])])
    F["irreg14"] = dict(Sigma=ultra_tree(tree)[0], kind="ultra", tree=repr(tree),
                        desc="irregular tree, 14 leaves: {(3 @0.85, 2 @0.6) @0.45, (1, 4 @0.7, 2 @0.55) @0.3, 2 @0.9} @0.08")
    rng = np.random.default_rng(7)
    F["randultra16"] = dict(Sigma=random_ultra(rng, 16), kind="ultra", desc="random ultrametric tree, 16 leaves (seed 7)")
    # unequal variances
    var_rng = np.random.default_rng(11)
    v = var_rng.uniform(0.5, 1.5, 16)
    F["hier16_drd"] = dict(Sigma=hier * np.sqrt(np.outer(v, v)), kind="drd", variances=v,
                           desc="D R D with R = hier16, variances U(0.5, 1.5) (seed 11)")
    Sv = hier.copy()
    vv = np.where(np.arange(16) % 3 == 0, 0.85, 1.4)
    np.fill_diagonal(Sv, vv)
    F["hier16_ultravar"] = dict(Sigma=Sv, kind="ultra_var", variances=vv,
                                desc="hier16 covariances off the diagonal, variances 0.85 (every 3rd input) or 1.4: still an ultrametric matrix")
    # non-ultrametric
    for amp in (0.01, 0.03, 0.1):
        for s in range(3):
            r = np.random.default_rng(100 + s)
            E = r.uniform(-amp, amp, (16, 16))
            E = np.triu(E, 1)
            E = E + E.T
            S = core.clamp_psd(hier + E, unit_diag=True)
            F[f"hier16_noise{amp}_s{s}"] = dict(Sigma=S, kind="nonultra",
                                               desc=f"hier16 + symmetric uniform noise +-{amp} off the diagonal (seed {100 + s}), PSD-clamped, unit diagonal")
    F["block12"] = dict(Sigma=cm.cov_block([4, 4, 4], 0.5, 0.1), kind="nonultra_ultra",
                        desc="3 blocks of 4, within 0.5, across 0.1 (this one is in fact ultrametric)")
    F["ring16"] = dict(Sigma=cm.cov_ring(16, 1.5), kind="nonultra", desc="ring of 16, exp(-d^2/(2*1.5^2))")
    for k, f in F.items():
        lam = np.linalg.eigvalsh(f["Sigma"])
        assert lam[0] > -1e-10, (k, lam[0])
    return F


# ----------------------------------------------------------------------------- enumeration
_SUBSETS = {}


def subsets(B):
    if B not in _SUBSETS:
        d = {}
        for r in range(1, B + 1):
            idx = np.array(list(itertools.combinations(range(B), r)), dtype=np.int64)
            d[r] = (idx, (1 << idx).sum(1))
        _SUBSETS[B] = d
    return _SUBSETS[B]


def mask_list(m):
    return [b for b in range(64) if (m >> b) & 1]


def enumerate_fast(Sigma, theta, tol=TOL, only_stable=False):
    """All parts (vectorised). Returns list of dict(mask, S, h (B,), c, margin, gap, stable).
    Degenerate eigenspaces: eigh picks some basis, so positive vectors hidden in a degenerate
    eigenspace can be missed; such parts are never stable (gap 0) -- same as common.enumerate_parts."""
    B = len(Sigma)
    out = []
    for r, (idx, masks) in subsets(B).items():
        sub = Sigma[idx[:, :, None], idx[:, None, :]] - theta
        ev, V = np.linalg.eigh(sub)                    # (M, r), (M, r, r)
        ks = [r - 1] if only_stable else range(r)
        for k in ks:
            v = V[:, :, k]
            v = v * np.where(v.sum(1) >= 0, 1.0, -1.0)[:, None]
            ok = (v.min(1) > tol) & (ev[:, k] > tol)
            if not ok.any():
                continue
            ii = np.nonzero(ok)[0]
            h = v[ii] / v[ii].sum(1, keepdims=True)
            H = np.zeros((len(ii), B))
            np.put_along_axis(H, idx[ii], h, 1)
            Ah = H @ Sigma                             # Sigma_b. h (sum h = 1)
            onS = np.zeros((len(ii), B), bool)
            np.put_along_axis(onS, idx[ii], True, 1)
            offmax = np.where(onS, -np.inf, Ah).max(1)
            margin = theta - offmax                     # +inf when S = everything
            if r > 1:
                others = np.delete(ev[ii], k, axis=1).max(1)
                gap = ev[ii, k] - others
            else:
                gap = np.full(len(ii), np.inf)
            keep = margin >= 0
            for j in np.nonzero(keep)[0]:
                st = bool(k == r - 1 and gap[j] > tol and margin[j] > tol)
                if only_stable and not st:
                    continue
                out.append(dict(mask=int(masks[ii[j]]), S=idx[ii[j]].tolist(), h=H[j], c=float(ev[ii[j], k]),
                                margin=float(margin[j]), gap=float(gap[j]), stable=st))
    return out


# ----------------------------------------------------------------------------- threshold graph
def threshold_graph(Sigma, theta):
    """Components and maximal cliques of G_theta = {a != b: Sigma_ab > theta} (all B vertices).
    Returns dict(components=[masks], cliques=[masks]) (cliques maximal by inclusion; an isolated
    vertex is a clique of size 1)."""
    B = len(Sigma)
    adj = np.zeros(B, np.int64)
    for a in range(B):
        for b in range(B):
            if a != b and Sigma[a, b] > theta:
                adj[a] |= 1 << b
    allm = np.arange(1, 1 << B, dtype=np.int64)
    isc = np.ones(len(allm), bool)
    for a in range(B):
        bit = np.int64(1 << a)
        has = (allm & bit) != 0
        bad = (allm & ~adj[a] & ~bit) != 0
        isc &= ~(has & bad)
    cl = set(allm[isc].tolist())
    maximal = []
    for m in cl:
        if not any(((m | (1 << b)) in cl) for b in range(B) if not (m >> b) & 1):
            maximal.append(m)
    # components
    seen, comps = 0, []
    for a in range(B):
        if (seen >> a) & 1:
            continue
        comp, frontier = 1 << a, 1 << a
        while frontier:
            new = 0
            for b in mask_list(frontier):
                new |= int(adj[b])
            frontier = new & ~comp
            comp |= new
        seen |= comp
        comps.append(comp)
    return dict(components=sorted(comps), cliques=sorted(maximal))


def predicted_positive(Sigma, masks, theta):
    """Keep only sets whose A_SS has a positive top eigenvalue (needed for any part)."""
    out = []
    for m in masks:
        S = mask_list(m)
        if np.linalg.eigvalsh(Sigma[np.ix_(S, S)] - theta)[-1] > TOL:
            out.append(m)
    return sorted(out)


def compare(Sigma, theta):
    parts = enumerate_fast(Sigma, theta)
    st = sorted(p["mask"] for p in parts if p["stable"])
    g = threshold_graph(Sigma, theta)
    comps = predicted_positive(Sigma, g["components"], theta)
    cliq = predicted_positive(Sigma, g["cliques"], theta)
    return parts, st, comps, cliq


def diagram(name, fam, thetas):
    Sigma = fam["Sigma"]
    B = len(Sigma)
    rows = []
    t0 = time.time()
    for th in thetas:
        parts, st, comps, cliq = compare(Sigma, th)
        stab = sorted([p for p in parts if p["stable"]], key=lambda p: -p["c"])
        covered = 0
        for p in stab:
            covered |= p["mask"]
        rows.append(dict(
            theta=th,
            parts=[dict(S=p["S"], c=p["c"], margin=p["margin"], gap=p["gap"] if np.isfinite(p["gap"]) else None,
                        h=[float(f"{p['h'][b]:.3g}") for b in p["S"]]) for p in stab],
            n_parts_all=len(parts), n_stable=len(st),
            n_covered=bin(covered).count("1"),
            components=[mask_list(m) for m in comps], cliques=[mask_list(m) for m in cliq],
            eq_components=st == comps, eq_cliques=st == cliq,
            stable_not_clique=[mask_list(m) for m in st if m not in cliq],
            clique_not_stable=[mask_list(m) for m in cliq if m not in st]))
    print(f"diagram {name}: {len(thetas)} thetas {time.time() - t0:.0f}s; "
          f"eq_cliques {sum(r['eq_cliques'] for r in rows)}/{len(rows)}, eq_components {sum(r['eq_components'] for r in rows)}/{len(rows)}",
          flush=True)
    return dict(name=name, desc=fam["desc"], kind=fam["kind"], B=B, Sigma=Sigma, kappa0=float(Sigma.mean()),
                variances=np.diag(Sigma), Sigma_eigenvalues=np.linalg.eigvalsh(Sigma)[::-1], rows=rows)


def theta_grid(Sigma, n=200):
    hi = float(np.diag(Sigma).max())
    return list(np.round((np.arange(n) + 0.5) / n * hi, 6))


# ----------------------------------------------------------------------------- jobs
def selftest():
    F = sigma_families()
    for name in ("hier16", "block12", "ring16", "hier16_noise0.03_s0", "hier16_drd"):
        S = F[name]["Sigma"]
        for th in (0.07, 0.3, 0.52, 0.77):
            A, Bp = cm.enumerate_parts(S, th), enumerate_fast(S, th)
            a = sorted((tuple(p["S"]), round(p["c"], 8)) for p in A if p["stable"])
            b = sorted((tuple(p["S"]), round(p["c"], 8)) for p in Bp if p["stable"])
            # all-part counts can differ only through degenerate eigenspaces (basis choice of eigh)
            print(name, th, "stable", len(a), len(b), a == b, "all", len(A), len(Bp), flush=True)
            assert a == b


def _diag_job(args):
    name, fam = args
    return diagram(name, fam, theta_grid(fam["Sigma"]))


def edges_job():
    """Edge cases: theta exactly at, just below and above every off-diagonal level and every
    variance, for the ultrametric families."""
    F = sigma_families()
    out = []
    for name, fam in F.items():
        if fam["kind"] not in ("ultra", "ultra_var", "drd", "nonultra_ultra"):
            continue
        S = fam["Sigma"]
        lv = np.unique(np.round(np.concatenate([S[np.triu_indices(len(S), 1)], np.diag(S)]), 12))
        for L in lv:
            for d, lab in ((-1e-7, "below"), (0.0, "at"), (1e-7, "above")):
                th = float(L + d)
                if th <= 0:
                    continue
                parts, st, comps, cliq = compare(S, th)
                stp = [p for p in parts if p["stable"]]
                near = [p for p in parts if not p["stable"] and p["S"] and
                        (p["margin"] < 1e-6 or (p["gap"] < 1e-6))]
                out.append(dict(family=name, level=float(L), where=lab, theta=th, n_stable=len(st),
                                stable=[mask_list(m) for m in st], cliques=[mask_list(m) for m in cliq],
                                eq_cliques=st == cliq, n_parts_all=len(parts),
                                n_marginal=len(near),
                                marginal_examples=[dict(S=p["S"], margin=p["margin"], gap=p["gap"]) for p in near[:6]]))
        print("edges", name, flush=True)
    return out


def random_job(seed):
    """Falsification sweep: random ultrametric trees (unit variance) and random ultrametric
    matrices with unequal diagonal, on a 120-point theta grid."""
    rng = np.random.default_rng(1000 + seed)
    B = int(rng.integers(8, 15))
    S = random_ultra(rng, B)
    res = dict(seed=seed, B=B, unit=dict(n=0, fail=[]), var=dict(n=0, fail=[]))
    Sv = S.copy()
    np.fill_diagonal(Sv, S.max(1, initial=0, where=~np.eye(B, dtype=bool)) + rng.uniform(0.0, 0.8, B))
    for key, M in (("unit", S), ("var", Sv)):
        for th in (np.arange(120) + 0.5) / 120 * float(np.diag(M).max()):
            _, st, comps, cliq = compare(M, float(th))
            res[key]["n"] += 1
            if not (st == cliq == comps):
                res[key]["fail"].append(dict(theta=float(th), stable=[mask_list(m) for m in st],
                                             cliques=[mask_list(m) for m in cliq]))
    print(f"random {seed} B={B}: unit fails {len(res['unit']['fail'])}, var fails {len(res['var']['fail'])}", flush=True)
    return res


def main(which):
    import multiprocessing as mp
    RAW.mkdir(parents=True, exist_ok=True)
    if which in ("selftest", "all"):
        selftest()
    if which in ("diagrams", "all"):
        F = sigma_families()
        with mp.get_context("fork").Pool(int(os.environ.get("P2_PROCS", "4"))) as p:
            diags = p.map(_diag_job, list(F.items()), chunksize=1)
        cm.write_json(RAW / "diagrams_full.json", dict(diagrams=diags), sig=6)
    if which in ("edges", "all"):
        cm.write_json(RAW / "edges.json", dict(edges=edges_job()), sig=8)
    if which in ("random", "all"):
        with mp.get_context("fork").Pool(int(os.environ.get("P2_PROCS", "4"))) as p:
            rr = p.map(random_job, range(int(os.environ.get("P2_NRAND", "24"))), chunksize=1)
        cm.write_json(RAW / "random_ultra.json", dict(runs=rr), sig=6)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
