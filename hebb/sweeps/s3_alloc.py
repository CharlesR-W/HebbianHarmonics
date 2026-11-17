"""Q2: allocation — how many neurons end up in each part, and why.

A. gamma=0 / N->inf limit (vectorised independent neurons; exact basin measure of the
   nonnegative-PCA flow under the init distribution). Independent inputs with variances v_b
   (Sigma = diag v), theta grid x init (scale and shape). Fit n_b ∝ v_b^beta.
B. Planted family (network, N=150, paired seeds): 4 blocks of 3 inputs (within corr 0.6, across 0);
   B1 vary block-0 variance v0; B2 vary block-0 size m0 at unit variance; B3 vary m0 at matched
   top eigenvalue. gamma in {0,1}, theta in {0.05, 0.15, 0.25}. Learned clusters are matched to
   planted supports (>= 80% of centroid mass on one block).
C. Descriptive: powerlaw_pos12 / ring16 — every learned cluster's size vs its energy measures.

Output: site/data/sweeps/alloc.json (+ alloc_limit.json for part A).
"""
import numpy as np

from hebb.sweeps.common import (Timer, cluster_embed, cov_block, cov_powerlaw_pos, cov_ring, pmap, save_json,
                                simulate)

N = 150
SEEDS = list(range(8))
RHO, M3 = 0.6, 3


# ----------------------------------------------------------------------------- A: gamma=0 limit
def limit_flow(Sigma, theta, W0, dt=0.05, T=4000, wmin=1e-6):
    W = W0.copy()
    for _ in range(T):
        W = np.maximum(W, wmin)
        H = W / W.sum(1, keepdims=True)
        W = W + dt * (H @ Sigma - theta - W)
    W = np.maximum(W, wmin)
    return W / W.sum(1, keepdims=True)


def init_draw(rng, M, B, shape, scale):
    if shape == "exp":
        return rng.exponential(scale, (M, B))
    if shape == "uniform":
        return rng.uniform(0, 2 * scale, (M, B))
    if shape == "lognormal":  # sigma=1, mean = scale
        return scale * np.exp(rng.standard_normal((M, B)) - 0.5)
    if shape == "halfnormal":
        return scale * np.sqrt(np.pi / 2) * np.abs(rng.standard_normal((M, B)))
    raise ValueError(shape)


def fit_beta(v, n):
    m = n > 0
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(v[m]), np.log(n[m]), 1)[0])


def jobA(args):
    v, theta, shape, scale, M, seed = args
    rng = np.random.default_rng(seed)
    S = np.diag(v)
    H = limit_flow(S, theta, init_draw(rng, M, len(v), shape, scale))
    win = H.argmax(1)
    pure = H.max(1) > 0.9
    n = np.bincount(win[pure], minlength=len(v)).astype(float)
    return dict(theta=theta, shape=shape, scale=scale, frac=n / M, other=1 - n.sum() / M,
                beta=fit_beta(v, n))


def partA():
    v = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    thetas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]
    jobs = []
    for th in thetas:
        for shape in ("exp", "uniform", "lognormal"):
            for sc in ((0.02, 0.2, 2.0) if shape == "exp" else (0.2,)):
                jobs.append((v, th, shape, sc, 3000, 0))
    with Timer(f"A ({len(jobs)})"):
        R = pmap(jobA, jobs)
    # argmax law (decision at t=0 by argmax v_b w_b(0)); exact for Exp via Monte Carlo
    rng = np.random.default_rng(1)
    E = rng.exponential(1, (200000, len(v)))
    amax = np.bincount((E * v).argmax(1), minlength=len(v)) / len(E)
    meta = dict(script="hebb/sweeps/s3_alloc.py (part A)",
                description="gamma=0 (no lateral plasticity), N->inf limit: fraction of neurons converging to each "
                            "input of Sigma=diag(v), as a function of theta and of the init distribution of W_IB. "
                            "beta = log-log slope of fraction vs variance over inputs that receive neurons.",
                v=v, thetas=thetas, M=3000, seed=0, dt=0.05, T=4000, wmin=1e-6,
                shapes=dict(exp="W ~ Exp(scale)", uniform="W ~ U(0, 2*scale)", lognormal="W = scale*exp(Z - 1/2)"),
                argmax_law="fraction choosing b if decision = argmax_b v_b W_b(0) with W ~ Exp: " + str(np.round(amax, 4).tolist()),
                part_exists="input b supports a part only if c_b = v_b - theta > 0")
    save_json("alloc_limit", dict(rows=R, argmax_law=amax, argmax_beta=fit_beta(v, amax)), meta)
    for r in R:
        print(f"A th={r['theta']:.2f} {r['shape']:9s} sc={r['scale']:<4} beta={r['beta']:.2f} frac={np.round(r['frac'], 3)} other={r['other']:.3f}")
    print("argmax law", np.round(amax, 3), "beta", fit_beta(v, amax))


# ----------------------------------------------------------------------------- B: planted family
def planted(var0=1.0, m0=3, var_other=1.0):
    sizes = [m0, M3, M3, M3]
    S, lab = cov_block(sizes, RHO, 0.0, var=[var0, var_other, var_other, var_other])
    return S, lab


def assign(cents, lab, K=4, thr=0.8):
    """Planted block of each learned centroid (>= thr of mass on one block) else -1."""
    mass = np.array([[c[lab == k].sum() for k in range(K)] for c in cents])
    a = mass.argmax(1)
    return np.where(mass.max(1) >= thr, a, -1)


def jobB(args):
    fam, x, theta, gamma, seed, S, lab, kw = args
    r = simulate(S, N, theta, gamma=gamma, seed=seed, wmin=kw.get("wmin", 1e-4), Tmax=30000,
                 ii_scale=kw.get("ii_scale", 0.0), ib_scale=kw.get("ib_scale", 0.2))
    l, sizes, cents = cluster_embed(r["H"], 0.1)
    dead = r["Wib"].max(1) <= 1.5 * kw.get("wmin", 1e-4)
    a = assign(cents, lab)
    n = np.zeros(4)
    nother = 0
    for k in range(len(sizes)):
        if dead[l == k].all():
            continue
        if a[k] >= 0:
            n[a[k]] += sizes[k]
        else:
            nother += sizes[k]
    return dict(fam=fam, x=x, theta=theta, gamma=gamma, seed=seed, n=n, other=nother, dead=int(dead.sum()),
                res=r["res"], K=int(len(sizes)))


def partB():
    jobs = []
    v0s = [0.5, 0.7, 1.0, 1.4, 2.0, 2.8]
    m0s = [1, 2, 3, 4, 6]
    lam_ref = 1 + (M3 - 1) * RHO
    for th in (0.05, 0.15, 0.25):
        for g in (0.0, 1.0):
            for s in SEEDS:
                for v0 in v0s:
                    S, lab = planted(var0=v0)
                    jobs.append(("var", v0, th, g, s, S, lab, {}))
                for m0 in m0s:
                    S, lab = planted(m0=m0)
                    jobs.append(("size", m0, th, g, s, S, lab, {}))
                    S, lab = planted(m0=m0, var0=lam_ref / (1 + (m0 - 1) * RHO))
                    jobs.append(("size_eqlam", m0, th, g, s, S, lab, {}))
    with Timer(f"B ({len(jobs)})"):
        R = pmap(jobB, jobs)
    out = []
    for fam, xs in (("var", v0s), ("size", m0s), ("size_eqlam", m0s)):
        for th in (0.05, 0.15, 0.25):
            for g in (0.0, 1.0):
                rr = [r for r in R if r["fam"] == fam and r["theta"] == th and r["gamma"] == g]
                frac0 = np.array([[next(r["n"][0] for r in rr if r["x"] == x and r["seed"] == s) / N for x in xs] for s in SEEDS])
                fref = np.array([[np.mean(next(r["n"][1:] for r in rr if r["x"] == x and r["seed"] == s)) / N for x in xs] for s in SEEDS])
                missing = np.array([[int((next(r["n"] for r in rr if r["x"] == x and r["seed"] == s) == 0).sum()) for x in xs] for s in SEEDS])
                other = np.array([[next(r["other"] + r["dead"] for r in rr if r["x"] == x and r["seed"] == s) / N for x in xs] for s in SEEDS])
                # planted energies of block 0
                if fam == "var":
                    lam0 = np.array(xs) * lam_ref
                elif fam == "size":
                    lam0 = 1 + (np.array(xs) - 1) * RHO
                else:
                    lam0 = np.full(len(xs), lam_ref)
                sizes0 = np.array(xs if fam != "var" else [M3] * len(xs), float)
                c0 = lam0 - sizes0 * th
                # pooled slope of log(frac0/fref) vs log(x) using seeds where both >0
                ratio = np.where((frac0 > 0) & (fref > 0), frac0 / np.maximum(fref, 1e-12), np.nan)
                betas = [np.polyfit(np.log(np.array(xs, float)[np.isfinite(rw)]), np.log(rw[np.isfinite(rw)]), 1)[0]
                         if np.isfinite(rw).sum() >= 3 else np.nan for rw in ratio]
                out.append(dict(family=fam, theta=th, gamma=g, x=xs, lam0=lam0, c0=c0,
                                frac0=frac0, fref=fref, missing=missing, other=other,
                                frac0_mean=frac0.mean(0), frac0_sd=frac0.std(0), fref_mean=fref.mean(0),
                                beta_per_seed=betas, beta_mean=float(np.nanmean(betas)) if np.isfinite(betas).any() else None,
                                beta_sd=float(np.nanstd(betas)) if np.isfinite(betas).any() else None,
                                res_max=float(max(r["res"] for r in rr))))
                print(f"B {fam:10s} th={th:.2f} g={g:.0f} frac0={np.round(frac0.mean(0), 3)} ±{np.round(frac0.std(0), 3)} "
                      f"fref={np.round(fref.mean(0), 3)} beta={out[-1]['beta_mean']} miss={missing.mean(0)} other={np.round(other.mean(0), 2)} res={out[-1]['res_max']:.1e}")
    return out


# ----------------------------------------------------------------------------- C: descriptive
def jobC(args):
    name, S, theta, gamma, seed = args
    r = simulate(S, N, theta, gamma=gamma, seed=seed, wmin=1e-4, Tmax=30000)
    l, sizes, cents = cluster_embed(r["H"], 0.1)
    dead = r["Wib"].max(1) <= 1.5e-4
    rows = []
    for k in range(len(sizes)):
        if dead[l == k].all():
            continue
        h = cents[k]
        supp = np.nonzero(h > 0.02)[0]
        E = float(h @ S @ h)
        c = float((E - theta) / (h @ h))
        lt = float(np.linalg.eigvalsh(S[np.ix_(supp, supp)])[-1]) if len(supp) else 0.0
        rows.append([int(sizes[k]), E, c, lt, int(len(supp))])
    return dict(cov=name, theta=theta, gamma=gamma, seed=seed, clusters=rows, res=r["res"])


def partC():
    C = {"powerlaw_pos12": cov_powerlaw_pos(12, 1.0, seed=0), "ring16": cov_ring(16, 1.5)}
    jobs = [(n, S, th, g, s) for n, S in C.items() for th in (0.1, 0.2, 0.3, 0.5) for g in (0.0, 1.0) for s in range(4)]
    with Timer(f"C ({len(jobs)})"):
        R = pmap(jobC, jobs)
    fits = []
    for n in C:
        for th in (0.1, 0.2, 0.3, 0.5):
            for g in (0.0, 1.0):
                rows = [c for r in R if r["cov"] == n and r["theta"] == th and r["gamma"] == g for c in r["clusters"] if c[0] >= 2]
                if len(rows) >= 3:
                    a = np.array(rows, float)
                    fit = {}
                    for j, key in ((1, "E"), (2, "c"), (3, "lam_top")):
                        m = a[:, j] > 0
                        if m.sum() >= 3 and np.ptp(np.log(a[m, j])) > 1e-6:
                            sl, ic = np.polyfit(np.log(a[m, j]), np.log(a[m, 0]), 1)
                            rr = np.corrcoef(np.log(a[m, j]), np.log(a[m, 0]))[0, 1]
                            fit[key] = dict(beta=float(sl), r=float(rr))
                    fits.append(dict(cov=n, theta=th, gamma=g, fit=fit, n_clusters=len(rows)))
                    print("C", n, th, g, {k: (round(v['beta'], 2), round(v['r'], 2)) for k, v in fit.items()}, len(rows))
    return R, fits


def limit_flow_general(Sigma, theta, W0, rho="linear", par=1.0, dt=0.05, T=4000, wmin=1e-6, batch=None, seed=0,
                       snaps=()):
    """gamma=0 limit with a general weight-only regularizer and optional minibatch noise.
    dw_b = (Sigma_hat h)_b - theta + r(w_b), r: linear -w; power -w^par; softfloor -w + par/w."""
    rng = np.random.default_rng(seed)
    W = W0.copy()
    B = Sigma.shape[0]
    Lc = np.linalg.cholesky(Sigma)
    out = {}
    for t in range(1, T + 1):
        W = np.maximum(W, wmin)
        H = W / W.sum(1, keepdims=True)
        if batch:
            # independent minibatch per neuron would be wrong (all neurons see the same stimuli):
            g = rng.standard_normal((batch, B)) @ Lc.T
            g -= g.mean(0)
            S = g.T @ g / (batch - 1)
        else:
            S = Sigma
        if rho == "linear":
            r = -W
        elif rho == "power":
            r = -W ** par
        elif rho == "softfloor":
            r = -W + par / W
        W = W + dt * (H @ S - theta + r)
        if t in snaps:
            Wc = np.maximum(W, wmin)
            out[t] = Wc / Wc.sum(1, keepdims=True)
    W = np.maximum(W, wmin)
    out[T] = W / W.sum(1, keepdims=True)
    return out


def jobD(args):
    tag, v, theta, kw = args
    rng = np.random.default_rng(0)
    M = 2000
    if "logsig" in kw:
        s = kw["logsig"]
        W0 = 0.2 * np.exp(s * rng.standard_normal((M, len(v))) - s * s / 2)
    else:
        W0 = rng.exponential(0.2, (M, len(v)))
    T = kw.get("T", 4000)
    snaps = kw.get("snaps", ())
    res = limit_flow_general(np.diag(v), theta, W0, kw.get("rho", "linear"), kw.get("par", 1.0), T=T,
                             batch=kw.get("batch"), snaps=snaps, dt=kw.get("dt", 0.05))
    rows = {}
    for t, H in res.items():
        win = H.argmax(1); pure = H.max(1) > 0.8
        n = np.bincount(win[pure], minlength=len(v)).astype(float)
        rows[t] = dict(frac=n / M, beta=fit_beta(v, n), other=1 - n.sum() / M)
    return dict(tag=tag, theta=theta, kw={k: x for k, x in kw.items() if k != "snaps"}, by_time=rows)


def partD():
    v = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    jobs = []
    for th in (0.1, 0.2, 0.3):
        for s in (0.25, 0.5, 1.0, 2.0, 3.0):
            jobs.append(("init_logsigma", v, th, dict(logsig=s)))
        for p in (0.5, 2.0, 3.0):
            jobs.append(("rho_power", v, th, dict(rho="power", par=p)))
        for e in (1e-3, 1e-2, 3e-2):
            jobs.append(("rho_softfloor", v, th, dict(rho="softfloor", par=e)))
        for b in (8, 32, 128):
            jobs.append(("noise_batch", v, th, dict(batch=b, T=40000, snaps=(4000, 10000, 20000), dt=0.05)))
    with Timer(f"D ({len(jobs)})"):
        R = pmap(jobD, jobs)
    for r in R:
        print("D", r["tag"], r["theta"], r["kw"], {t: (round(x["beta"], 2), round(x["other"], 3)) for t, x in r["by_time"].items()})
    meta = dict(script="hebb/sweeps/s3_alloc.py (part D)",
                description="gamma=0 limit (independent neurons, M=2000), Sigma=diag(v): which ingredient changes the "
                            "allocation exponent beta (fraction ∝ v^beta)? init log-weight spread (W = 0.2 exp(sZ - s^2/2)), "
                            "nonlinear weight-only regularizers, minibatch noise (all neurons see the same minibatch; "
                            "'by_time' = steps).",
                v=v, M=2000, dt=0.05, wmin=1e-6,
                rho=dict(linear="-w", power="-w^p", softfloor="-w + eps/w"))
    save_json("alloc_ingredients", dict(rows=R), meta)


def main():
    partA()
    partD()
    B = partB()
    Cr, Cf = partC()
    meta = dict(script="hebb/sweeps/s3_alloc.py", N=N, seeds=SEEDS,
                params=dict(lam=1, eta=1, dt=0.05, wmin=1e-4, ii_scale=0.0, ib_scale=0.2, p=1.0, conv_tol=1e-5, Tmax=30000),
                planted=dict(blocks="4 blocks: block 0 has size m0 (default 3) and variance v0 (default 1); blocks 1-3 size 3 "
                                    "variance 1; within-block corr 0.6, across 0",
                             families=dict(var="vary v0", size="vary m0 at unit variance",
                                           size_eqlam="vary m0 with v0 = 2.2/(1+0.6(m0-1)) so block-0 top eigenvalue stays 2.2"),
                             match="learned cluster assigned to planted block if >= 80% of centroid mass on it; dead neurons and "
                                   "unassigned clusters count as 'other'",
                             fields=dict(frac0="fraction of N in block 0 [seed][x]", fref="mean fraction per other block",
                                         missing="number of planted blocks with zero neurons", lam0="planted top eigenvalue of block 0",
                                         c0="lam0 - m0*theta = supra-threshold eigenvalue (part exists iff > 0 and theta < 0.6*v0)",
                                         beta="per seed slope of log(frac0/fref) vs log(x)")),
                descriptive="clusters: [size, E=h^T Sigma h, c=(E-theta)/|h|^2, top eigenvalue of Sigma on support, |support|]")
    save_json("alloc", dict(planted=B, descriptive=Cr, descriptive_fits=Cf), meta)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        {"A": partA, "D": partD, "B": partB, "C": partC}[sys.argv[1]]()
    else:
        main()
