"""Q (follow-up): allocation under PRIVATE (per-synapse, independent) noise.

Noise models (the regularizer stays rho(w) = w0 - lam*w; noise is added on top, never a regularizer):
  add  : additive white noise per synapse, dw = drift dt + sigma sqrt(dt) xi, then clip at wmin.
  mul  : multiplicative, w <- w exp(sigma sqrt(dt) xi) after the drift step, then clip at wmin.
  inp  : private covariance-estimation noise. Each synapse (x,y) estimates Cov(u_x,u_y) from its OWN
         Gaussian samples, n = 1/sigma^2 samples per unit time (time unit 1/lam). Gaussian (Isserlis)
         approximation of the estimation error: Var = sigma^2 dt (C_xx C_yy + C_xy^2) per step, independent
         across synapses. I-B: C_xx = q_i = h_i'Sigma h_i, C_yy = Sigma_bb, C_xy = (Sigma h_i)_b.
  inpn : as inp but one private sample stream per NEURON (noise correlated across a neuron's synapses):
         xi = sigma sqrt(dt) (sqrt(q) L z + (Sigma h) z'), L L' = Sigma.  Robustness check only.

Parts:
  A  gamma=0 lone-neuron limit (no lateral synapses), Sigma = diag(v), v = (.5,.75,1,1.5,2,3).
     All (theta, sigma, init) conditions are stacked as independent rows of one array.
     Inits: 'weak' (all at the weakest vertex's fixed point), 'strong' (all at v=3), 'rand' (Exp(0.2)).
     Records allocation vs time, core occupancy, transit, blur, and a milestoning rate matrix
     (label = last core visited; core = cos(h, part) >= 0.9).
  R  Kramers: rates out of each vertex vs sigma (start M neurons at each vertex), stationary law of the
     measured rate matrix, Arrhenius fits.
  M  multiplicative noise: floor pinning, escape vs 2 theta/(sigma^2 wmin).
  P  planted family of F6 (4 blocks of 3, within 0.6), block-0 variance or size varied, lone limit.
  N  networks (N=120, gamma in {0,1}), noise on all synapses vs I-B only; allocation from different inits,
     hopping, within-cluster diversity (instantaneous vs time-averaged spread, voltage correlations).

  X  extreme noise (sigma up to 5), DT step-size check, L theory checks (Lyapunov, curl, floor lift).

Usage: PYTHONPATH=. uv run python -u -m hebb.sweeps.s8_private_noise {A,R,M,P,X,DT,L,N,assemble}
       (prefix S8_MODAL=1 and use a Python with `modal` installed to run the units on Modal CPU containers;
        all parts ran that way on 2026-11-16, ~1-5 min each; locally on 4 free CPUs expect ~10-30 min per part)
Each part writes scratch/out/sweeps/s8_<part>.json; 'assemble' writes site/data/sweeps/alloc_private_noise.json.
"""
import json
import sys

import numpy as np

from hebb import core
from hebb.core import Params, harmonic, init_state
from hebb.sweeps.common import RAW, Timer, _clean, cluster_embed, cov_block, pmap, save_json

V6 = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
DT = 0.05
WMIN = 1e-4
CORE = 0.9          # cos(h, part) >= CORE  =>  neuron is "in" that part
SNAPS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000)


# ----------------------------------------------------------------------------- lone-neuron engine
def noise_step(W, H, SH, Sd, kind, sig, sq, rng, L=None):
    """Apply private noise to W (rows = neurons). sig: per-row (M,1) amplitudes."""
    if kind == "add":
        return W + sig * sq * rng.standard_normal(W.shape)
    if kind == "mul":
        return W * np.exp(sig * sq * rng.standard_normal(W.shape))
    if kind == "inp":
        q = (H * SH).sum(1, keepdims=True)
        return W + sig * sq * np.sqrt(np.maximum(q * Sd + SH ** 2, 0)) * rng.standard_normal(W.shape)
    if kind == "inpn":
        q = (H * SH).sum(1, keepdims=True)
        z = rng.standard_normal(W.shape) @ L.T
        return W + sig * sq * (np.sqrt(q) * z + SH * rng.standard_normal((len(W), 1)))
    if kind == "none":
        return W
    raise ValueError(kind)


def lone_run(Sigma, parts, theta, sig, group, G, W0, T, kind, seed=0, wmin=WMIN, dt=DT, snaps=SNAPS,
             check=10, meas_from=None, diag=None):
    """Independent lone neurons (rows). theta, sig: per-row arrays; group: per-row int in [0,G).
    parts: (K,B) part embeddings (rows sum to 1). Returns per-group time series and rate counts.
    Milestoning: label = last core part visited (-1 before first core visit)."""
    rng = np.random.default_rng(seed)
    M, B = W0.shape
    K = len(parts)
    Pn = parts / np.linalg.norm(parts, axis=1, keepdims=True)
    th = np.asarray(theta, float)[:, None]
    sg = np.asarray(sig, float)[:, None]
    W = W0.copy()
    Sd = np.diag(Sigma)[None, :]
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(B)) if kind == "inpn" else None
    isdiag = diag if diag is not None else np.allclose(Sigma, np.diag(np.diag(Sigma)))
    sq = np.sqrt(dt)
    nsteps = int(round(T / dt))
    meas_from = T / 2 if meas_from is None else meas_from

    def classify(W):
        H = W / W.sum(1, keepdims=True)
        cs = (H @ Pn.T) / np.linalg.norm(H, axis=1, keepdims=True)
        a = cs.argmax(1)
        cmax = cs[np.arange(len(a)), a]
        return H, a, cmax

    H, a, cmax = classify(W)
    label = np.where(cmax >= CORE, a, -1)
    snap_steps = {int(round(s / dt)): s for s in snaps if s <= T}
    ts = []
    alloc_t, core_t, transit_t, off_t, cos_t, dead_t = [], [], [], [], [], []
    acc = dict(alloc=np.zeros(G * K), core=np.zeros(G * K), transit=np.zeros(G), cos=np.zeros(G),
               off=np.zeros(G), dead=np.zeros(G), n=0)
    cnt = np.zeros(G * K * K)      # transitions k->l (milestoning)
    tau = np.zeros(G * K)          # time with label k
    gK = group * K
    gsize = np.bincount(group, minlength=G).astype(float)
    cos_part_acc = np.zeros((G, K, B))   # summed h of neurons in core k (for cluster-mean cosine)
    for t in range(1, nsteps + 1):
        W = np.maximum(W, wmin)
        H = W / W.sum(1, keepdims=True)
        SH = H * Sd if isdiag else H @ Sigma
        W = W + dt * (SH - th - W)
        W = noise_step(W, H, SH, Sd, kind, sg, sq, rng, L)
        W = np.maximum(W, wmin)
        snap = t in snap_steps
        meas = t * dt > meas_from
        if t % check == 0 or snap:
            H, a, cmax = classify(W)
            incore = cmax >= CORE
            newlab = np.where(incore, a, label)
            if meas and t % check == 0:
                ch = (newlab != label) & (label >= 0)
                if ch.any():
                    cnt += np.bincount((gK[ch] + label[ch]) * K + newlab[ch], minlength=G * K * K)
                ok = newlab >= 0
                tau += np.bincount(gK[ok] + newlab[ok], minlength=G * K) * (check * dt)
            label = newlab
            if snap or (meas and t % (check * 10) == 0):
                al = np.bincount(gK + a, minlength=G * K) / np.repeat(gsize, K)
                co = np.bincount(gK[incore] + a[incore], minlength=G * K) / np.repeat(gsize, K)
                tr = 1 - np.bincount(group[incore], minlength=G) / gsize
                # off-part mass: mass of h outside the support of its nearest part
                offm = np.bincount(group, weights=(H * (parts[a] <= 1e-9)).sum(1), minlength=G) / gsize
                cs_ = np.bincount(group, weights=cmax, minlength=G) / gsize
                dead = np.bincount(group, weights=(W.max(1) < 0.05).astype(float), minlength=G) / gsize
                if snap:
                    ts.append(snap_steps[t]); alloc_t.append(al); core_t.append(co); transit_t.append(tr)
                    off_t.append(offm); cos_t.append(cs_); dead_t.append(dead)
                if meas and t % (check * 10) == 0:
                    acc["alloc"] += al; acc["core"] += co; acc["transit"] += tr; acc["off"] += offm
                    acc["cos"] += cs_; acc["dead"] += dead; acc["n"] += 1
                    for k in range(K):
                        m = incore & (a == k)
                        if m.any():
                            np.add.at(cos_part_acc[:, k], group[m], H[m])
    n = max(acc["n"], 1)
    # cosine between the (time- and member-) averaged embedding of core-k neurons and the noiseless part
    cm = np.full((G, K), np.nan)
    for gi in range(G):
        for k in range(K):
            x = cos_part_acc[gi, k]
            if x.sum() > 0:
                cm[gi, k] = x @ Pn[k] / np.linalg.norm(x)
    return dict(t=ts, alloc_t=np.array(alloc_t).reshape(len(ts), G, K).transpose(1, 0, 2),
                core_t=np.array(core_t).reshape(len(ts), G, K).transpose(1, 0, 2),
                transit_t=np.array(transit_t).T, off_t=np.array(off_t).T, cos_t=np.array(cos_t).T,
                dead_t=np.array(dead_t).T,
                alloc=(acc["alloc"] / n).reshape(G, K), core=(acc["core"] / n).reshape(G, K),
                transit=acc["transit"] / n, off=acc["off"] / n, cos=acc["cos"] / n, dead=acc["dead"] / n,
                cnt=cnt.reshape(G, K, K), tau=tau.reshape(G, K), cos_mean=cm, W=W)


def vertex_init(M, B, k, c, wmin=WMIN):
    W = np.full((M, B), wmin)
    W[:, k] = max(c, 0.05)
    return W


def part_init(M, part, c, wmin=WMIN):
    W = np.full((M, len(part)), wmin)
    W += np.outer(np.ones(M), part) * max(c, 0.05) * (part > 0)
    return W


def rate_matrix(cnt, tau):
    """Q_kl = cnt_kl / tau_k (rows with tau=0 -> 0); stationary distribution of Q."""
    K = len(tau)
    Q = np.where(tau[:, None] > 0, cnt / np.maximum(tau[:, None], 1e-300), 0.0)
    np.fill_diagonal(Q, 0)
    Qg = Q - np.diag(Q.sum(1))
    # stationary: pi Qg = 0, sum pi = 1 (least squares; fine for irreducible chains)
    Aeq = np.vstack([Qg.T, np.ones(K)])
    b = np.r_[np.zeros(K), 1.0]
    pi = np.linalg.lstsq(Aeq, b, rcond=None)[0]
    ev = np.sort(np.real(np.linalg.eigvals(Qg)))
    relax = float(1 / -ev[-2]) if K > 1 and ev[-2] < -1e-12 else float("inf")
    irreducible = bool(_irreducible(Q > 0))
    return Q, pi, relax, irreducible


def _irreducible(Adj):
    K = len(Adj)
    R = np.eye(K, dtype=bool) | Adj
    for _ in range(K):
        R = R | (R.astype(int) @ R.astype(int) > 0)
    return R.all()


def fit_powerlaw(x, y, floor=0.0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = (y > floor) & (x > 0)
    if m.sum() < 3:
        return dict(beta=None, r=None, n=int(m.sum()))
    b, a = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    r = np.corrcoef(np.log(x[m]), np.log(y[m]))[0, 1]
    res = np.log(y[m]) - (a + b * np.log(x[m]))
    return dict(beta=float(b), r=float(r), rms=float(np.sqrt((res ** 2).mean())), n=int(m.sum()))


def fit_exp(x, y, floor=0.0):
    """ln y = a + x / T  (Boltzmann in x)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = y > floor
    if m.sum() < 3:
        return dict(T=None, r=None, n=int(m.sum()))
    b, a = np.polyfit(x[m], np.log(y[m]), 1)
    r = np.corrcoef(x[m], np.log(y[m]))[0, 1]
    res = np.log(y[m]) - (a + b * x[m])
    return dict(T=float(1 / b) if b != 0 else None, slope=float(b), r=float(r), rms=float(np.sqrt((res ** 2).mean())),
                n=int(m.sum()))


def _map(fn, jobs, procs=4):
    """Local multiprocessing, or Modal CPU containers when S8_MODAL=1 (see s8_modal.py)."""
    import os
    if os.environ.get("S8_MODAL"):
        from hebb.sweeps.s8_modal import modal_map
        return modal_map(fn.__name__, jobs)
    return pmap(fn, jobs, procs=procs)


def raw_save(name, obj):
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"s8_{name}.json").write_text(json.dumps(_clean(obj), separators=(",", ":")))


def raw_load(name):
    p = RAW / f"s8_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------- A: lone limit, diag Sigma
THETAS_A = (0.05, 0.2, 0.4)
SIG_A = dict(add=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7),
             inp=(0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8),
             mul=(0.3, 1.0, 2.0),
             inpn=(0.3, 0.4, 0.6))
INITS = ("weak", "strong", "rand")


def jobA(args):
    kind, M, T, thetas = args[:4]
    sigs = args[4] if len(args) > 4 else SIG_A[kind]
    v = V6
    B = len(v)
    parts = np.eye(B)
    conds = [(th, s, ini) for th in thetas for s in sigs for ini in INITS]
    rng = np.random.default_rng(123)
    W0, th_r, sg_r, grp = [], [], [], []
    for gi, (th, s, ini) in enumerate(conds):
        if ini == "rand":
            W = rng.exponential(0.2, (M, B))
        else:
            k = 0 if ini == "weak" else B - 1
            W = vertex_init(M, B, k, v[k] - th)
        W0.append(W); th_r.append(np.full(M, th)); sg_r.append(np.full(M, s)); grp.append(np.full(M, gi))
    r = lone_run(np.diag(v), parts, np.concatenate(th_r), np.concatenate(sg_r), np.concatenate(grp), len(conds),
                 np.vstack(W0), T, kind, seed=7, diag=True)
    out = []
    for gi, (th, s, ini) in enumerate(conds):
        Q, pi, relax, irr = rate_matrix(r["cnt"][gi], r["tau"][gi])
        out.append(dict(kind=kind, theta=th, sigma=s, init=ini, t=r["t"], alloc_t=r["alloc_t"][gi],
                        core_t=r["core_t"][gi], transit_t=r["transit_t"][gi], off_t=r["off_t"][gi],
                        alloc=r["alloc"][gi], core=r["core"][gi], transit=r["transit"][gi], off=r["off"][gi],
                        cos=r["cos"][gi], dead=r["dead"][gi], cos_mean=r["cos_mean"][gi],
                        hops=float(r["cnt"][gi].sum()), tau=r["tau"][gi], cnt=r["cnt"][gi], Q=Q, pi_Q=pi, relax=relax, irreducible=irr))
    return out


def partA(M=1000, T=2000):
    jobs = [(k, M, T, (th,)) for k in ("add", "inp", "mul") for th in THETAS_A] + [("inpn", M, T, (0.2,))]
    with Timer("A"):
        R = [x for rr in _map(jobA, jobs) for x in rr]
    summ = summarize_A(R, T, M)
    raw_save("A", dict(rows=R, summary=summ, M=M, T=T))
    for s in summ:
        print(f"A {s['kind']:4s} th={s['theta']:.2f} sig={s['sigma']:<5} TV(w,s)={s['tv_ws']:.3f} TV(r,s)={s['tv_rs']:.3f} "
              f"pi={np.round(s['alloc'], 3)} transit={s['transit']:.2f} off={s['off']:.2f} cos={s['cos']:.3f} "
              f"hop/neuron/t={s['hop_rate']:.2e} tmix={s['t_mix']} relax={s['relax']:.3g} beta={s['fit_pl']['beta']}",
              flush=True)


def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def summarize_A(R, T, M=1000, v=V6):
    out = []
    keys = sorted({(r["kind"], r["theta"], r["sigma"]) for r in R})
    for kind, th, s in keys:
        rr = {r["init"]: r for r in R if (r["kind"], r["theta"], r["sigma"]) == (kind, th, s)}
        w, st, ra = rr["weak"], rr["strong"], rr["rand"]
        alloc = (np.array(w["alloc"]) + np.array(st["alloc"])) / 2
        core = (np.array(w["core"]) + np.array(st["core"])) / 2
        tvs = [tv(a, b) for a, b in zip(w["alloc_t"], st["alloc_t"])]
        tmix = next((t for t, x in zip(w["t"], tvs) if x < 0.05 and all(y < 0.05 for y in tvs[w["t"].index(t):])), None)
        cnt = sum(np.array(r["cnt"]) for r in (w, st, ra))
        tau = sum(np.array(r["tau"]) for r in (w, st, ra))
        Q, pi, relax, irr = rate_matrix(cnt, tau)
        hop_rate = float(sum(r["hops"] for r in (w, st, ra)) / (3 * M * T / 2))
        out.append(dict(kind=kind, theta=th, sigma=s, alloc=alloc, core=core,
                        tv_ws=tv(w["alloc"], st["alloc"]), tv_rs=tv(ra["alloc"], st["alloc"]),
                        equilibrated=bool(tv(w["alloc"], st["alloc"]) < 0.05 and tv(ra["alloc"], st["alloc"]) < 0.05),
                        transit=float((w["transit"] + st["transit"]) / 2), off=float((w["off"] + st["off"]) / 2),
                        cos=float((w["cos"] + st["cos"]) / 2), dead=float((w["dead"] + st["dead"]) / 2),
                        cos_mean=np.nanmean(np.array([w["cos_mean"], st["cos_mean"]], float), 0),
                        hop_rate=hop_rate, t_mix=tmix, relax=relax, irreducible=irr, pi_Q=pi, Q=Q,
                        fit_pl=fit_powerlaw(v, alloc, 1e-3), fit_pl_core=fit_powerlaw(v, core, 1e-3),
                        fit_exp_v=fit_exp(v, alloc, 1e-3)))
    return out


def partX(M=1000, T=600):
    """Extreme noise (does beta -> 1 as sigma -> inf?), theta = 0.2, diag Sigma."""
    jobs = [("inp", M, T, (0.2,), (s,)) for s in (1.0, 1.5, 2.0, 3.0, 5.0)]
    jobs += [("add", M, T, (0.2,), (s,)) for s in (1.0, 1.5, 2.0, 3.0)]
    with Timer("X"):
        R = [x for rr in _map(jobA, jobs) for x in rr]
    summ = summarize_A(R, T, M)
    raw_save("X", dict(summary=summ, M=M, T=T))
    for s in summ:
        print(f"X {s['kind']} sig={s['sigma']} pi={np.round(s['alloc'], 3)} TV={s['tv_ws']:.3f} transit={s['transit']:.2f} "
              f"off={s['off']:.2f} beta={s['fit_pl']['beta']} rms={s['fit_pl']['rms']} T_exp={s['fit_exp_v']['T']} "
              f"rms_exp={s['fit_exp_v']['rms']}", flush=True)


# ----------------------------------------------------------------------------- R: Kramers rates
SIG_R = dict(add=(0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4), inp=(0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5))


def jobR(args):
    kind, th, M, T = args
    v = V6; B = len(v)
    sigs = SIG_R[kind]
    W0, sg, grp, conds = [], [], [], []
    for si, s in enumerate(sigs):
        for k in range(B):
            conds.append((s, k))
            W0.append(vertex_init(M, B, k, v[k] - th)); sg.append(np.full(M, s)); grp.append(np.full(M, len(conds) - 1))
    # measure from t = 5 (after local relaxation) to T; first exits recorded through milestoning counts
    r = lone_run(np.diag(v), np.eye(B), np.full(len(conds) * M, th), np.concatenate(sg), np.concatenate(grp),
                 len(conds), np.vstack(W0), T, kind, seed=11, diag=True, meas_from=5.0, snaps=(T,))
    out = []
    for si, s in enumerate(sigs):
        cnt = sum(r["cnt"][si * B + k] for k in range(B))
        tau = sum(r["tau"][si * B + k] for k in range(B))
        Q, pi, relax, irr = rate_matrix(cnt, tau)
        out.append(dict(kind=kind, theta=th, sigma=s, cnt=cnt, tau=tau, Q=Q, out_rate=Q.sum(1), pi_Q=pi,
                        relax=relax, irreducible=irr, off=float(np.mean([r["off"][si * B + k] for k in range(B)]))))
    return out


def partR(M=500, T=1000):
    jobs = [(k, th, M, T) for k in ("add", "inp") for th in (0.05, 0.2, 0.4)]
    with Timer("R"):
        R = [x for rr in _map(jobR, jobs) for x in rr]
    # Arrhenius fits per (kind, theta, vertex): ln rate = a - b/sigma^2  -> barrier Delta = b/2 (D = sigma^2/2)
    fits = []
    for kind in ("add", "inp"):
        for th in (0.05, 0.2, 0.4):
            rr = sorted([r for r in R if r["kind"] == kind and r["theta"] == th], key=lambda r: r["sigma"])
            s = np.array([r["sigma"] for r in rr])
            for k in range(6):
                rate = np.array([r["out_rate"][k] for r in rr])
                m = rate > 0
                if m.sum() >= 3:
                    b, a = np.polyfit(1 / s[m] ** 2, np.log(rate[m]), 1)
                    rc = np.corrcoef(1 / s[m] ** 2, np.log(rate[m]))[0, 1]
                    fits.append(dict(kind=kind, theta=th, vertex=k, v=V6[k], barrier=float(-b / 2), prefactor=float(np.exp(a)),
                                     r=float(rc), n=int(m.sum())))
                else:
                    fits.append(dict(kind=kind, theta=th, vertex=k, v=V6[k], barrier=None, n=int(m.sum())))
    raw_save("R", dict(rows=R, arrhenius=fits, M=M, T=T))
    for r in R:
        print(f"R {r['kind']} th={r['theta']} sig={r['sigma']} out_rate={np.array2string(np.asarray(r['out_rate']), precision=2)} "
              f"pi_Q={np.round(r['pi_Q'], 3)} relax={r['relax']:.3g} irr={r['irreducible']} off={r['off']:.2f}", flush=True)
    for f in fits:
        print("Arrhenius", f)


# ----------------------------------------------------------------------------- M: multiplicative pinning
def jobM(args):
    th, wmin, sigs, M, T = args
    v = V6; B = len(v)
    W0, sg, grp, conds = [], [], [], []
    for s in sigs:
        for k in (0, B - 1):
            conds.append((s, k))
            W0.append(vertex_init(M, B, k, v[k] - th, wmin)); sg.append(np.full(M, s)); grp.append(np.full(M, len(conds) - 1))
    r = lone_run(np.diag(v), np.eye(B), np.full(len(conds) * M, th), np.concatenate(sg), np.concatenate(grp), len(conds),
                 np.vstack(W0), T, "mul", seed=5, wmin=wmin, diag=True, meas_from=5.0, snaps=(T,))
    out = []
    for gi, (s, k) in enumerate(conds):
        tau = r["tau"][gi].sum()
        out.append(dict(theta=th, wmin=wmin, sigma=s, start=k, exits=float(r["cnt"][gi].sum()), time=float(tau),
                        rate=float(r["cnt"][gi].sum() / max(tau, 1e-9)), x=2 * th / (s * s * wmin),
                        alloc_end=r["alloc"][gi], off=float(r["off"][gi])))
    return out


def partM(M=500, T=500):
    jobs = []
    for th in (0.05, 0.2):
        for wmin in (1e-4, 3e-3, 1e-2, 3e-2):
            jobs.append((th, wmin, (0.5, 1.0, 1.5, 2.0, 3.0), M, T))
    with Timer("M"):
        R = [x for rr in _map(jobM, jobs) for x in rr]
    raw_save("M", dict(rows=R, M=M, T=T))
    for r in R:
        print(f"M th={r['theta']} wmin={r['wmin']} sig={r['sigma']} start={r['start']} x=2th/(s^2 wmin)={r['x']:.3g} "
              f"rate={r['rate']:.3g} off={r['off']:.2f} alloc_end={np.round(r['alloc_end'], 2)}", flush=True)


# ----------------------------------------------------------------------------- P: planted family (lone limit)
RHO = 0.6


def planted(var0=1.0, m0=3):
    S, lab = cov_block([m0, 3, 3, 3], RHO, 0.0, var=[var0, 1, 1, 1])
    parts = np.array([(lab == k) / (lab == k).sum() for k in range(4)], float)
    return S, lab, parts


def jobP(args):
    fam, kind, th, sigs, M, T = args
    xs = (0.5, 0.7, 1.0, 1.4, 2.0, 2.8) if fam == "var" else (1, 2, 3, 4, 6)
    out = []
    rng = np.random.default_rng(3)
    for x in xs:
        S, lab, parts = planted(var0=x) if fam == "var" else planted(m0=x)
        B = len(S)
        lam = np.array([np.linalg.eigvalsh(S[np.ix_(lab == k, lab == k)])[-1] for k in range(4)])
        W0, sg, grp, conds = [], [], [], []
        for s in sigs:
            for ini in ("rand", "block0", "block1"):
                conds.append((s, ini))
                if ini == "rand":
                    W = rng.exponential(0.2, (M, B))
                else:
                    k = 0 if ini == "block0" else 1
                    W = part_init(M, parts[k], lam[k] - len(np.nonzero(lab == k)[0]) * th)
                W0.append(W); sg.append(np.full(M, s)); grp.append(np.full(M, len(conds) - 1))
        r = lone_run(S, parts, np.full(len(conds) * M, th), np.concatenate(sg), np.concatenate(grp), len(conds),
                     np.vstack(W0), T, kind, seed=17, diag=False, snaps=(10, 100, T))
        for si, s in enumerate(sigs):
            g = [si * 3 + j for j in range(3)]
            alloc = np.mean([r["alloc"][j] for j in g], 0)
            core = np.mean([r["core"][j] for j in g], 0)
            out.append(dict(fam=fam, kind=kind, theta=th, sigma=s, x=x, lam=lam, alloc=alloc, core=core,
                            alloc_by_init=[r["alloc"][j] for j in g],
                            tv_max=max(tv(r["alloc"][a], r["alloc"][b]) for a in g for b in g),
                            transit=float(np.mean([r["transit"][j] for j in g])), off=float(np.mean([r["off"][j] for j in g])),
                            cos_mean=np.nanmean([r["cos_mean"][j] for j in g], 0)))
    return out


SIG_P = dict(add=(0.3, 0.4, 0.5), inp=(0.25, 0.3, 0.4, 0.6))


def partP(M=600, T=600):
    jobs = [(fam, kind, th, SIG_P[kind], M, T) for fam in ("var", "size") for kind in ("add", "inp") for th in (0.15, 0.25)]
    with Timer("P"):
        R = [x for rr in _map(jobP, jobs) for x in rr]
    fits = []
    for fam in ("var", "size"):
        for kind in ("add", "inp"):
            for th in (0.15, 0.25):
                for s in SIG_P[kind]:
                    rr = [r for r in R if (r["fam"], r["kind"], r["theta"], r["sigma"]) == (fam, kind, th, s)]
                    lam0 = np.array([r["lam"][0] for r in rr]); lref = rr[0]["lam"][1]
                    ratio = np.array([r["alloc"][0] / max(np.mean(r["alloc"][1:]), 1e-9) for r in rr])
                    ratio_c = np.array([r["core"][0] / max(np.mean(r["core"][1:]), 1e-9) for r in rr])
                    f = fit_powerlaw(lam0 / lref, ratio, 1e-3)
                    fc = fit_powerlaw(lam0 / lref, ratio_c, 1e-3)
                    fits.append(dict(fam=fam, kind=kind, theta=th, sigma=s, x=[r["x"] for r in rr], lam0=lam0,
                                     frac0=[r["alloc"][0] for r in rr], fref=[float(np.mean(r["alloc"][1:])) for r in rr],
                                     core0=[r["core"][0] for r in rr], ratio=ratio, beta=f["beta"], r=f["r"],
                                     beta_core=fc["beta"], tv_max=max(r["tv_max"] for r in rr),
                                     transit=float(np.mean([r["transit"] for r in rr])),
                                     off=float(np.mean([r["off"] for r in rr]))))
                    print(f"P {fam} {kind} th={th} sig={s} frac0={np.round(fits[-1]['frac0'], 3)} fref={np.round(fits[-1]['fref'], 3)} "
                          f"beta={f['beta']} r={f['r']} beta_core={fc['beta']} tvmax={fits[-1]['tv_max']:.3f} "
                          f"transit={fits[-1]['transit']:.2f} off={fits[-1]['off']:.2f}", flush=True)
    raw_save("P", dict(rows=R, fits=fits, M=M, T=T))


# ----------------------------------------------------------------------------- N: networks
def net_noise(Wii, Wib, Mii, Mib, H, Sigma, kind, sig, sq, rng, lateral=True, gamma=1.0):
    N, B = Wib.shape
    if kind == "add":
        Zb = sig * sq * rng.standard_normal((N, B))
        Zi = sig * sq * rng.standard_normal((N, N)) if lateral else 0.0
    elif kind == "inp":
        SH = H @ Sigma
        q = (H * SH).sum(1)
        Zb = sig * sq * np.sqrt(np.maximum(q[:, None] * np.diag(Sigma)[None] + SH ** 2, 0)) * rng.standard_normal((N, B))
        if lateral:
            C = SH @ H.T
            # lateral estimation noise carries the lateral gain gamma (eta_II = gamma eta)
            Zi = gamma * sig * sq * np.sqrt(np.maximum(np.outer(q, q) + C ** 2, 0)) * rng.standard_normal((N, N))
        else:
            Zi = 0.0
    elif kind == "none":
        return Wii, Wib
    else:
        raise ValueError(kind)
    if lateral:
        Zi = np.triu(Zi, 1); Zi = Zi + Zi.T
    Wib = np.maximum(Wib + Zb, 0) * Mib
    Wii = (Wii + Zi) * Mii
    return Wii, Wib


def jobN(args):
    (cov, theta, gamma, kind, sig, lateral, init, seed, N, T) = args
    S, lab, parts = cov
    B = len(S); K = len(parts)
    rng = np.random.default_rng(seed)
    Wii, Wib, Mii, Mib = init_state(N, B, rng, 1.0, 0.03 if init == "rand" else 0.0, 0.2)
    lamk = np.array([np.linalg.eigvalsh(S[np.ix_(lab == k, lab == k)])[-1] for k in range(K)])
    if init != "rand":
        k = int(init[-1])
        Wib = part_init(N, parts[k], lamk[k] - 3 * theta, WMIN) * Mib
        Wii = np.full((N, N), WMIN) * Mii
    P = Params(theta=theta, gamma=gamma, wmin=WMIN, dt=DT)
    sq = np.sqrt(DT)
    # independent noise stream per (seed, init) -- a shared stream synchronises runs from different inits
    nrng = np.random.default_rng([seed, 999, {"rand": 0, "block0": 1, "block3": 2}.get(init, 3)])
    Pn = parts / np.linalg.norm(parts, axis=1, keepdims=True)
    nsteps = int(round(T / DT))
    snaps = {int(round(s / DT)) for s in (1, 5, 20, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000) if s <= T}
    ts, alloc_t = [], []
    label = None; hops = 0
    sp_l, vc_l, pr_l, ncl_l, cos_l = [], [], [], [], []
    q3, q4 = [], []       # snapshots of H in 3rd / 4th quarter (persistence test)
    lab3, lab4 = [], []
    for t in range(1, nsteps + 1):
        Wii, Wib, H = core.step(Wii, Wib, Mii, Mib, S, P)
        Wii, Wib = net_noise(Wii, Wib, Mii, Mib, H, S, kind, sig, sq, nrng, lateral, gamma)
        Wii = np.maximum(Wii, WMIN) * Mii; np.fill_diagonal(Wii, 0.0)
        Wib = np.maximum(Wib, WMIN) * Mib
        if t % 20 == 0 or t in snaps:
            H = harmonic(Wii, Wib)
            cs = (H @ Pn.T) / np.linalg.norm(H, axis=1, keepdims=True)
            a = cs.argmax(1); cmax = cs[np.arange(N), a]; incore = cmax >= CORE
            new = np.where(incore, a, -1 if label is None else label)
            if label is not None and t * DT > T / 2:
                hops += int(((new != label) & (label >= 0) & (new >= 0)).sum())
            label = new
            if t in snaps:
                ts.append(t * DT)
                alloc_t.append(np.r_[np.bincount(a[incore], minlength=K) / N, 1 - incore.mean()])
            if t * DT > T / 2 and t % 100 == 0:
                cos_l.append(float(cmax.mean()))
                sp, vc, pr = [], [], []
                for k in range(K):
                    m = incore & (a == k)
                    n = int(m.sum())
                    if n >= 3:
                        X = H[m]
                        sp.append(np.abs(X - X.mean(0)).sum(1).mean())
                        Cu = X @ S @ X.T; d = np.sqrt(np.diag(Cu))
                        vc.append(((Cu / np.outer(d, d)).sum() - n) / (n * (n - 1)))
                        ev = np.clip(np.linalg.eigvalsh(Cu), 0, None); pr.append(ev.sum() ** 2 / (ev ** 2).sum())
                ncl_l.append(int(sum((incore & (a == k)).sum() >= max(2, 0.02 * N) for k in range(K))))
                if sp:
                    sp_l.append(np.mean(sp)); vc_l.append(np.mean(vc)); pr_l.append(np.mean(pr))
                if t * DT > 3 * T / 4:
                    q4.append(H); lab4.append(np.where(incore, a, -1))
                else:
                    q3.append(H); lab3.append(np.where(incore, a, -1))
    # persistence: neurons in the core of the same part at every snapshot of both quarters ("stayers")
    L3, L4 = np.array(lab3), np.array(lab4)
    stay = (L3 == L3[0]).all(0) & (L4 == L3[0]).all(0) & (L3[0] >= 0)
    H3, H4 = np.mean(q3, 0), np.mean(q4, 0)
    num = den1 = den2 = 0.0; spbar = []
    for k in range(K):
        m = stay & (L3[0] == k)
        if m.sum() >= 3:
            d3 = H3[m] - H3[m].mean(0); d4 = H4[m] - H4[m].mean(0)
            num += (d3 * d4).sum(); den1 += (d3 ** 2).sum(); den2 += (d4 ** 2).sum()
            Hb = (H3[m] + H4[m]) / 2
            spbar.append(np.abs(Hb - Hb.mean(0)).sum(1).mean())
    persist = float(num / np.sqrt(den1 * den2)) if den1 > 0 and den2 > 0 else None
    stat = [x for x, tt in zip(alloc_t, ts) if tt > T / 2] or [alloc_t[-1]]
    lat = Wii[np.triu_indices(N, 1)]
    return dict(theta=theta, gamma=gamma, kind=kind, sigma=sig, lateral=lateral, init=init, seed=seed, N=N, T=T,
                t=ts, alloc_t=alloc_t, alloc=np.mean([x[:K] for x in stat], 0), transit=float(np.mean([x[K] for x in stat])),
                hop_rate=hops / (N * T / 2), K_core=float(np.mean(ncl_l)) if ncl_l else None,
                spread_inst=float(np.mean(sp_l)) if sp_l else None, vcorr_inst=float(np.mean(vc_l)) if vc_l else None,
                pr_inst=float(np.mean(pr_l)) if pr_l else None, cos_inst=float(np.mean(cos_l)) if cos_l else None,
                spread_bar=float(np.mean(spbar)) if spbar else None, persistence=persist, n_stayers=int(stay.sum()),
                lat_sum_per_neuron=float(Wii.sum(1).mean()), ib_sum_per_neuron=float(Wib.sum(1).mean()),
                lat_mean=float(lat.mean()))


NET_VARS = dict(graded=(0.7, 1.0, 1.4, 2.0), mild=(0.85, 1.0, 1.15, 1.3))


def net_cov(name="graded"):
    # 4 blocks of 3 (within 0.6, across 0); block variances v_k  =>  lambda_k = 2.2 v_k
    S, lab = cov_block([3, 3, 3, 3], RHO, 0.0, var=list(NET_VARS[name]))
    parts = np.array([(lab == k) / 3 for k in range(4)], float)
    return S, lab, parts


def jobNlone(args):
    """Lone-limit prediction for the network covariance (same noise, M lone neurons, no lateral synapses)."""
    name, theta, kind, sig, M, T = args
    S, lab, parts = net_cov(name)
    rng = np.random.default_rng(5)
    W0 = rng.exponential(0.2, (M, 12))
    r = lone_run(S, parts, np.full(M, theta), np.full(M, sig), np.zeros(M, int), 1, W0, T, kind if sig > 0 else "none",
                 seed=6, diag=False, snaps=(T,))
    return dict(cov=name, theta=theta, kind=kind, sigma=sig, alloc=r["alloc"][0], core=r["core"][0],
                transit=float(r["transit"][0]))


def partN(which="main"):
    jobs, ljobs = [], []
    N, T = 120, 800
    for name in ("graded", "mild"):
        cov = net_cov(name)
        # theta above (0.35) and below (0.12) the collective threshold gamma*kappa0 (0.23 graded, 0.20 mild)
        for theta in (0.12, 0.35):
            for kind, sigs in (("none", (0.0,)), ("inp", (0.2, 0.3, 0.4)), ("add", (0.1, 0.3))):
                for sig in sigs:
                    ljobs.append((name, theta, kind, sig, 600, 400))
                    for gamma in (0.0, 1.0):
                        for lateral in ((True, False) if sig > 0 else (True,)):
                            for init in ("rand", "block0", "block3"):
                                for seed in (0, 1):
                                    jobs.append((cov, theta, gamma, kind, sig, lateral, init, seed, N, T))
    with Timer(f"N ({len(jobs)} + {len(ljobs)})"):
        R = _map(jobN, jobs)
        RL = _map(jobNlone, ljobs)
    covname = {id(None): None}
    for r, j in zip(R, jobs):
        r["cov"] = "graded" if np.allclose(j[0][0], net_cov("graded")[0]) else "mild"
    for r in R:
        print(f"N {r['cov']} th={r['theta']} g={r['gamma']:.0f} {r['kind']} sig={r['sigma']} lat={r['lateral']:d} {r['init']:6s} "
              f"s{r['seed']} alloc={np.round(r['alloc'], 2)} tr={r['transit']:.2f} hop={r['hop_rate']:.1e} K={r['K_core']} "
              f"sp={_f(r['spread_inst'])} spbar={_f(r['spread_bar'])} pers={_f(r['persistence'])} vc={_f(r['vcorr_inst'], 5)} "
              f"pr={_f(r['pr_inst'], 4)} cos={_f(r['cos_inst'])} lat={r['lat_sum_per_neuron']:.3g} ib={r['ib_sum_per_neuron']:.3g}",
              flush=True)
    for r in RL:
        print(f"NL {r['cov']} th={r['theta']} {r['kind']} sig={r['sigma']} alloc={np.round(r['alloc'], 3)} "
              f"core={np.round(r['core'], 3)} tr={r['transit']:.2f}", flush=True)
    kap = {n: float(net_cov(n)[0].sum() / 144) for n in NET_VARS}
    raw_save("N", dict(rows=R, lone=RL, kappa0=kap, lam={n: [2.2 * x for x in v] for n, v in NET_VARS.items()}, N=N, T=T))


def _f(x, d=3):
    return "None" if x is None else f"{x:.{d}g}"



# ----------------------------------------------------------------------------- DT: step-size sensitivity
def jobDT(args):
    kind, th, s, dt, M, T = args
    v = V6; B = len(v)
    W0 = np.vstack([vertex_init(M, B, 0, v[0] - th), vertex_init(M, B, B - 1, v[-1] - th)])
    r = lone_run(np.diag(v), np.eye(B), np.full(2 * M, th), np.full(2 * M, s), np.repeat([0, 1], M), 2, W0, T, kind,
                 seed=21, dt=dt, diag=True, snaps=(T,), check=max(1, int(round(0.5 / dt))))
    return dict(kind=kind, theta=th, sigma=s, dt=dt, alloc_weak=r["alloc"][0], alloc_strong=r["alloc"][1],
                off=float(r["off"][1]), transit=float(r["transit"][1]),
                mean_off_w=float(r["W"][M:, :5].mean()), pred_cont=s * s / (2 * th))


def partDT(M=500, T=400):
    jobs = [(k, 0.2, s, dt, M, T) for k, ss in (("add", (0.1, 0.4)), ("inp", (0.2, 0.4))) for s in ss
            for dt in (0.05, 0.0125)]
    jobs += [("mul", 0.2, 2.0, dt, M, T) for dt in (0.05, 0.0125)]
    with Timer("DT"):
        R = _map(jobDT, jobs)
    raw_save("DT", dict(rows=R, M=M, T=T))
    for r in R:
        print(f"DT {r['kind']} th={r['theta']} sig={r['sigma']} dt={r['dt']} weak={np.round(r['alloc_weak'], 3)} "
              f"strong={np.round(r['alloc_strong'], 3)} off={r['off']:.3f} transit={r['transit']:.2f} "
              f"floor_mean={r['mean_off_w']:.4f} cont_pred={r['pred_cont']:.4f}", flush=True)


# ----------------------------------------------------------------------------- L: theory checks (gamma = 0)
def partL():
    """(i) Rayleigh quotient R(h) = h'Ah/h'h is non-decreasing along the noiseless clipped lone flow
    (h-dynamics = normalised linear flow x' = Ax in time tau = int dt/s); (ii) the drift f(w) = Ah - w is
    not a gradient field: relative antisymmetric part of its Jacobian; (iii) floor-synapse stationary law
    under additive noise (reflected drift -theta): mean sigma^2/(2 theta) -- compared with simulation."""
    rng = np.random.default_rng(0)
    out = dict(lyapunov=[], curl=[], floor=[])
    for trial, name in enumerate(("signed_powerlaw8", "signed_powerlaw8_b", "diag8")):
        B = 8
        Q, _ = np.linalg.qr(rng.standard_normal((B, B)))
        S = (Q * np.arange(1, B + 1) ** -1.0) @ Q.T; S *= B / np.trace(S)
        if name == "diag8":
            S = np.diag(rng.uniform(0.5, 3, B))
        th, dt = 0.15, 0.01
        A = S - th
        W = rng.exponential(0.2, (500, B)); Rprev = None; worst = 0.0; rise = 0.0
        for t in range(20000):
            W = np.maximum(W, 1e-6); H = W / W.sum(1, keepdims=True)
            R = np.einsum("ib,bc,ic->i", H, A, H) / (H * H).sum(1)
            if Rprev is not None:
                worst = min(worst, float((R - Rprev).min())); rise += float((R - Rprev).clip(0).mean())
            Rprev = R
            W = W + dt * (H @ S - th - W)
        out["lyapunov"].append(dict(cov=name, theta=th, most_negative_dR_per_step=worst, mean_total_rise=rise))
    v = V6
    for th in (0.05, 0.2, 0.4):
        ratios = []
        for _ in range(200):
            w = rng.exponential(1, 6); e = 1e-6
            f = lambda w: v * w / w.sum() - th - w
            J = np.array([(f(w + e * np.eye(6)[j]) - f(w - e * np.eye(6)[j])) / (2 * e) for j in range(6)]).T
            ratios.append(np.linalg.norm(J - J.T) / np.linalg.norm(J + J.T))
        out["curl"].append(dict(theta=th, antisym_over_sym_median=float(np.median(ratios)), max=float(np.max(ratios))))
    # floor synapse: neuron at vertex 5 (v=3), additive noise; mean off-support weight vs sigma^2/(2 theta)
    for th in (0.05, 0.2, 0.4):
        for s in (0.05, 0.1, 0.2):
            M = 500
            r = lone_run(np.diag(v), np.eye(6), np.full(M, th), np.full(M, s), np.zeros(M, int), 1,
                         vertex_init(M, 6, 5, v[5] - th), 100, "add", seed=2, diag=True, snaps=(100,))
            W = r["W"]
            out["floor"].append(dict(theta=th, sigma=s, mean_off=float(W[:, :5].mean()), pred=s * s / (2 * th),
                                     off_mass=float(r["off"][0])))
    raw_save("L", out)
    print(json.dumps(_clean(out), indent=1))


def aggregate_network(rows):
    """One record per (cov, theta, gamma, kind, sigma, lateral): per-init allocation (core fractions, mean of 2 seeds),
    TV between block0- and block3-started runs, seed-0 time series per init, mean metrics."""
    keys = []
    for r in rows:
        k = (r["cov"], r["theta"], r["gamma"], r["kind"], r["sigma"], r["lateral"])
        if k not in keys:
            keys.append(k)
    out = []
    mets = ("transit", "hop_rate", "K_core", "spread_inst", "spread_bar", "persistence", "vcorr_inst", "pr_inst",
            "cos_inst", "lat_sum_per_neuron", "ib_sum_per_neuron")
    for k in keys:
        rr = [r for r in rows if (r["cov"], r["theta"], r["gamma"], r["kind"], r["sigma"], r["lateral"]) == k]
        by = {i: np.mean([r["alloc"] for r in rr if r["init"] == i], 0) for i in ("rand", "block0", "block3")}
        rec = dict(cov=k[0], theta=k[1], gamma=k[2], kind=k[3], sigma=k[4], lateral=k[5],
                   alloc_by_init=by, tv_block0_block3=tv(by["block0"], by["block3"]),
                   alloc_t_seed0={r["init"]: r["alloc_t"] for r in rr if r["seed"] == 0})
        for m in mets:
            x = [r[m] for r in rr if r[m] is not None]
            rec[m] = float(np.mean(x)) if x else None
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- assemble site JSON
def assemble():
    A, R, M, P, X, DT, L, NN = (raw_load(k) for k in ("A", "R", "M", "P", "X", "DT", "L", "N"))
    out = {}
    # A: allocation vs time from three inits + stationary summaries
    out["alloc_vs_time"] = [dict(kind=r["kind"], theta=r["theta"], sigma=r["sigma"], init=r["init"], alloc=r["alloc_t"],
                                 transit=r["transit_t"], off=r["off_t"]) for r in A["rows"]]
    out["snap_times"] = A["rows"][0]["t"]
    summ = A["summary"] + (X["summary"] if X else [])
    out["stationary_diag"] = [dict(kind=s["kind"], theta=s["theta"], sigma=s["sigma"], alloc=s["alloc"], core=s["core"],
                                   tv_weak_strong=s["tv_ws"], tv_rand_strong=s["tv_rs"], equilibrated=s["equilibrated"],
                                   transit=s["transit"], off_mass=s["off"], cos_inst=s["cos"], cos_mean=s["cos_mean"],
                                   dead=s["dead"], hop_rate=s["hop_rate"], relax=s["relax"], pi_rates=s["pi_Q"],
                                   beta=s["fit_pl"].get("beta"), beta_rms=s["fit_pl"].get("rms"), beta_r=s["fit_pl"].get("r"),
                                   beta_core=s["fit_pl_core"].get("beta"), T_exp=s["fit_exp_v"].get("T"),
                                   T_exp_rms=s["fit_exp_v"].get("rms"))
                              for s in summ]
    # R: rates out of each vertex (milestoning), Arrhenius fits
    out["rates"] = [dict(kind=r["kind"], theta=r["theta"], sigma=r["sigma"], out_rate=r["out_rate"], pi_rates=r["pi_Q"],
                         relax=r["relax"], irreducible=r["irreducible"], off_mass=r["off"]) for r in R["rows"]]
    out["arrhenius"] = R["arrhenius"]
    out["multiplicative_floor"] = M["rows"]
    out["planted"] = P["fits"]
    out["dt_check"] = DT["rows"]
    out["theory_checks"] = L
    if NN:
        out["network"] = aggregate_network(NN["rows"])
        out["network_snap_times"] = NN["rows"][0]["t"]
        out["network_lone_prediction"] = NN["lone"]
        out["network_meta"] = dict(kappa0=NN["kappa0"], lam=NN["lam"], N=NN["N"], T=NN["T"])
    meta = dict(
        script="hebb/sweeps/s8_private_noise.py (parts A R M P X DT L N; Modal helper s8_modal.py)",
        description="Allocation of neurons to parts under PRIVATE (independent per-synapse) noise. Does the allocation "
                    "equilibrate (forget the init) and, if so, what law n_k(lambda_k; theta, sigma) does the stationary "
                    "state obey? gamma=0 lone-neuron limit (thousands of independent neurons; no lateral synapses) and "
                    "N=120 networks with gamma in {0,1}.",
        units="lam = eta = 1: weights and theta in covariance units, time in units of 1/lam; Euler dt = 0.05; "
              "wmin = 1e-4 unless stated. sigma is per sqrt(time unit).",
        noise_models=dict(
            add="dw = drift dt + sigma sqrt(dt) xi, then clip at wmin",
            mul="w <- w exp(sigma sqrt(dt) xi) after the drift step, then clip at wmin (median-neutral)",
            inp="private covariance-estimation noise: each synapse estimates Cov from its own samples, n = 1/sigma^2 "
                "samples per unit time; Gaussian approx, Var per step = sigma^2 dt (C_xx C_yy + C_xy^2), independent "
                "across synapses (I-B: q_i Sigma_bb + (Sigma h_i)_b^2; I-I: gamma^2 (q_i q_j + C_ij^2))",
            inpn="as inp but one private sample stream per neuron (correlated across its synapses)"),
        diag_sigma=dict(v=V6, note="Sigma = diag(v); parts = vertices e_b with c_b = v_b - theta; lambda_b = v_b"),
        measures=dict(
            alloc="fraction of neurons whose nearest part (max cosine) is k, time-averaged over the second half",
            core="fraction with cos(h, part k) >= 0.9 (in the part); transit = 1 - sum(core)",
            off_mass="mean mass of h outside the support of the nearest part",
            cos_inst="mean over neurons of cos(h_i, nearest part)", cos_mean="cos(time- and member-averaged h of core-k neurons, part k)",
            hop_rate="milestoning transitions between part cores per neuron per unit time (second half)",
            relax="slowest relaxation time 1/|eig_2| of the measured milestoning rate matrix (inf if not irreducible)",
            pi_rates="stationary distribution of the measured rate matrix",
            beta="slope of log alloc vs log v over parts with alloc > 1e-3 (rms = residual in log units)",
            T_exp="Boltzmann fit alloc ∝ exp(v/T)",
            tv_weak_strong="total variation between the stationary allocations started from 'weak' and 'strong' inits",
            arrhenius="ln(out-rate of vertex) = a - 2*barrier/sigma^2 (barrier in units of D = sigma^2/2), fitted over sigma",
            network="per condition: alloc_by_init = core fractions per block (last entry of alloc_t = transit), mean "
                    "over seeds 0,1 and the second half of the run; metrics are means over seeds and inits; spread_inst = "
                    "mean L1 of members' h to their part-cluster mean (core members); vcorr_inst = mean pairwise voltage "
                    "correlation within a part-cluster; pr_inst = participation ratio of within-cluster voltage covariance; "
                    "lat_sum_per_neuron = summed lateral conductance per neuron at the end",
            persistence="network: correlation between the deviations of a neuron's time-averaged embedding from its "
                        "cluster mean in the 3rd vs 4th quarter of the run (1 = frozen individual differences, 0 = jitter)"),
        inits=dict(weak="all neurons at the weakest vertex fixed point", strong="all at v=3", rand="W_IB ~ Exp(0.2)",
                   block0="network: all neurons on block 0 (weakest)", block3="network: all on block 3 (loudest)"),
        seeds=dict(A="noise seed 7, init seed 123, M=1000 per init", R="seed 11, M=500 per start vertex",
                   M="seed 5, M=500", P="seed 17/3, M=600 per init", X="as A, T=600", N="seeds 0,1 (init and noise)"),
        params=dict(A=dict(T=A["T"], M=A["M"], thetas=THETAS_A, sigmas=SIG_A), R=dict(T=R["T"], M=R["M"], sigmas=SIG_R),
                    P=dict(T=P["T"], M=P["M"], sigmas=SIG_P, thetas=(0.15, 0.25),
                           family="4 blocks of 3, within 0.6, across 0; var: block-0 variance x; size: block-0 size x at "
                                  "unit variance; ratio = frac0 / mean frac of the other blocks; beta = slope of log ratio "
                                  "vs log(lambda0/lambda_ref)"),
                    N=dict(N=120, T=800, covs={k: dict(block_var=v, lam=[2.2 * x for x in v]) for k, v in NET_VARS.items()},
                           wmin=WMIN, init="W_IB ~ Exp(0.2), W_II ~ U(0,0.03) for rand; block inits put all neurons on one "
                                           "block with W_II at the floor", lateral="noise on lateral synapses too (True) "
                                           "or only on I-B synapses (False)")))
    p = save_json("alloc_private_noise", out, meta)
    print("wrote", p, p.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "L":
        partL()
    elif arg == "DT":
        partDT()
    elif arg == "X":
        partX()
    elif arg == "assemble":
        assemble()
    if arg == "A":
        partA()
    elif arg == "R":
        partR()
    elif arg == "M":
        partM()
    elif arg == "P":
        partP()
    elif arg == "N":
        partN(sys.argv[2] if len(sys.argv) > 2 else "main")
