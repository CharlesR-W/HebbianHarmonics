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

import os as _os
# Revision: S8_REFLECT=1 makes add/inp/inpn use the reflected scheme (drift+noise from one state, reflect at wmin)
# and tags raw outputs with suffix '_r'. Original results (projected/clipped Euler) are kept under the old names.
REFLECT = bool(_os.environ.get("S8_REFLECT"))
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


def floor_step(W, c, sig, sq, dt, wmin, kind, noise_fn, rng, scheme):
    """One step of dw = (c - w) dt + noise, c = (Sigma h)_b - theta frozen over the step (h explicit).
    scheme 'clip'   : original Euler-Maruyama, then clip at wmin (projected); for 'mul' = Euler drift, then
                      w *= exp(sigma sqrt(dt) xi), then clip (NOT floor-compatible: see revision note).
    scheme 'reflect': (add/inp) Strang splitting: exact flow of dw/dt = c - w over dt/2, sticking at wmin (exact
                      for a deterministic flow into the floor), additive increment reflected at wmin (exact for
                      Brownian motion), exact flow dt/2. (Reflecting the whole Euler increment would lift a
                      noiseless floored synapse by theta*dt per step.)
    scheme 'log'    : (mul) Ito SDE dw = (c - w) dt + sigma w dW reflected at wmin, integrated in x = log w:
                      Strang splitting: exact flow of dw/dt = c - (1 + sigma^2/2) w over dt/2 (sticks at wmin
                      instead of crossing it), x += sigma sqrt(dt) xi reflected at log wmin, exact flow dt/2."""
    if scheme == "log":   # Strang splitting: exact drift dt/2, log-noise, exact drift dt/2
        k = 1.0 + 0.5 * sig ** 2
        ck = c / k
        e = np.exp(-k * dt / 2)
        Wd = np.maximum(ck + (W - ck) * e, wmin)
        x = np.log(Wd) + sig * sq * rng.standard_normal(W.shape)
        xm = np.log(wmin)
        x = np.where(x < xm, 2 * xm - x, x)
        return np.maximum(ck + (np.exp(x) - ck) * e, wmin)
    if scheme == "reflect":   # Strang: exact drift dt/2 (stick at floor), noise reflected at wmin, exact drift dt/2
        e = np.exp(-dt / 2)
        Wd = np.maximum(c + (W - c) * e, wmin)
        Wn = noise_fn(Wd)          # noise amplitude from the start-of-step state (closure)
        Wn = np.where(Wn < wmin, 2 * wmin - Wn, Wn)
        return np.maximum(c + (Wn - c) * e, wmin)
    Wn = noise_fn(W + dt * (c - W))
    return np.maximum(Wn, wmin)


def lone_run(Sigma, parts, theta, sig, group, G, W0, T, kind, seed=0, wmin=WMIN, dt=DT, snaps=SNAPS,
             check=10, meas_from=None, diag=None, scheme="clip"):
    """Independent lone neurons (rows). theta, sig: per-row arrays; group: per-row int in [0,G).
    parts: (K,B) part embeddings (rows sum to 1). Returns per-group time series and rate counts.
    Milestoning: label = last core part visited (-1 before first core visit)."""
    rng = np.random.default_rng(seed)
    if REFLECT and scheme == "clip" and kind in ("add", "inp", "inpn"):
        scheme = "reflect"
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
        if scheme == "clip":
            W = W + dt * (SH - th - W)
            W = noise_step(W, H, SH, Sd, kind, sg, sq, rng, L)
            W = np.maximum(W, wmin)
        else:
            W = floor_step(W, SH - th, sg, sq, dt, wmin, kind,
                           lambda X: noise_step(X, H, SH, Sd, kind, sg, sq, rng, L), rng, scheme)
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
    if REFLECT:
        name = name + "_r"
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
    sigs = args[4] if len(args) > 4 and args[4] is not None else SIG_A[kind]
    kw = args[5] if len(args) > 5 else {}
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
            W = vertex_init(M, B, k, v[k] - th, kw.get("wmin", WMIN))
        W0.append(W); th_r.append(np.full(M, th)); sg_r.append(np.full(M, s)); grp.append(np.full(M, gi))
    r = lone_run(np.diag(v), parts, np.concatenate(th_r), np.concatenate(sg_r), np.concatenate(grp), len(conds),
                 np.vstack(W0), T, kind, seed=7, diag=True, dt=kw.get("dt", DT), scheme=kw.get("scheme", "clip"),
                 wmin=kw.get("wmin", WMIN), check=max(1, int(round(0.5 / kw.get("dt", DT)))))
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
    kinds = ("add", "inp") if REFLECT else ("add", "inp", "mul")
    jobs = [(k, M, T, (th,)) for k in kinds for th in THETAS_A] + [("inpn", M, T, (0.2,))]
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
    th, wmin, sigs, M, T = args[:5]
    kw = args[5] if len(args) > 5 else {}
    dt = kw.get("dt", DT); scheme = kw.get("scheme", "clip")
    v = V6; B = len(v)
    W0, sg, grp, conds = [], [], [], []
    for s in sigs:
        for k in (0, B - 1):
            conds.append((s, k))
            W0.append(vertex_init(M, B, k, v[k] - th, wmin)); sg.append(np.full(M, s)); grp.append(np.full(M, len(conds) - 1))
    r = lone_run(np.diag(v), np.eye(B), np.full(len(conds) * M, th), np.concatenate(sg), np.concatenate(grp), len(conds),
                 np.vstack(W0), T, "mul", seed=5, wmin=wmin, diag=True, meas_from=5.0, snaps=(T,), dt=dt, scheme=scheme,
                 check=max(1, int(round(0.5 / dt))))
    out = []
    for gi, (s, k) in enumerate(conds):
        tau = r["tau"][gi].sum()
        out.append(dict(theta=th, wmin=wmin, sigma=s, start=k, dt=dt, scheme=scheme, exits=float(r["cnt"][gi].sum()), time=float(tau),
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
def net_noise(Wii, Wib, Mii, Mib, H, Sigma, kind, sig, sq, rng, lateral=True, gamma=1.0, clip=True):
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
    Wib = (np.maximum(Wib + Zb, 0) if clip else Wib + Zb) * Mib
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
        if REFLECT and kind != "none":
            # same update as core.step, but drift and noise from one state, then reflect at the floor
            # core.step's update with the floor-compatible Strang splitting of floor_step (frozen H over the step)
            Wii = np.maximum(Wii, WMIN * Mii); Wib = np.maximum(Wib, WMIN * Mib)
            H = harmonic(Wii, Wib); HS = H @ S
            cii = -theta + gamma * (HS @ H.T); cib = -theta + HS
            e = np.exp(-DT / 2)
            Wii = np.maximum(cii + (Wii - cii) * e, WMIN); Wib = np.maximum(cib + (Wib - cib) * e, WMIN)
            Wii, Wib = net_noise(Wii, Wib, Mii, Mib, H, S, kind, sig, sq, nrng, lateral, gamma, clip=False)
            Wii = np.where(Wii < WMIN, 2 * WMIN - Wii, Wii); Wib = np.where(Wib < WMIN, 2 * WMIN - Wib, Wib)
            Wii = np.maximum(cii + (Wii - cii) * e, WMIN); Wib = np.maximum(cib + (Wib - cib) * e, WMIN)
        else:
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



# ----------------------------------------------------------------------------- V: scheme validation (revision)
def mfpt_quad(theta, a, sig, L, b, n=40000):
    """ln of the mean first-passage time L -> b of the Ito SDE dw = (-theta + a w) dt + sig w dW, reflected at L.
    T = int_L^b s(y) int_L^y m(z) dz dy, ln s(y) = (2theta/sig^2)(1/L - 1/y) - (2a/sig^2) ln(y/L),
    m(z) = 2/(sig^2 z^2 s(z)). Log-space quadrature on a log grid."""
    u = np.linspace(np.log(L), np.log(b), n); du = u[1] - u[0]; y = np.exp(u)
    ls = (2 * theta / sig ** 2) * (1 / L - 1 / y) - (2 * a / sig ** 2) * (u - np.log(L))
    lm = np.log(2) - 2 * np.log(sig) - 2 * u - ls
    lI = np.logaddexp.accumulate(lm + u + np.log(du))
    from scipy.special import logsumexp
    return float(logsumexp(ls + lI + u + np.log(du)))


def sim_escape(args):
    """Simulate the same 1-D problem with the lone_run update structure (c = -theta + (1+a) w explicit, -w part
    per scheme). Returns escape rate = absorbed / total time (exponential first passage)."""
    theta, a, sig, L, b, dt, scheme, M, Tmax, seed = args
    rng = np.random.default_rng(seed)
    w = np.full(M, L); alive = np.ones(M, bool); t_abs = np.full(M, np.nan)
    sq = np.sqrt(dt); n = int(round(Tmax / dt))
    for t in range(1, n + 1):
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            break
        x = w[idx]
        c = -theta + (1 + a) * x
        if scheme == "clip":   # original mul scheme
            x = np.maximum((x + dt * (c - x)) * np.exp(sig * sq * rng.standard_normal(len(x))), L)
        else:
            x = floor_step(x, c, sig, sq, dt, L, "mul", None, rng, "log")
        w[idx] = x
        hit = x >= b
        t_abs[idx[hit]] = t * dt
        alive[idx[hit]] = False
    nabs = int((~alive).sum())
    tot = float(np.nansum(t_abs) + alive.sum() * n * dt)
    return dict(theta=theta, a=a, sigma=sig, wmin=L, b=b, dt=dt, scheme=scheme, M=M, Tmax=Tmax, absorbed=nabs,
                rate=nabs / tot, x=2 * theta / (sig ** 2 * L))


def sim_stationary(args):
    """1-D floored synapse dw = (-theta - w) dt + noise, reflected at L; histogram of log w over t in [T/2, T]."""
    kind, theta, sig, L, dt, scheme, M, T, seed = args
    rng = np.random.default_rng(seed)
    w = np.full(M, L); sq = np.sqrt(dt); n = int(round(T / dt))
    edges = np.log(L) + np.linspace(0, 8, 41)
    hist = np.zeros(len(edges) - 1); s1 = 0.0; cnt = 0
    for t in range(1, n + 1):
        c = np.full(M, -theta)
        if kind == "mul":
            if scheme == "clip":
                w = np.maximum((w + dt * (c - w)) * np.exp(sig * sq * rng.standard_normal(M)), L)
            else:
                w = floor_step(w, c, sig, sq, dt, L, "mul", None, rng, "log")
        else:
            w = floor_step(w, c, sig, sq, dt, L, "add", lambda X: X + sig * sq * rng.standard_normal(M), rng, scheme)
        if t * dt > T / 2 and t % 4 == 0:
            hist += np.histogram(np.log(w), edges)[0]; s1 += w.sum(); cnt += M
    # analytic
    xg = np.linspace(edges[0], edges[-1] + 30, 200001); wg = np.exp(xg)
    if kind == "mul":   # p(w) ∝ w^(-2-2/sig^2) exp(2 theta/(sig^2 w)); in x: times w
        lp = (-1 - 2 / sig ** 2) * xg + 2 * theta / (sig ** 2 * wg)
    else:               # p(w) ∝ exp(-(2/sig^2)(theta w + w^2/2)); in x: times w
        lp = -(2 / sig ** 2) * (theta * wg + wg ** 2 / 2) + xg
    pg = np.exp(lp - lp.max()); pg /= pg.sum()
    mass = np.array([pg[(xg >= edges[i]) & (xg < edges[i + 1])].sum() for i in range(len(edges) - 1)])
    mean_an = float((pg * wg).sum())
    hist = hist / cnt
    tvd = 0.5 * (np.abs(hist - mass).sum() + abs((1 - hist.sum()) - (1 - mass.sum())))   # incl. mass beyond the bins
    return dict(kind=kind, theta=theta, sigma=sig, wmin=L, dt=dt, scheme=scheme, tv=float(tvd),
                mean_sim=s1 / cnt, mean_analytic=mean_an, hist=hist, analytic=mass,
                edges=edges)


def partV():
    th, a = 0.05, 3 / 1.95 - 1
    b = 3 * th / a
    esc_cases = [(0.03, 1.0), (0.01, 2.0), (0.01, 1.5), (0.03, 0.7), (0.003, 2.0), (0.01, 1.0)]
    jobs = []
    for L, s in esc_cases:
        for dt in (0.05, 0.0125, 0.003):
            for sch in ("log", "clip"):
                if sch == "clip" and dt == 0.003:
                    continue
                jobs.append((th, a, s, L, b, dt, sch, 2000, 400.0, 1))
    with Timer(f"V esc ({len(jobs)})"):
        E = _map(sim_escape, jobs)
    for e in E:
        e["rate_quad"] = float(np.exp(-mfpt_quad(th, a, e["sigma"], e["wmin"], b)))
        print(f"V esc x={e['x']:.3g} wmin={e['wmin']} sig={e['sigma']} dt={e['dt']} {e['scheme']:4s} rate={e['rate']:.3e} "
              f"quad={e['rate_quad']:.3e} absorbed={e['absorbed']}/{e['M']}", flush=True)
    # default-floor predictions (quadrature only)
    pred = []
    for th_ in (0.05, 0.2):
        for s in (0.5, 1.0, 2.0, 3.0):
            lT = mfpt_quad(th_, a, s, 1e-4, 3 * th_ / a)
            pred.append(dict(theta=th_, sigma=s, wmin=1e-4, x=2 * th_ / (s * s * 1e-4), ln_mfpt=lT))
            print(f"V pred wmin=1e-4 th={th_} sig={s} x={2 * th_ / (s * s * 1e-4):.3g} ln MFPT={lT:.1f}", flush=True)
    sj = []
    for s in (1.0, 2.0):
        for dt in (0.05, 0.0125):
            for sch in ("log", "clip"):
                sj.append(("mul", 0.05, s, 0.01, dt, sch, 20000, 100.0, 2))
    for s in (0.1, 0.4):
        for dt in (0.05, 0.0125, 0.003):
            for sch in ("reflect", "clip"):
                sj.append(("add", 0.2, s, 1e-4, dt, sch, 20000, 40.0, 3))
    with Timer(f"V stat ({len(sj)})"):
        S = _map(sim_stationary, sj)
    for r in S:
        print(f"V stat {r['kind']} th={r['theta']} sig={r['sigma']} wmin={r['wmin']} dt={r['dt']} {r['scheme']:7s} "
              f"TV(hist,analytic)={r['tv']:.3f} mean={r['mean_sim']:.4g} analytic={r['mean_analytic']:.4g}", flush=True)
    raw_save("V", dict(escape=E, default_floor_prediction=pred, stationary=S, escape_problem=dict(
        theta=th, a=a, b=b, note="dw = (-theta + a w) dt + sigma w dW (Ito), reflected at wmin, absorbed at b = 3 theta/a; "
                                 "a = 3/1.95 - 1: floored synapse to input v=3 of a neuron at vertex v=2, theta=0.05")))


def partL2():
    """Revision: most negative dR/dt of the Rayleigh quotient along the clipped lone flow vs the floor wmin."""
    rng = np.random.default_rng(0)
    B = 8
    Q, _ = np.linalg.qr(rng.standard_normal((B, B)))
    S = (Q * np.arange(1, B + 1) ** -1.0) @ Q.T; S *= B / np.trace(S)
    out = []
    for name, Sig in (("signed_powerlaw8", S), ("diag8", np.diag(rng.uniform(0.5, 3, B)))):
        for wmin in (0.0, 1e-6, 1e-4, 1e-3):
            th, dt = 0.15, 0.01
            A = Sig - th
            W = rng.exponential(0.2, (500, B)); Rp = None; worst = 0.0
            for t in range(20000):
                W = np.maximum(W, wmin)
                H = W / W.sum(1, keepdims=True)
                R = np.einsum("ib,bc,ic->i", H, A, H) / (H * H).sum(1)
                if Rp is not None:
                    worst = min(worst, float(((R - Rp) / dt).min()))
                Rp = R
                W = W + dt * (H @ Sig - th - W)
            out.append(dict(cov=name, wmin=wmin, theta=0.15, most_negative_dRdt=worst))
            print("L2", out[-1], flush=True)
    raw_save("L2", dict(rows=out))


# ----------------------------------------------------------------------------- revision reruns
def partA2():
    """Multiplicative noise rerun with the floor-compatible log scheme (diag Sigma, three inits)."""
    D = 0.0125
    jobs = [("mul", 1000, 1000, (th,), (s,), dict(scheme="log", dt=D)) for th in THETAS_A for s in (0.3, 1.0, 2.0, 3.0)]
    jobs += [("mul", 1000, 1000, (th,), (s,), dict(scheme="log", wmin=wm, dt=D)) for th in (0.05, 0.2)
             for wm in (0.01, 0.03) for s in (1.0, 1.5, 2.0, 3.0)]
    jobs += [("mul", 1000, 1000, (0.05,), (2.0,), dict(scheme="log", wmin=0.01, dt=d)) for d in (0.05, 0.003)]
    with Timer(f"A2 ({len(jobs)})"):
        RR = _map(jobA, jobs)
    out = []
    for j, rr in zip(jobs, RR):
        sm = summarize_A(rr, 1000, 1000)[0]
        sm.update(wmin=j[5].get("wmin", WMIN), dt=j[5].get("dt", DT), scheme="log")
        out.append(dict(summary=sm, rows=rr))
        print(f"A2 mul th={sm['theta']} sig={sm['sigma']} wmin={sm['wmin']} dt={sm['dt']} TV(w,s)={sm['tv_ws']:.3f} "
              f"TV(r,s)={sm['tv_rs']:.3f} pi={np.round(sm['alloc'], 3)} transit={sm['transit']:.2f} off={sm['off']:.2f} "
              f"hop={sm['hop_rate']:.2e} relax={sm['relax']:.3g} beta={sm['fit_pl']['beta']}", flush=True)
    raw_save("A2", dict(cells=out, T=1000, M=1000))


def partM2(M=500, T=500):
    jobs = []
    for th in (0.05, 0.2):
        for wmin in (1e-4, 3e-3, 1e-2, 3e-2):
            jobs.append((th, wmin, (0.5, 1.0, 1.5, 2.0, 3.0), M, T, dict(scheme="log", dt=0.0125)))
    jobs.append((0.05, 1e-2, (1.0, 1.5, 2.0), M, T, dict(scheme="log", dt=0.003)))
    jobs.append((0.05, 1e-2, (1.0, 1.5, 2.0), M, T, dict(scheme="log", dt=0.05)))
    with Timer("M2"):
        R = [x for rr in _map(jobM, jobs) for x in rr]
    raw_save("M2", dict(rows=R, M=M, T=T))
    for r in R:
        print(f"M2 th={r['theta']} wmin={r['wmin']} dt={r['dt']} sig={r['sigma']} start={r['start']} x={r['x']:.3g} "
              f"rate={r['rate']:.3g} off={r['off']:.2f} alloc_end={np.round(r['alloc_end'], 3)}", flush=True)


def partH():
    """Headline additive / input cells with the reflected scheme at dt = 0.0125 (vs original clip at 0.05)."""
    jobs = []
    for kind, sigs in (("add", (0.3, 0.4, 0.5, 0.7)), ("inp", (0.2, 0.3, 0.4, 0.6))):
        for s in sigs:
            jobs.append((kind, 1000, 600, (0.2,), (s,), dict(scheme="reflect", dt=0.0125)))
            jobs.append((kind, 1000, 600, (0.2,), (s,), dict(scheme="reflect", dt=0.05)))
    with Timer(f"H ({len(jobs)})"):
        RR = _map(jobA, jobs)
    out = []
    for j, rr in zip(jobs, RR):
        sm = summarize_A(rr, 600, 1000)[0]
        sm.update(dt=j[5]["dt"], scheme="reflect")
        out.append(sm)
        print(f"H {sm['kind']} th=0.2 sig={sm['sigma']} dt={sm['dt']} reflect pi={np.round(sm['alloc'], 3)} TV={sm['tv_ws']:.3f} "
              f"transit={sm['transit']:.2f} off={sm['off']:.2f} beta={sm['fit_pl']['beta']} T_exp={sm['fit_exp_v']['T']}",
              flush=True)
    raw_save("H", dict(summary=out))


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
def _raw(name):
    p = RAW / f"s8_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _stat_row(s, scheme, wmin=WMIN, dt=DT):
    return dict(kind=s["kind"], theta=s["theta"], sigma=s["sigma"], scheme=scheme, wmin=wmin, dt=dt,
                alloc=s["alloc"], core=s["core"], tv_weak_strong=s["tv_ws"], tv_rand_strong=s["tv_rs"],
                equilibrated=s["equilibrated"], transit=s["transit"], off_mass=s["off"], cos_inst=s["cos"],
                cos_mean=s["cos_mean"], dead=s["dead"], hop_rate=s["hop_rate"], relax=s["relax"], pi_rates=s["pi_Q"],
                beta=s["fit_pl"].get("beta"), beta_rms=s["fit_pl"].get("rms"), beta_r=s["fit_pl"].get("r"),
                beta_core=s["fit_pl_core"].get("beta"), T_exp=s["fit_exp_v"].get("T"), T_exp_rms=s["fit_exp_v"].get("rms"))


def assemble():
    """Revision 2026-11-17: add/inp/inpn data from the reflected (Strang) scheme ('_r' raw files), multiplicative data
    from the validated Ito log scheme (A2, M2); original projected-Euler results stay in scratch (s8_A.json etc.)."""
    A, R, P, X, NN = (_raw(k) for k in ("A_r", "R_r", "P_r", "X_r", "N_r"))
    A2, M2, V, H, DTc, L, L2, A0 = (_raw(k) for k in ("A2", "M2", "V", "H", "DT", "L", "L2", "A"))
    out = {}
    mulrows = [r for c in A2["cells"] for r in c["rows"]]
    out["alloc_vs_time"] = [dict(kind=r["kind"], theta=r["theta"], sigma=r["sigma"], init=r["init"],
                                 scheme="log" if r["kind"] == "mul" else "reflect",
                                 wmin=c["summary"]["wmin"] if r["kind"] == "mul" else WMIN,
                                 dt=c["summary"]["dt"] if r["kind"] == "mul" else DT,
                                 alloc=r["alloc_t"], transit=r["transit_t"], off=r["off_t"])
                            for c in [dict(summary=dict(wmin=WMIN, dt=DT), rows=A["rows"])] + A2["cells"]
                            for r in c["rows"]]
    out["snap_times"] = A["rows"][0]["t"]
    out["snap_times_note"] = "mul rows (T=1000) have snapshots up to 1000 only"
    out["stationary_diag"] = ([_stat_row(s, "reflect") for s in A["summary"] + X["summary"]] +
                              [_stat_row(c["summary"], "log", c["summary"]["wmin"], c["summary"]["dt"]) for c in A2["cells"]])
    out["rates"] = [dict(kind=r["kind"], theta=r["theta"], sigma=r["sigma"], out_rate=r["out_rate"], pi_rates=r["pi_Q"],
                         relax=r["relax"], irreducible=r["irreducible"], off_mass=r["off"]) for r in R["rows"]]
    out["arrhenius"] = R["arrhenius"]
    out["multiplicative_floor"] = M2["rows"]
    out["planted"] = P["fits"]
    out["scheme_validation"] = dict(problem_escape=V["escape_problem"], escape=V["escape"],
                                    default_floor_prediction=V["default_floor_prediction"],
                                    stationary=V["stationary"])
    # headline cells: reflected (H, two dt) vs original projected Euler (A0, dt 0.05)
    orig = {(s["kind"], s["theta"], s["sigma"]): s for s in A0["summary"]}
    out["scheme_check_headline"] = [dict(kind=h["kind"], theta=h["theta"], sigma=h["sigma"], dt=h["dt"], scheme="reflect",
                                         alloc=h["alloc"], transit=h["transit"], off_mass=h["off"],
                                         beta=h["fit_pl"].get("beta"), T_exp=h["fit_exp_v"].get("T"),
                                         orig_clip_alloc=orig[(h["kind"], h["theta"], h["sigma"])]["alloc"],
                                         orig_clip_transit=orig[(h["kind"], h["theta"], h["sigma"])]["transit"],
                                         orig_clip_beta=orig[(h["kind"], h["theta"], h["sigma"])]["fit_pl"].get("beta"))
                                    for h in H["summary"]]
    out["dt_check_original_scheme"] = DTc["rows"]
    out["theory_checks"] = dict(L, rayleigh_vs_floor=L2["rows"])
    out["network"] = aggregate_network(NN["rows"])
    out["network_snap_times"] = NN["rows"][0]["t"]
    out["network_lone_prediction"] = NN["lone"]
    out["network_meta"] = dict(kappa0=NN["kappa0"], lam=NN["lam"], N=NN["N"], T=NN["T"], scheme="reflect (Strang)")
    meta = dict(
        script="hebb/sweeps/s8_private_noise.py (parts A R P X N rerun with S8_REFLECT=1; A2 M2 V H L2 added in revision; "
               "DT L original; Modal helper s8_modal.py)",
        description="Allocation of neurons to parts under PRIVATE (independent per-synapse) noise. Does the allocation "
                    "equilibrate (forget the init) and, if so, what law n_k(lambda_k; theta, sigma) does the stationary "
                    "state obey? gamma=0 lone-neuron limit (thousands of independent neurons; no lateral synapses) and "
                    "N=120 networks with gamma in {0,1}.",
        units="lam = eta = 1: weights and theta in covariance units, time in units of 1/lam; Euler dt = 0.05; "
              "wmin = 1e-4 unless stated. sigma is per sqrt(time unit).",
        noise_models=dict(
            add="Ito dw = (c - w) dt + sigma dW, reflected at wmin (c = (Sigma h)_b - theta, h frozen per step); "
                "Strang scheme: exact drift dt/2 sticking at wmin, increment reflected at wmin, exact drift dt/2",
            mul="Ito dw = (c - w) dt + sigma w dW, reflected at wmin (mean-neutral noise); integrated in log w: "
                "Strang splitting of the exact flow dw/dt = c - (1 + sigma^2/2) w (sticks at wmin) and "
                "log-increment sigma sqrt(dt) xi reflected at log wmin; dt = 0.0125",
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
            scheme_validation="1-D floored synapse: escape = Ito dw = (-theta + a w) dt + sigma w dW reflected at wmin, "
                              "absorbed at b; rate = absorbed/total time vs exp(-ln MFPT) from quadrature; stationary: "
                              "dw = (-theta - w) dt + noise, histogram of log w vs analytic density (TV incl. tail mass)",
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
    meta["revision"] = (
        "2026-11-17 (after independent review). (1) The original multiplicative integrator (Euler drift, then "
        "w *= exp(sigma sqrt(dt) xi), then clip) was not floor-compatible: a floored synapse's pre-noise weight is "
        "already negative, so it could never leave the floor at wmin=1e-4; it also integrated the Stratonovich SDE "
        "(Ito drift +sigma^2 w/2; at sigma=2 its mean weight diverges: 83-183 vs analytic 0.017). Replaced by the "
        "validated Ito log scheme (scheme_validation: escape rate within 1-4% of quadrature at dt=0.003, 3-18% at "
        "0.0125; stationary density TV 0.01-0.02 at 0.0125). All multiplicative data (stationary_diag kind=mul, "
        "alloc_vs_time kind=mul, multiplicative_floor) are new. (2) Additive/input data were produced by projected "
        "Euler (clip), which biases floored synapses low (floor mean 0.012 vs analytic 0.021 at sigma=0.1, theta=0.2). "
        "All add/inp/inpn data here (stationary_diag, alloc_vs_time, rates, arrhenius, planted, network) are reruns "
        "with a reflected Strang scheme that reproduces the analytic floor mean at dt=0.05 (0.0207 vs 0.0210). "
        "scheme_check_headline compares reflected (dt 0.05 and 0.0125) with the original values. "
        "dt_check_original_scheme is the pre-revision dt check (projected Euler). theory_checks.rayleigh_vs_floor "
        "added: R(h) is non-decreasing only for the zero-floor flow.")
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
    elif arg == "L2":
        partL2()
    elif arg == "V":
        partV()
    elif arg == "A2":
        partA2()
    elif arg == "M2":
        partM2()
    elif arg == "H":
        partH()
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
