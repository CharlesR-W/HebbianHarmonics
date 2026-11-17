"""Q5: weight-space readout of Sigma (SPEC §3 / §7.3-7.4), across theta and during training.

Estimator (my own numpy version): masked ridge least squares over symmetric C.
  active I-B edge (w_ib > 1.5 wmin):  (H C)_ib = theta + (lam/eta) w_ib
  active I-I edge (w_ij > 1.5 wmin):  gamma * h_i^T C h_j = theta + (lam/eta) w_ij
Settings: known (true theta, lam/eta; full-covariance score, no projection), scale (same C_hat,
best scale), contrast (theta fitted jointly as a free offset; score P C P, P = I - 11^T/B).
BW: PSD-clip C_hat first (negative mass reported), BW^2/tr(Sigma), scale-fitted except in 'known'.
Baselines: identity, diag(Sigma), rank-k truncation of Sigma (k = #clusters), the note's naive
pseudo-inverse formula, readout of the untrained net (equilibrium violated), and the 'oracle mask':
Sigma with entries kept only where both inputs share a learned part's support.

Output: site/data/sweeps/readout.json
"""
import numpy as np

from hebb.core import harmonic
from hebb.sweeps.common import (Timer, center, cluster_embed, cov_block, cov_hier, cov_powerlaw, cov_powerlaw_pos,
                                pmap, save_json, score3, simulate)

N = 120
WMIN = 1e-4
GAMMA = 1.0


def _coef_rows(A, Bm, iu):
    M = np.einsum("ni,nj->nij", A, Bm)
    M = M + M.transpose(0, 2, 1)
    d = np.arange(A.shape[1])
    M[:, d, d] *= 0.5
    return M[:, iu[0], iu[1]]


def readout_ls(Wii, Wib, theta, gamma=GAMMA, wmin=WMIN, use=("ii", "ib"), fit_theta=False, ridge=1e-8,
               active=1.5):
    Nn, B = Wib.shape
    H = harmonic(np.maximum(Wii, wmin * (1 - np.eye(Nn))), np.maximum(Wib, wmin))
    iu = np.triu_indices(B)
    X, y = [], []
    if "ib" in use:
        I, Bk = np.nonzero(Wib > active * wmin)
        if len(I):
            X.append(np.c_[_coef_rows(H[I], np.eye(B)[Bk], iu), -np.ones(len(I))])
            y.append(Wib[I, Bk])
    if "ii" in use and gamma > 0:
        I, J = np.nonzero(np.triu(Wii > active * wmin, 1))
        if len(I):
            X.append(np.c_[gamma * _coef_rows(H[I], H[J], iu), -np.ones(len(I))])
            y.append(Wii[I, J])
    if not X:
        return np.zeros((B, B)), dict(n_eq=0, rank=0, cond=None)
    X = np.vstack(X); y = np.concatenate(y)
    if fit_theta:
        Xf = X
    else:
        y = y + theta  # move known theta to the rhs:  coef.c = theta + w
        Xf = X[:, :-1]
    # dedupe for rank/conditioning report (many clusters give identical rows)
    s = np.linalg.svd(Xf, compute_uv=False)
    rank = int((s > s[0] * 1e-8).sum())
    G = Xf.T @ Xf
    sol = np.linalg.solve(G + ridge * np.trace(G) / len(G) * np.eye(len(G)), Xf.T @ y)
    C = np.zeros((B, B)); C[iu] = sol[:len(iu[0])]; C = C + C.T - np.diag(np.diag(C))
    nonzero = s[s > s[0] * 1e-8]
    return C, dict(n_eq=int(len(y)), rank=rank, n_params=len(iu[0]) + int(fit_theta),
                   cond=float(nonzero[0] / nonzero[-1]), theta_hat=float(sol[-1]) if fit_theta else None)


def parts_readout(Wii, Wib, theta, wmin=WMIN, tol=0.1):
    """Parts-sum readout: at a fixed point the weights determine only the parts h_k and their
    supra-threshold eigenvalues c_k = (lam/eta) * sum_b w_ib (active edges of any member).
    C_parts = theta 11^T + sum_k c_k h_k h_k^T / |h_k|^2  (satisfies every active-edge equation
    exactly when supports are disjoint). Returns (known-theta version, theta-free version)."""
    Nn, B = Wib.shape
    H = harmonic(np.maximum(Wii, wmin * (1 - np.eye(Nn))), np.maximum(Wib, wmin))
    lab, sizes, cents = cluster_embed(H, tol)
    C = np.zeros((B, B))
    for k, n in enumerate(sizes):
        mem = lab == k
        act = Wib[mem] > 1.5 * wmin
        if not act.any():
            continue
        ck = float(np.mean((Wib[mem] * act).sum(1)))
        h = cents[k]
        C += ck * np.outer(h, h) / (h @ h)
    return C + theta, C


def naive(Wii, Wib, theta, gamma=GAMMA, wmin=WMIN):
    Nn = len(Wii)
    H = harmonic(np.maximum(Wii, wmin * (1 - np.eye(Nn))), np.maximum(Wib, wmin))
    R = (theta + Wii) / gamma
    np.fill_diagonal(R, 0)
    np.fill_diagonal(R, R.sum(1) / (Nn - 1))
    Hp = np.linalg.pinv(H, rcond=1e-8)
    return Hp @ R @ Hp.T


def covs():
    return {"block12": cov_block([4, 4, 4], 0.5, 0.1)[0], "hier8": cov_hier((0.6, 0.3, 0.05), (2, 2, 2)),
            "powerlaw12": cov_powerlaw(12, 1.0, seed=0), "powerlaw_pos12": cov_powerlaw_pos(12, 1.0, seed=0)}


THETAS = [-0.05, 0.0, 0.03, 0.07, 0.13, 0.18, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]  # avoids exact correlation levels


def score_state(Wii, Wib, S, theta, K):
    out = {}
    for fam in (("ii", "ib"), ("ib",), ("ii",)):
        Ck, info = readout_ls(Wii, Wib, theta, use=fam)
        Cf, infof = readout_ls(Wii, Wib, theta, use=fam, fit_theta=True)
        sc = score3(Ck, Cf, S)
        out["+".join(fam)] = dict(**sc, info=info, info_fit=infof)
    Cn = naive(Wii, Wib, theta)
    out["naive"] = score3(Cn, Cn, S)
    Cp, Cp0 = parts_readout(Wii, Wib, theta)
    out["parts"] = score3(Cp, Cp0, S)
    return out


def oracle_mask(H, S):
    l, sizes, cents = cluster_embed(H, 0.1)
    Mk = np.zeros_like(S)
    for k, n in enumerate(sizes):
        if n >= 2:
            sup = cents[k] > 0.05 * cents[k].max()
            Mk = np.maximum(Mk, np.outer(sup, sup))
    return S * Mk


def baselines(S, k):
    B = len(S)
    ev, V = np.linalg.eigh(S)
    k = max(1, min(k, B))
    rk = (V[:, -k:] * ev[-k:]) @ V[:, -k:].T
    return {"identity": score3(np.eye(B), np.eye(B), S), "diag": score3(np.diag(np.diag(S)), np.diag(np.diag(S)), S),
            f"rank_k": score3(rk, rk, S)}


def job(args):
    name, theta, seed = args
    S = covs()[name]
    r = simulate(S, N, theta, gamma=GAMMA, wmin=WMIN, seed=seed, Tmax=30000)
    l, sizes, cents = cluster_embed(r["H"], 0.1)
    K = int((sizes >= 3).sum())
    sc = score_state(r["Wii"], r["Wib"], S, theta, K)
    om = oracle_mask(r["H"], S)
    sc["oracle_mask"] = score3(om, om, S)
    sc["baselines"] = baselines(S, K)
    # untrained net (equilibrium violated)
    r0 = simulate(S, N, theta, gamma=GAMMA, wmin=WMIN, seed=seed, Tmax=1, check=1)
    sc["untrained"] = score_state(r0["Wii"], r0["Wib"], S, theta, K)["ii+ib"]
    return dict(cov=name, theta=theta, seed=seed, K=K, res=r["res"], scores=sc)


def train_job(args):
    name, theta, seed = args
    S = covs()[name]
    out = []
    st = None
    t = 0
    for T in [1, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 400, 640, 1280, 2560, 5120]:
        r = simulate(S, N, theta, gamma=GAMMA, wmin=WMIN, seed=seed, state=st, Tmax=T - t, check=T - t)
        st = (r["Wii"], r["Wib"], r["Mii"], r["Mib"]); t = T
        sc = score_state(r["Wii"], r["Wib"], S, theta, 3)
        out.append(dict(step=T, res=r["res"], **{k: sc[k] for k in ("ii+ib", "ib", "ii")}))
    return dict(cov=name, theta=theta, seed=seed, traj=out)


def mask_job(args):
    name, theta, p, seed = args
    S = covs()[name]
    r = simulate(S, N, theta, gamma=GAMMA, wmin=WMIN, seed=seed, p=p, Tmax=30000)
    Ck, info = readout_ls(r["Wii"], r["Wib"], theta)
    Cf, _ = readout_ls(r["Wii"], r["Wib"], theta, fit_theta=True)
    return dict(cov=name, theta=theta, p=p, seed=seed, res=r["res"], scores=score3(Ck, Cf, S), info=info)


def main():
    jobs = [(n, th, s) for n in covs() for th in THETAS for s in (0, 1)]
    tj = [(c, th, s) for c, th in (("block12", 0.05), ("block12", 0.35), ("block12", 0.45), ("block12", 0.65),
                                    ("powerlaw12", 0.13), ("powerlaw_pos12", 0.25)) for s in (0, 1, 2)]
    mj = [(c, th, p, s) for c, th in (("block12", 0.35), ("powerlaw12", 0.25)) for p in (1.0, 0.5, 0.25) for s in (0, 1)]
    with Timer(f"readout ({len(jobs) + len(tj)})"):
        R = pmap(_d, [("t", j) for j in tj] + [("m", j) for j in mj] + [("r", j) for j in jobs])
    rows = [r for t, r in R if t == "r"]
    traj = [r for t, r in R if t == "t"]
    masks = [r for t, r in R if t == "m"]
    for m in masks:
        print("mask", m["cov"], m["theta"], m["p"], m["seed"], "rank", m["info"]["rank"], "known cos %.3f relF %.3f contrast cos %.3f bw %.3f"
              % (m["scores"]["known"]["cos"], m["scores"]["known"]["relF"], m["scores"]["contrast"]["cos"], m["scores"]["contrast"]["bw"]))
    for n in covs():
        print(n)
        for th in THETAS:
            rr = [r for r in rows if r["cov"] == n and r["theta"] == th]
            f = lambda key, setting, m: np.mean([r["scores"][key][setting][m] for r in rr])
            print(f"  th={th:+.2f} K={[r['K'] for r in rr]} res={max(r['res'] for r in rr):.1e} "
                  f"contrastBW: both={f('ii+ib','contrast','bw'):.3f} ib={f('ib','contrast','bw'):.3f} ii={f('ii','contrast','bw'):.3f} "
                  f"naive={f('naive','contrast','bw'):.3f} parts={f('parts','contrast','bw'):.3f}/{f('parts','known','cos'):.3f} untrained={f('untrained','contrast','bw'):.3f} oracle={f('oracle_mask','contrast','bw'):.3f} "
                  f"diag={f('baselines','diag','contrast') if False else np.mean([r['scores']['baselines']['diag']['contrast']['bw'] for r in rr]):.3f} "
                  f"| knownRelF={f('ii+ib','known','relF'):.3f} cos={f('ii+ib','contrast','cos'):.3f} "
                  f"rank={rr[0]['scores']['ii+ib']['info']['rank']}/{rr[0]['scores']['ii+ib']['info']['n_params']}")
    for tr in traj:
        print("traj", tr["cov"], tr["theta"], [(p["step"], round(p["ii+ib"]["contrast"]["bw"], 3), round(p["ii+ib"]["contrast"]["cos"], 3)) for p in tr["traj"]])
    meta = dict(script="hebb/sweeps/s6_readout.py", N=N, wmin=WMIN, gamma=GAMMA, thetas=THETAS, seeds=[0, 1],
                params=dict(lam=1, eta=1, dt=0.05, ii_scale=0.0, ib_scale=0.2, p=1.0, conv_tol=1e-5),
                estimator="masked ridge LS over symmetric C (ridge 1e-8*tr(G)/dim), active = w > 1.5*wmin; families "
                          "'ii+ib', 'ib', 'ii'; 'naive' = H^+ R H^+T with R = theta + w_II (diag filled by row mean)",
                settings=dict(known="true theta and lam/eta; scored on full Sigma: relF=||C-S||/||S||, bw=BW^2/trS (no scale fit)",
                              scale="same C_hat; bw scale-fitted = 1 - F^2/(trC trS); cos = Frobenius cosine",
                              contrast="theta fitted as free offset; scored on P C P vs P S P, scale fitted"),
                bw_note="C_hat PSD-clipped before BW; 'neg' = fraction of spectral mass removed",
                baselines="identity, diag(Sigma), rank-k truncation (k=#clusters>=3), untrained net (equilibrium violated), "
                          "oracle_mask = Sigma restricted to within-part-support pairs",
                parts="parts-sum readout theta 11^T + sum_k c_k h_k h_k^T/|h_k|^2 with c_k = summed active I-B weight of a member "
                      "(contrast version drops theta 11^T)",
                info="rank/n_params of the measurement operator on symmetric matrices (+theta if fitted), cond = ratio of "
                     "largest to smallest nonzero singular value")
    meta["training"] = "readout applied along training (equilibrium assumption violated before convergence); seeds 0-2; steps of dt=0.05"
    meta["masks"] = "labelled comparison: fixed sparse potential-synapse mask p (I-I and I-B) at equilibrium, ii+ib readout"
    save_json("readout", dict(rows=rows, training=traj, masks=masks), meta)


def _d(a):
    t, j = a
    return t, dict(r=job, t=train_job, m=mask_job)[t](j)


if __name__ == "__main__":
    main()
