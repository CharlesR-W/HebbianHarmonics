"""Q-eigen (T3, SPEC §3): population response operator S f = sum_i F~_i <F~_i, f>, its
eigenfunctions psi_a, vs (i) cluster prototypes and (ii) eigenmodes of Sigma_g.

Derived (linear case, zero-mean g): for linear functionals f(g) = theta.g, <f,f'> = theta^T Sigma theta',
and S acts as theta -> (H^T H) Sigma theta. In whitened coordinates phi = Sigma^{1/2} theta the operator
is Sigma^{1/2} K Sigma^{1/2} with K = H^T H (the population's coverage of input space). So the
population eigenfunctions are Sigma's eigenmodes iff K commutes with Sigma; then q_a = lambda_a k_a with
k_a = sum_i (h_i . v_a)^2.

Parts:
  A  the explicit shared+contrast example F_A = s + c, F_B = s - c (two clusters), vs the 2x2 theory;
  B  trained networks on several input presets (linear and rectifying): prototype orthogonality,
     psi vs prototypes, psi vs Sigma modes, q vs nu*n, commutator ||[K, Sigma]||.
All population quantities on the evaluation bank.

Run: uv run python -m hebb.sweeps2.eigen   -> site/data/sweeps2/eigen.json
"""
import hebb.sweeps2  # noqa: F401

import numpy as np

from hebb import onsite as o
from hebb import onsite_analysis as oa


def unit_cols(X):
    Xc = X - X.mean(0)
    return Xc / np.maximum(np.linalg.norm(Xc, axis=0), 1e-300)


# ----------------------------------------------------------------------------- A: s +- c
def part_a():
    rows = []
    for sc2 in (0.1, 0.25, 0.5, 1.0):
        Sig = np.array([[1 + sc2, 1 - sc2], [1 - sc2, 1 + sc2]])        # g1 = s + c, g2 = s - c, var s = 1
        for nA, nB in ((12, 12), (16, 8), (20, 4)):
            for mix in (0.0, 0.2):
                seed = 0
                rng = oa.rng_for(seed, 0)
                eps = rng.uniform(0, mix, nA + nB)
                A = np.zeros((nA + nB, 2))
                A[:nA, 0], A[:nA, 1] = 1 - eps[:nA], eps[:nA]
                A[nA:, 1], A[nA:, 0] = 1 - eps[nA:], eps[nA:]
                G, Ge = oa.banks(Sig, seed)
                b = 0.5 * sc2                                              # < 2 sc2: A and B stay apart
                net = o.Net(A=A, t=0.0, klo=0.0, khi=0.0)
                r = oa.solve(net, G, b, 1.0)
                lab = o.components(net.W)
                Ue = o.equilibrate(net, Ge)
                pm = o.population_modes(Ue)
                Psi = pm["Psi"][:, :2]
                # s and c on the eval bank: s = (g1+g2)/2, c = (g1-g2)/2
                sfun, cfun = (Ge[:, 0] + Ge[:, 1]) / 2, (Ge[:, 0] - Ge[:, 1]) / 2
                E = unit_cols(np.stack([sfun, cfun], 1))
                protos = unit_cols(np.stack([Ue[:, lab == k].mean(1) for k in range(lab.max() + 1)], 1))
                cos_sc = np.abs(unit_cols(Psi).T @ E)                       # rows psi, cols (s, c)
                cos_pr = np.abs(unit_cols(Psi).T @ protos)
                # 2x2 theory with identical members: nu = |p|^2, rho = cos(p_A, p_B)
                nu, rho = 1 + sc2, (1 - sc2) / (1 + sc2)
                disc = np.sqrt((nA - nB) ** 2 + 4 * nA * nB * rho ** 2)
                q_th = [nu / 2 * (nA + nB + disc), nu / 2 * (nA + nB - disc)]
                Hm = o.harmonic(net)
                q_an = np.sort(np.linalg.eigvalsh(Hm @ Sig @ Hm.T))[::-1][:2]
                rows.append(dict(sc2=sc2, nA=nA, nB=nB, mix=mix, K=int(lab.max() + 1), residual=r["residual"],
                                 q=pm["q"][:2], q_analytic=q_an, q_theory_identical=q_th, nu_n=[nu * nA, nu * nB],
                                 cos_psi1_s=cos_sc[0, 0], cos_psi2_c=cos_sc[1, 1],
                                 cos_psi_proto_best=cos_pr.max(1), proto_cos=float(abs(protos[:, 0] @ protos[:, -1])),
                                 sigma_mode_is_s_c=True))
    return rows


# ----------------------------------------------------------------------------- B: presets
def sigma_blocks_corr(B=12, var=(2.0, 1.0, 0.5, 0.25), rho=0.8, rho_x=0.4):
    S = oa.sigma_blocks(B, var, 1.0)
    d = np.sqrt(np.diag(S))
    m = B // len(var)
    blk = np.arange(B) // m
    R = np.where(blk[:, None] == blk[None], rho, rho_x)
    np.fill_diagonal(R, 1.0)
    return R * np.outer(d, d)


N, B = 48, 12
PRESETS = dict(
    blocks_blockatt=(oa.sigma_blocks(B), "block"),
    blocks_randatt=(oa.sigma_blocks(B), "random"),
    blockscorr_blockatt=(sigma_blocks_corr(B), "block"),
    power_randatt=(oa.sigma_power(B), "random"),
)
POTS = dict(linear=oa.LINEAR, rect=oa.RECT)
BGRID = np.round(np.geomspace(0.01, 0.5, 6), 4)
SEEDS = [0, 1, 2]


def population_compare(net, Ue, Ge, Sigma, lab, linear):
    pm = o.population_modes(Ue)
    q, Psi = pm["q"], unit_cols(pm["Psi"])
    K = lab.max() + 1
    sizes = np.bincount(lab)
    multi = np.nonzero(sizes > 1)[0]
    P = unit_cols(np.stack([Ue[:, lab == k].mean(1) for k in range(K)], 1))
    nu = np.array([((Ue[:, lab == k] - Ue[:, lab == k].mean(0)) ** 2).mean(0).mean() for k in range(K)])
    # prototype orthogonality among multi-member clusters
    Pm = P[:, multi]
    Gp = np.abs(Pm.T @ Pm) if len(multi) > 1 else np.zeros((1, 1))
    off = Gp[~np.eye(len(Gp), dtype=bool)] if len(multi) > 1 else np.array([0.0])
    r = max(len(multi), 1)
    cos_proto = np.abs(Psi[:, :r].T @ P).max(1)                  # best prototype match per psi
    lam, Vs = oa.eig_desc(Sigma)
    Es = unit_cols(Ge @ Vs)                                      # Sigma eigenfunctions on the eval bank
    rs = min(r, B)
    cos_sig = np.abs(Psi[:, :rs].T @ Es).max(1)
    # q vs nu*n (sorted), and vs the identical-copies cluster Gram
    nun = np.sort(nu * sizes)[::-1][:r]
    q_rel_err = float(np.abs(q[:r] - nun).sum() / q[:r].sum())
    Gc = (P * np.sqrt(nu * sizes)).T @ (P * np.sqrt(nu * sizes))
    q_ident = np.sort(np.linalg.eigvalsh(Gc))[::-1][:r]
    q_ident_rel_err = float(np.abs(q[:r] - q_ident).sum() / q[:r].sum())
    st = o.cluster_stats(Ue, lab)
    dm = [c["diam_max"] for c in st if c["size"] > 1]
    dmean = [c["diam_mean"] for c in st if c["size"] > 1]
    out = dict(K=int(K), K_multi=int(len(multi)), proto_cos_mean=float(off.mean()), proto_cos_max=float(off.max()),
               diam_max=float(max(dm)) if dm else 0.0, diam_mean=float(np.mean(dmean)) if dmean else 0.0,
               proto_sd_mean=float(np.mean([c["proto"].std() for c in st if c["size"] > 1])) if dm else 0.0,
               psi_proto_cos_mean=float(cos_proto.mean()), psi_sigma_cos_mean=float(cos_sig.mean()),
               q_vs_nun_rel_err=q_rel_err, q_vs_identical_rel_err=q_ident_rel_err,
               frac_var_top_r=float(q[:r].sum() / q.sum()))
    if linear:
        H = o.harmonic(net)
        Kc = H.T @ H
        comm = Kc @ Sigma - Sigma @ Kc
        out["commutator_rel"] = float(np.linalg.norm(comm) / (np.linalg.norm(Kc) * np.linalg.norm(Sigma)))
        # exact identity: neuron Gram H Sigma H^T and Sigma^1/2 K Sigma^1/2 share their nonzero spectrum
        Sh = oa.psd_sqrt(Sigma)
        e1 = np.sort(np.linalg.eigvalsh(H @ Sigma @ H.T))[::-1][:B]
        e2 = np.sort(np.linalg.eigvalsh(Sh @ Kc @ Sh))[::-1]
        out["spectrum_identity_err"] = float(np.abs(e1 - e2).max() / e2[0])
        out["eval_vs_Sigma_q_rel_err"] = float(np.abs(q[:B] - e2).max() / e2[0])
    else:
        h = len(Ge) // 2
        aff = oa.affine_fit_err(pm["Psi"][:h, :r], Ge[:h], pm["Psi"][h:, :r], Ge[h:])
        out["psi_nonlinear_frac"] = float(aff.mean())
    return out


def unit_b(args):
    preset, pot, seed = args
    Sigma, att = PRESETS[preset]
    G, Ge = oa.banks(Sigma, seed)
    if att == "block":
        A, _ = oa.block_attachments(N, B, 4, 3, seed)
    else:
        A = o.make_attachments(N, B, 3, oa.rng_for(seed, 0))
    out = []
    for b in BGRID:
        net = oa.make_net(N, B, seed, A=A, **POTS[pot])
        r = oa.solve(net, G, b, 1.0)
        Ue = o.equilibrate(net, Ge)
        lab = o.components(net.W)
        rec = population_compare(net, Ue, Ge, Sigma, lab, pot == "linear")
        rec.update(residual=r["residual"], converged=r["converged"])
        out.append(rec)
    print(f"eigen {preset} {pot} seed={seed}", flush=True)
    return dict(preset=preset, pot=pot, seed=seed, runs=out)


def main():
    pa = part_a()
    print("part A done")
    for r in pa:
        print(r["sc2"], r["nA"], r["nB"], r["mix"], np.round(r["q"], 3), np.round(r["q_theory_identical"], 3),
              round(r["cos_psi1_s"], 3), round(r["cos_psi2_c"], 3), np.round(r["cos_psi_proto_best"], 3))
    jobs = [(p, pt, s) for p in PRESETS for pt in POTS for s in SEEDS]
    jobs.sort(key=lambda j: j[1] != "rect")
    res = oa.pool_map(unit_b, jobs)
    keys = sorted(set().union(*[set(x.keys()) for u in res for x in u["runs"]]))
    runs = [dict(preset=u["preset"], pot=u["pot"], seed=u["seed"],
                 **{k: [x.get(k) for x in u["runs"]] for k in keys}) for u in res]
    meta = dict(
        script="hebb/sweeps2/eigen.py",
        description="Q-eigen: population eigenfunctions vs cluster prototypes vs Sigma_g eigenmodes; the explicit "
                    "shared+contrast example; linear-case identity Sigma^1/2 H^T H Sigma^1/2.",
        config=dict(partA="B=2, g1 = s+c, g2 = s-c, var s = 1, var c = sc2; nA neurons on input 1, nB on input 2 "
                          "(attachment leak 'mix': share eps~U(0,mix) on the other input); b = sc2/2, alpha = 1",
                    N=N, B=B, presets={k: [v[1], k.split("_")[0]] for k, v in PRESETS.items()},
                    blockscorr="within-block corr .8, cross-block corr .4, block variances 2/1/.5/.25",
                    pots=POTS, b_grid=BGRID, alpha=1.0, S_train=oa.S_TRAIN, S_eval=oa.S_EVAL),
        seeds=dict(partA=[0], partB=SEEDS),
        fields={
            "partA[].q": "top two population eigenvalues (response variance units x neurons), eval bank",
            "partA[].q_analytic": "top two eigenvalues of H Sigma H^T (exact population spectrum, no sampling)",
            "partA[].q_theory_identical": "nu/2 (nA+nB +- sqrt((nA-nB)^2 + 4 nA nB rho^2)), identical members, nu = 1+sc2, rho = (1-sc2)/(1+sc2)",
            "partA[].cos_psi1_s / cos_psi2_c": "|cos| between the top / second eigenfunction and the shared s / contrast c (which are exactly Sigma_g's eigenmodes)",
            "partA[].cos_psi_proto_best": "per eigenfunction, best |cos| with a cluster prototype",
            "partA[].proto_cos": "|cos| between the two cluster prototypes",
            "runs[].<key>[k]": "value at b_grid[k]",
            "diam_max, diam_mean": "max / mean (over multi-member components) RMS distance of members to the prototype, eval bank (response units)",
            "proto_sd_mean": "mean prototype standard deviation over multi-member components (scale for the diameters)",
            "proto_cos_mean/max": "mean/max |cos| between distinct multi-member cluster prototypes (centred)",
            "psi_proto_cos_mean": "mean over the top r = K_multi eigenfunctions of the best |cos| with any prototype",
            "psi_sigma_cos_mean": "mean over the top r eigenfunctions of the best |cos| with any Sigma_g eigenfunction v_b.g",
            "q_vs_nun_rel_err": "sum_a |q_a - (nu n)_(a)| / sum q over the top r (sorted), nu = mean member variance of a cluster",
            "q_vs_identical_rel_err": "same vs the eigenvalues of the cluster Gram with identical members (accounts for prototype overlap)",
            "commutator_rel": "||[H^T H, Sigma]||_F / (||H^T H|| ||Sigma||) (linear only; 0 => psi = Sigma modes)",
            "spectrum_identity_err": "max |eig(H Sigma H^T) - eig(Sigma^1/2 K Sigma^1/2)| / top (exact identity)",
            "eval_vs_Sigma_q_rel_err": "eval-bank population spectrum vs the analytic one (sampling)",
            "psi_nonlinear_frac": "rect only: mean held-out relative MSE of an affine fit of the top r eigenfunctions from g",
            "frac_var_top_r": "fraction of population variance in the top r eigenvalues",
        })
    oa.dump("eigen", meta, dict(partA=pa, runs=runs), sig=4)


if __name__ == "__main__":
    main()
