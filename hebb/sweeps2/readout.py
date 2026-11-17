"""Q-readout (T4, SPEC §3): reading the input covariance from the weights alone.

Linear, zero potential: at the fixed point every live edge gives D_ij^2 = 2(b - alpha w_ij) and every
dead pair D^2 >= 2b, with D_ij^2 = d^T C_hat d, d = h_i - h_j (H = L^-1 A from the weights). H 1 = 1, so
d^T 1 = 0 and C + 1a^T + a1^T is invisible: the target is P C P, P = I - 11^T/B. We work in an
orthonormal contrast basis Q (B x (B-1), Q^T 1 = 0), unknown Y = Q^T C Q (symmetric), measurement
operator M: Y -> (c_e^T Y c_e), c_e = Q^T d_e, written in an orthonormal svec basis.
Recovery: masked ridge LS (min-norm on the identifiable subspace), optional hinge for dead pairs,
PSD projection. Scores vs P C_hat P (first) and P Sigma_g P (second). Baselines: identity and
diagonal covariances fitted to the same measurements, and shuffled weights.
With a quadratic leak (klo = khi = k, t = 0) H 1 != 1: the full C enters; rank/conditioning reported.
Nonlinear (rectifying): the weights give D on live edges exactly and lower bounds on dead pairs;
shortest-path completion inside components, MDS vs the true geometry; nothing further.

Run: uv run python -m hebb.sweeps2.readout   -> site/data/sweeps2/readout.json
"""
import hebb.sweeps2  # noqa: F401

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import procrustes

from hebb import onsite as o
from hebb import onsite_analysis as oa

B = 12
SIGMAS = dict(blocks=oa.sigma_blocks(B), power=oa.sigma_power(B))
BGRID = np.round(np.geomspace(0.003, 1.0, 8), 4)
ALPHA = 1.0
NS = (40, 80)
SEEDS = [0, 1, 2]
EPS = 1e-9


def svec_rows(Cv):
    """Rows m(c) with m(c).svec(Y) = c^T Y c, orthonormal svec (off-diagonals scaled by sqrt 2)."""
    n = Cv.shape[1]
    iu = np.triu_indices(n)
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return Cv[:, iu[0]] * Cv[:, iu[1]] * scale, iu, scale


def smat(s, n, iu, scale):
    Y = np.zeros((n, n))
    Y[iu] = s / scale
    return Y + np.triu(Y, 1).T


def svec(Y, iu, scale):
    return Y[iu] * scale


def psd(Y):
    lam, V = np.linalg.eigh((Y + Y.T) / 2)
    return (V * np.clip(lam, 0, None)) @ V.T


def score(Y, T, psd_ok):
    out = dict(cos=oa.fro_cos(Y, T), rel_err=float(np.linalg.norm(Y - T) / np.linalg.norm(T)))
    if psd_ok:
        out["bures_rel"] = oa.bures(Y, T) / float(np.sqrt(np.trace(T)))
    return out


def recover(Mm, y, n, iu, scale, dead_rows=None, two_b=None):
    sv = np.linalg.svd(Mm, compute_uv=False) if len(Mm) else np.zeros(1)
    rho = 1e-10 * max(sv[0] ** 2, 1e-300)
    s = np.linalg.solve(Mm.T @ Mm + rho * np.eye(Mm.shape[1]), Mm.T @ y) if len(Mm) else np.zeros(len(iu[0]))
    out = dict(ridge=smat(s, n, iu, scale))
    if dead_rows is not None and len(dead_rows):
        def f(x):
            r = Mm @ x - y
            h = np.maximum(two_b - dead_rows @ x, 0)
            return float(r @ r + rho * x @ x + h @ h), 2 * (Mm.T @ r + rho * x - dead_rows.T @ h)
        x = minimize(f, s, jac=True, method="L-BFGS-B", options=dict(maxiter=2000)).x
        out["hinge"] = smat(x, n, iu, scale)
    return out


def ident_err(Mm, s_hat, s_true, tol=1e-8):
    """Error on the identifiable subspace (row space of M at relative tolerance tol) and the
    fraction of ||s_true||^2 lying in it."""
    if not len(Mm):
        return float("nan"), 0.0
    _, sv, Vt = np.linalg.svd(Mm, full_matrices=False)
    Vr = Vt[sv > tol * sv[0]]
    pt, ph = Vr @ s_true, Vr @ s_hat
    return float(np.linalg.norm(ph - pt) / max(np.linalg.norm(pt), 1e-300)), float((pt ** 2).sum() / (s_true ** 2).sum())


def lin_unit(args):
    sig, N, seed, leak = args
    Sigma = SIGMAS[sig]
    G, _ = oa.banks(Sigma, seed)
    Chat = G.T @ G / len(G)
    out = []
    for b in BGRID:
        k = leak_vector(leak, N, seed)
        net = oa.make_net(N, B, seed, klo=k, khi=k)
        r = oa.solve(net, G, b, ALPHA)
        out.append(analyse(net, r, b, Sigma, Chat, leak, seed))
    print(f"readout {sig} N={N} seed={seed} leak={leak}", flush=True)
    return dict(sigma=sig, N=N, seed=seed, leak=leak, runs=out)


def leak_vector(leak, N, seed):
    """'0': no potential; 'u<k>': uniform quadratic leak k; 'h<k>': k_i ~ U(0, k) per neuron."""
    if leak == "0":
        return np.zeros(N)
    if leak[0] == "u":
        return np.full(N, float(leak[1:]))
    return oa.rng_for(seed, 4).uniform(0, float(leak[1:]), N)


def response_matrix(net):
    """u = H g for the quadratic potential V_i = k_i/2 u^2 (klo = khi = k_i, t = 0)."""
    return np.linalg.solve(o.laplacian(net) + np.diag(net.klo), net.A)


def analyse(net, r, b, Sigma, Chat, leak, seed, keep_spectra=False):
    N = net.N
    H = response_matrix(net)
    iu2 = np.triu_indices(N, 1)
    w = net.W[iu2]
    live = w > EPS
    d = H[iu2[0]] - H[iu2[1]]
    if leak == "0":
        basis = oa.contrast_basis(B)[:, 1:]                   # contrast space, P C P
    else:
        basis = np.eye(B)                                      # full C enters when H 1 != 1
    n = basis.shape[1]
    Cv = d @ basis
    rows, iu, scale = svec_rows(Cv)
    Mm, y = rows[live], 2 * (b - ALPHA * w[live])
    T1, T2 = basis.T @ Chat @ basis, basis.T @ Sigma @ basis
    s1 = svec(T1, iu, scale)
    sv = np.linalg.svd(Mm, compute_uv=False) if live.any() else np.zeros(1)
    fp_err = float(np.abs(y - Mm @ s1).max()) if live.any() else 0.0
    rec = recover(Mm, y, n, iu, scale, dead_rows=rows[~live], two_b=2 * b)
    cs = oa.cluster_summary(net.W, r["U"], b)
    res = dict(b=float(b), n_live=int(live.sum()), dim=int(len(s1)), residual=r["residual"], converged=r["converged"],
               K=cs["K"], K_multi=cs["K_multi"], largest=cs["largest"], diam_max=cs["diam_max"],
               span_over_range=cs["span_over_range"],
               rank_1e8=int((sv > 1e-8 * sv[0]).sum()) if live.any() else 0,
               rank_1e4=int((sv > 1e-4 * sv[0]).sum()) if live.any() else 0,
               sv_max=float(sv[0]), sv_min_nonzero=float(sv[sv > 1e-8 * sv[0]].min()) if live.any() else 0.0,
               ones_leak=float(np.abs(d.sum(1)).max()),
               err_fixed_point=fp_err,
               err_sampling=float(np.linalg.norm(T1 - T2) / np.linalg.norm(T2)))
    s_hat = svec(rec["ridge"], iu, scale)
    res["err_inverse_identifiable"], res["identifiable_frac"] = ident_err(Mm, s_hat, s1, tol=1e-4)
    res["identifiable_frac_1e8"] = ident_err(Mm, s_hat, s1, tol=1e-8)[1]
    if leak != "0":
        Qc = oa.contrast_basis(B)[:, 1:]
        res["ridge_psd_contrast_vs_Chat"] = score(Qc.T @ psd(rec["ridge"]) @ Qc, Qc.T @ Chat @ Qc, True)
    # sensitivity: multiplicative noise on the live weights; the WHOLE readout (H, d, operator and
    # measurement values) is recomputed from the noisy weight matrix
    for eps in (1e-6, 1e-4, 1e-2):
        if live.any():
            Wn = net.W * (1 + eps * oa.rng_for(seed, 8).standard_normal(net.W.shape))
            Wn = np.triu(Wn, 1); Wn = Wn + Wn.T
            nn = o.Net(A=net.A, t=0.0, klo=net.klo, khi=net.khi, W=Wn)
            Hn = response_matrix(nn)
            rn, _, _ = svec_rows((Hn[iu2[0]] - Hn[iu2[1]]) @ basis)
            wnl = Wn[iu2][live]
            Yn = recover(rn[live], 2 * (b - ALPHA * wnl), n, iu, scale)["ridge"]
            res[f"noise{eps:g}_ridge_psd_vs_Chat_cos"] = oa.fro_cos(psd(Yn), T1)
    for name, Y in (("ridge", rec["ridge"]), ("ridge_psd", psd(rec["ridge"])),
                    ("hinge_psd", psd(rec["hinge"]) if "hinge" in rec else psd(rec["ridge"]))):
        for tname, T in (("Chat", T1), ("Sigma", T2)):
            res[f"{name}_vs_{tname}"] = score(Y, T, name != "ridge")
    # baselines fitted to the same live measurements
    if live.any():
        q = (Cv[live] ** 2).sum(1)
        c_id = float((y * q).sum() / (q * q).sum())
        Y_id = c_id * np.eye(n)
        delta, _ = nnls(d[live] ** 2, y)
        Y_dg = basis.T @ np.diag(delta) @ basis
    else:
        Y_id = Y_dg = np.zeros((n, n))
    for name, Y in (("base_identity", Y_id), ("base_diag", Y_dg)):
        for tname, T in (("Chat", T1), ("Sigma", T2)):
            res[f"{name}_vs_{tname}"] = score(psd(Y), T, True)
    # shuffled-weights null: same weight values permuted over pairs, formula applied blindly
    rng = oa.rng_for(seed, 9)
    Ws = np.zeros_like(net.W)
    Ws[iu2] = rng.permutation(w)
    Ws = Ws + Ws.T
    sh = o.Net(A=net.A, t=0.0, klo=net.klo, khi=net.khi, W=Ws)
    Hs = response_matrix(sh)
    ds = (Hs[iu2[0]] - Hs[iu2[1]]) @ basis
    rs, _, _ = svec_rows(ds)
    ls = Ws[iu2] > EPS
    if ls.any():
        Ys = recover(rs[ls], 2 * (b - ALPHA * Ws[iu2][ls]), n, iu, scale)["ridge"]
        for tname, T in (("Chat", T1), ("Sigma", T2)):
            res[f"base_shuffled_vs_{tname}"] = score(psd(Ys), T, True)
    # untrained networks: W = 0 has no live edge (nothing to read). A random untrained network with the
    # same number of live edges at random pairs and w ~ U(0, b/alpha), formula applied blindly:
    Wu = np.zeros(len(w))
    if live.any():
        pick = rng.choice(len(w), int(live.sum()), replace=False)
        Wu[pick] = rng.uniform(0, b / ALPHA, len(pick))
        Wum = np.zeros_like(net.W); Wum[iu2] = Wu; Wum = Wum + Wum.T
        un = o.Net(A=net.A, t=0.0, klo=net.klo, khi=net.khi, W=Wum)
        Hu = response_matrix(un)
        ru, _, _ = svec_rows((Hu[iu2[0]] - Hu[iu2[1]]) @ basis)
        lu = Wu > EPS
        Yu = recover(ru[lu], 2 * (b - ALPHA * Wu[lu]), n, iu, scale)["ridge"]
        for tname, T in (("Chat", T1), ("Sigma", T2)):
            res[f"base_untrained_vs_{tname}"] = score(psd(Yu), T, True)
    if keep_spectra:
        res["spectra"] = dict(Chat=np.linalg.eigvalsh(T1)[::-1], Sigma=np.linalg.eigvalsh(T2)[::-1],
                              ridge_psd=np.linalg.eigvalsh(psd(rec["ridge"]))[::-1],
                              hinge_psd=np.linalg.eigvalsh(psd(rec.get("hinge", rec["ridge"])))[::-1],
                              base_identity=np.linalg.eigvalsh(Y_id)[::-1], base_diag=np.linalg.eigvalsh(Y_dg)[::-1],
                              operator_sv=sv)
    return res


def preset_spectra():
    """Seed 0, N = 80, blocks, linear: full spectra at every b (explorer)."""
    Sigma = SIGMAS["blocks"]
    G, _ = oa.banks(Sigma, 0)
    Chat = G.T @ G / len(G)
    out = []
    for b in BGRID:
        net = oa.make_net(80, B, 0)
        r = oa.solve(net, G, b, ALPHA)
        out.append(analyse(net, r, b, Sigma, Chat, "0", 0, keep_spectra=True)["spectra"])
    return out


# ----------------------------------------------------------------------------- nonlinear: geometry only
def nl_unit(args):
    seed, b = args
    Sigma = SIGMAS["blocks"]
    G, Ge = oa.banks(Sigma, seed)
    N = 40
    net = oa.make_net(N, B, seed, **oa.RECT)
    r = oa.solve(net, G, b, ALPHA)
    D_true = np.sqrt(np.clip(o.dist2(r["U"]), 0, None))                   # training bank (what the weights encode)
    D_eval = np.sqrt(np.clip(o.dist2(o.equilibrate(net, Ge)), 0, None))   # independent evaluation bank
    iu = np.triu_indices(N, 1)
    live = net.W > EPS
    L = np.where(live, np.sqrt(np.clip(2 * (b - ALPHA * net.W), 0, None)), 0.0)
    np.fill_diagonal(L, 0)
    D_sp = shortest_path(L, method="D", directed=False, unweighted=False)
    lab = o.components(net.W)
    same = lab[:, None] == lab[None]
    edge_err = float(np.abs(L[live] - D_true[live]).max()) if live.any() else 0.0
    inner = same & ~live & ~np.eye(N, dtype=bool)
    ratio = D_sp[inner] / np.maximum(D_true[inner], 1e-300) if inner.any() else np.array([1.0])
    ratio_e = D_sp[inner] / np.maximum(D_eval[inner], 1e-300) if inner.any() else np.array([1.0])
    edge_err_eval = float((np.abs(L[live] - D_eval[live]) / np.maximum(D_eval[live], 1e-300)).max()) if live.any() else 0.0
    cross = ~same
    lb_ok = bool((D_true[cross] >= np.sqrt(2 * b) - 1e-6).all()) if cross.any() else True
    # MDS of the largest component: completed vs true
    k = np.bincount(lab).argmax()
    idx = np.nonzero(lab == k)[0]
    disp = disp_e = float("nan")
    if len(idx) >= 4:
        def cmds(D, dim=3):
            n = len(D); J = np.eye(n) - 1.0 / n
            Bm = -0.5 * J @ (D ** 2) @ J
            lam, V = np.linalg.eigh(Bm)
            lam, V = lam[::-1][:dim], V[:, ::-1][:, :dim]
            return V * np.sqrt(np.clip(lam, 0, None))
        X1, X2 = cmds(D_sp[np.ix_(idx, idx)]), cmds(D_true[np.ix_(idx, idx)])
        if np.linalg.norm(X1) > 0 and np.linalg.norm(X2) > 0:
            disp = float(procrustes(X2, X1)[2])
            X3 = cmds(D_eval[np.ix_(idx, idx)])
            disp_e = float(procrustes(X3, X1)[2])
    n_pairs = len(iu[0])
    return dict(seed=seed, b=b, residual=r["residual"], converged=r["converged"], K=int(lab.max() + 1),
                frac_exact=float(live[iu].sum() / n_pairs), frac_upper_bounded=float(inner[iu].sum() / n_pairs),
                frac_lower_only=float(cross[iu].sum() / n_pairs), edge_err=edge_err,
                sp_ratio_median=float(np.median(ratio)), sp_ratio_max=float(ratio.max()),
                lower_bound_holds=lb_ok, largest=int(len(idx)), mds_procrustes_disparity=disp,
                edge_rel_err_eval=edge_err_eval, sp_ratio_eval_median=float(np.median(ratio_e)),
                lower_bound_holds_eval_frac=float((D_eval[cross] >= np.sqrt(2 * b)).mean()) if cross.any() else 1.0,
                mds_procrustes_disparity_eval=disp_e)


def main():
    jobs = [(sg, N, s, "0") for sg in SIGMAS for N in NS for s in SEEDS]
    jobs += [("blocks", 40, s, k) for s in SEEDS for k in ("u0.1", "u1", "h0.2", "h1")]
    lin = oa.pool_map(lin_unit, jobs)
    spec = preset_spectra()
    nl = oa.pool_map(nl_unit, [(s, b) for s in SEEDS for b in (0.02, 0.05, 0.1, 0.3)])
    meta = dict(
        script="hebb/sweeps2/readout.py",
        description="Q-readout: covariance from weights. Linear: live-edge measurement operator on the contrast "
                    "space (rank, singular values), masked ridge / hinge / PSD recovery scored vs P C_hat P and "
                    "P Sigma P, three error types, baselines; quadratic leak: full-C operator; rectifying: "
                    "partial distance geometry only.",
        config=dict(B=B, sigmas=dict(blocks="sigma_blocks(12,(2,1,.5,.25),.8)", power="sigma_power(12)"),
                    b_grid=BGRID, alpha=ALPHA, Ns=NS, s=3,
                    leaks="'0' none; 'u<k>' uniform quadratic leak k_i = k; 'h<k>' k_i ~ U(0,k) per neuron (stream [seed,4])",
                    rect=oa.RECT, S_train=oa.S_TRAIN,
                    ridge="rho = 1e-10 sv_max^2 (min-norm on the identifiable subspace; unidentified part -> 0, a prior)",
                    hinge="sum over dead pairs (2b - d^T Y d)_+^2, L-BFGS from the ridge solution",
                    contrast_basis="DCT-II columns 1..11 (orthonormal, orthogonal to 1)"),
        seeds=SEEDS,
        fields={
            "lin[].runs[k]": "analysis at b_grid[k] for (sigma, N, seed, leak); leak '0' = zero potential (unknown on the contrast space, dim 66); leaks u/h: unknown = full C (dim 78)",
            "identifiable_frac_1e8": "as identifiable_frac at tolerance 1e-8",
            "ridge_psd_contrast_vs_Chat": "leak arms: the full recovery restricted to the contrast space, vs P C_hat P",
            "noise<eps>_ridge_psd_vs_Chat_cos": "Frobenius cosine vs C_hat target when every weight is multiplied by (1 + eps xi), xi ~ N(0,1), and the whole readout (H, operator, values) is recomputed from the noisy weights",
            "K, K_multi, largest, diam_max, span_over_range": "cluster structure of the readout network (training bank), see fusion.json fields",
            "base_untrained_vs_<target>": "random untrained network: same number of live edges at random pairs, w ~ U(0, b/alpha), formula applied (W = 0 itself has no live edge and gives nothing)",
            "n_live, dim": "live edges (measurements) and dimension of the unknown symmetric matrix space",
            "rank_1e8 / rank_1e4": "rank of the measurement operator at relative singular-value tolerance 1e-8 / 1e-4",
            "sv_max, sv_min_nonzero": "largest and smallest (above 1e-8 relative) singular values of the operator (units: attachment-share^2)",
            "ones_leak": "max |d^T 1| over pairs (0 when H 1 = 1)",
            "err_fixed_point": "max over live edges |2(b - alpha w) - d^T C_hat d| (D^2 units): fixed-point/solver error",
            "err_sampling": "||P C_hat P - P Sigma P||_F / ||P Sigma P||_F (sampling error)",
            "err_inverse_identifiable": "relative error of the ridge recovery projected on the identifiable subspace (singular values > 1e-4 of the largest) vs the target projected likewise",
            "identifiable_frac": "fraction of ||target||_F^2 in that subspace (tol 1e-4)",
            "<method>_vs_<target>": "cos = Frobenius cosine; rel_err = ||Y - T||/||T||; bures_rel = Bures-Wasserstein distance / sqrt(Tr T); methods ridge, ridge_psd, hinge_psd; base_identity (c I fitted to live measurements), base_diag (nonneg diagonal fitted by NNLS), base_shuffled (weights permuted over pairs, formula applied)",
            "spectra[k]": "seed 0, blocks, N=80, linear, at b_grid[k]: eigenvalues of the targets and recoveries on the contrast space, and the operator singular values",
            "nl[]": "rectifying N=40 blocks (D_true = training bank, *_eval = independent evaluation bank): frac_exact (pairs with D known exactly = live), frac_upper_bounded (same component, not an edge: shortest-path upper bound), frac_lower_only (different components: only D >= sqrt(2b)); edge_err = max |sqrt(2(b - alpha w)) - D| on edges; sp_ratio = shortest-path / true D inside components; mds_procrustes_disparity of classical-MDS (3-D) embeddings of the largest component, completed vs true (0 = identical up to similarity); eval-bank versions: edge_rel_err_eval = max relative |weight-implied D - D_eval| on edges (sampling), sp_ratio_eval_median, lower_bound_holds_eval_frac = fraction of cross-component pairs with D_eval >= sqrt(2b), mds_procrustes_disparity_eval",
        })
    oa.dump("readout", meta, dict(lin=lin, spectra=spec, nl=nl), sig=4)


if __name__ == "__main__":
    main()
