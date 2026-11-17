"""Analyses for the rectified-output experiments: higher-order gap, probes, cluster energies."""
from __future__ import annotations

import numpy as np
import torch

from hebb.nonlin import dual
from hebb.nonlin.model import Data, thresholds
from hebb.nonlin.readout import predict, _Tab


def rate_moments(H, D: Data, tau, G=None, chunk=20000):
    """Empirical (over the 60k training pixels, or G) rate covariances, unit gain:
    returns (Cii (N,N), Cib (N,B), Cbb (B,B)) float64."""
    G = D.Gtr if G is None else G
    bi, bb = thresholds(H, D, tau)
    n = G.shape[0]; N, B = H.shape
    acc = None
    for i in range(0, n, chunk):
        g = G[i:i + chunk]
        R = torch.relu(g @ H.T - bi).double(); S = torch.relu(g - bb).double()
        part = [R.sum(0), S.sum(0), R.T @ R, R.T @ S, S.T @ S]
        acc = part if acc is None else [a + p for a, p in zip(acc, part)]
    m1, m2 = acc[0] / n, acc[1] / n
    return (acc[2] / n - torch.outer(m1, m1), acc[3] / n - torch.outer(m1, m2), acc[4] / n - torch.outer(m2, m2))


def _corr(C, v1, v2):
    den = torch.sqrt(torch.outer(v1.clamp_min(0), v2.clamp_min(0)))
    return torch.where(den > 0, C / den.clamp_min(1e-300), torch.zeros_like(C))


def gap_analysis(H, D: Data, tau, Mii=None, Mib=None, n_pairs=3000, seed=0):
    """Compare the empirical rate covariance with the Gaussian dual-activation prediction from
    the (exact) voltage covariance.  The gap is the part of what the rule sees that depends on
    moments beyond the second."""
    H = H.double(); S = D.Sigma_t.double()
    tab = _Tab(tau, device=H.device)
    Eii, Eib, Ebb = rate_moments(H.to(D.dtype), D, tau)
    Pii, Pib = predict(H, S, tab)
    sd_b = D.sd_t.double()
    Pbb = torch.outer(sd_b, sd_b) * _corr_k(S, sd_b, tab)
    HS = H @ S; Cuu = HS @ H.T
    var_u = Cuu.diagonal(); var_b = S.diagonal()
    rho_ii = _corr(Cuu, var_u, var_u); rho_ib = _corr(HS, var_u, var_b); rho_bb = _corr(S, var_b, var_b)
    ev_i, ev_b = Eii.diagonal(), Ebb.diagonal()
    out = {}
    N, B = H.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    iu = torch.triu_indices(N, N, 1)
    ib_live = (var_b > 1e-8)
    fams = {
        "ii": (Eii[iu[0], iu[1]], Pii[iu[0], iu[1]], rho_ii[iu[0], iu[1]], _corr(Eii, ev_i, ev_i)[iu[0], iu[1]],
               None if Mii is None else Mii.bool()[iu[0], iu[1]].cpu()),
        "ib": (Eib[:, ib_live].flatten(), Pib[:, ib_live].flatten(), rho_ib[:, ib_live].flatten(),
               _corr(Eib, ev_i, ev_b)[:, ib_live].flatten(),
               None if Mib is None else Mib.bool()[:, ib_live].flatten().cpu()),
    }
    bl = torch.nonzero(ib_live).flatten()
    bu = torch.triu_indices(len(bl), len(bl), 1)
    bi, bj = bl[bu[0]], bl[bu[1]]
    fams["bb"] = (Ebb[bi, bj], Pbb[bi, bj], rho_bb[bi, bj], _corr(Ebb, ev_b, ev_b)[bi, bj], None)
    for fam, (E, P, rho, rcorr, act) in fams.items():
        E, P, rho, rcorr = E.cpu(), P.cpu(), rho.cpu(), rcorr.cpu()
        d = dict(n=int(E.numel()))
        d["rel_fro_gap"] = float((E - P).norm() / E.norm().clamp_min(1e-300))
        d["corr_emp_pred"] = float(np.corrcoef(E.numpy(), P.numpy())[0, 1]) if E.numel() > 2 else None
        d["mean_emp"], d["mean_pred"] = float(E.mean()), float(P.mean())
        d["emp_quantiles"] = [float(q) for q in torch.quantile(E[:2_000_000].float(), torch.tensor([.01, .1, .25, .5, .75, .9, .99]))]
        kt = torch.as_tensor(dual.k(rho.numpy(), tau) / dual.k(1.0, tau))
        d["rate_corr_rel_gap"] = float((rcorr - kt).norm() / rcorr.norm().clamp_min(1e-300))
        d["rate_corr_mean_abs_gap"] = float((rcorr - kt).abs().mean())
        if act is not None and bool(act.any()):
            d["active_n"] = int(act.sum())
            d["active_rel_fro_gap"] = float((E[act] - P[act]).norm() / E[act].norm().clamp_min(1e-300))
            d["active_mean_ratio_emp_over_pred"] = float((E[act] / P[act].clamp_min(1e-12)).median())
        sel = torch.randperm(E.numel(), generator=g)[:n_pairs]
        d["sample"] = dict(rho=rho[sel].float().numpy().round(4).tolist(), rate_corr=rcorr[sel].float().numpy().round(4).tolist(),
                           emp=E[sel].float().numpy().tolist(), pred=P[sel].float().numpy().tolist())
        out[fam] = d
    # variance mismatch: Var_emp(r_i) / (k(1) var_u)
    vr = (ev_i / (dual.k(1.0, tau) * var_u.clamp_min(1e-300))).cpu()
    out["var_ratio_interior_quantiles"] = [float(q) for q in torch.quantile(vr.float(), torch.tensor([.1, .5, .9]))]
    vb = (ev_b[ib_live] / (dual.k(1.0, tau) * var_b[ib_live])).cpu()
    out["var_ratio_pixel_quantiles"] = [float(q) for q in torch.quantile(vb.float(), torch.tensor([.1, .5, .9]))]
    return out


def _corr_k(S, sd, tab):
    from hebb.nonlin.readout import _KFn
    den = torch.outer(sd, sd)
    rho = torch.where(den > 0, S / den.clamp_min(1e-300), torch.zeros_like(S))
    return _KFn.apply(rho, tab)


# ---------------------------------------------------------------- probes
def logistic_probe(Ftr, ytr, Fte, yte, wd=1e-4, iters=150):
    """Multinomial logistic regression on z-scored features, full-batch L-BFGS. Test accuracy."""
    dev = Ftr.device
    mu = Ftr.mean(0, keepdim=True); sd = Ftr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Ftr - mu) / sd; Xte = (Fte - mu) / sd
    ytr_t = torch.as_tensor(ytr, device=dev); yte_t = torch.as_tensor(yte, device=dev)
    W = torch.zeros(Xtr.shape[1], 10, device=dev, dtype=Xtr.dtype, requires_grad=True)
    b = torch.zeros(10, device=dev, dtype=Xtr.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr_t) + wd * (W * W).sum()
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        acc_te = float(((Xte @ W + b).argmax(1) == yte_t).float().mean())
        acc_tr = float(((Xtr @ W + b).argmax(1) == ytr_t).float().mean())
    return dict(test=acc_te, train=acc_tr, d=int(Xtr.shape[1]))


def probe_suite(H, H0, D: Data, B_sub, seed=0, taus=(0.0, 0.5, 1.0), wd=1e-4):
    """Probe accuracies: pixels; voltages and rates (several tau) of the trained net (H) and
    the untrained init (H0), all N neurons and a random subset of B_sub neurons; random ReLU
    features (Gaussian projections of centred pixels, same thresholds) as a further control."""
    out = {}
    Gtr, Gte = D.Gtr, D.Gte
    out["pixels"] = logistic_probe(Gtr, D.ytr, Gte, D.yte, wd)
    N, B = H.shape
    rng = np.random.default_rng(seed + 77)
    sub = torch.as_tensor(np.sort(rng.choice(N, min(B_sub, N), replace=False)), device=H.device)
    gen = torch.Generator(device="cpu").manual_seed(seed + 5)
    V = torch.randn(B, N, generator=gen).to(H.device, H.dtype) / B ** 0.5
    nets = {"trained": H, "init": H0}
    for nm, Hn in nets.items():
        Utr, Ute = Gtr @ Hn.T, Gte @ Hn.T
        out[f"{nm}_volt_all"] = logistic_probe(Utr, D.ytr, Ute, D.yte, wd)
        out[f"{nm}_volt_sub"] = logistic_probe(Utr[:, sub], D.ytr, Ute[:, sub], D.yte, wd)
        for tau in taus:
            bi, _ = thresholds(Hn, D, tau)
            Rtr, Rte = torch.relu(Utr - bi), torch.relu(Ute - bi)
            out[f"{nm}_rate_t{tau:g}_all"] = logistic_probe(Rtr, D.ytr, Rte, D.yte, wd)
            out[f"{nm}_rate_t{tau:g}_sub"] = logistic_probe(Rtr[:, sub], D.ytr, Rte[:, sub], D.yte, wd)
            out[f"{nm}_rate_t{tau:g}_all"]["frac_active"] = float((Rtr > 0).float().mean())
        del Utr, Ute
    Ptr, Pte = (Gtr - D.mu_t) @ V, (Gte - D.mu_t) @ V
    sdp = Ptr.std(0)
    for tau in taus:
        out[f"randrelu_t{tau:g}_all"] = logistic_probe(torch.relu(Ptr - tau * sdp), D.ytr, torch.relu(Pte - tau * sdp), D.yte, wd)
        out[f"randrelu_t{tau:g}_sub"] = logistic_probe(torch.relu(Ptr[:, sub] - tau * sdp[sub]), D.ytr,
                                                      torch.relu(Pte[:, sub] - tau * sdp[sub]), D.yte, wd)
    return out


def cluster_rate_energy(cent, D: Data, tau, chunk=20000):
    """Per cluster part c_k: voltage variance c^T Sigma c and (unit-gain) rate variance."""
    cent = cent.to(D.dtype)
    vv = ((cent @ D.Sigma_t) * cent).sum(1)
    bi, _ = thresholds(cent, D, tau)
    n = D.Gtr.shape[0]; s1 = 0; s2 = 0
    for i in range(0, n, chunk):
        R = torch.relu(D.Gtr[i:i + chunk] @ cent.T - bi).double()
        s1 = s1 + R.sum(0); s2 = s2 + (R * R).sum(0)
    rv = s2 / n - (s1 / n) ** 2
    return vv.double().cpu().numpy(), rv.cpu().numpy()
