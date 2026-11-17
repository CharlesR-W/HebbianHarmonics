"""Weight-space readout of Sigma_g (SPEC §3 + §7), torch.

Observed: the weights only (plus the model parameters theta, lambda/eta, gamma when the
"parameters known" setting is scored).  From the weights compute H, then on ACTIVE edges
(w > 2*w_min) the fixed-point equations are linear in C:

    I-B: (H C)_ib       = theta + (lambda/eta) w_ib
    I-I: h_i^T C h_j    = theta/gamma + lambda/(gamma*eta) w_ij        (i != j)

Estimator `ls_readout`: masked ridge least squares over symmetric C, each edge family's
squared residual averaged over its equations (so the ~N^2/2 lateral equations do not
swamp the ~N*B feedforward ones), ridge alpha = alpha_rel * ||A|| (power iteration),
solved by conjugate gradients on the normal equations.  Families: "both", "ii", "ib".
No hinge penalty for clipped edges, no PSD projection inside the fit (PSD clipping is
applied, and reported, only for Bures-Wasserstein).

`naive_readout`: the note's C = H^+ R H^{+T} with R built from ALL lateral weights
(clipped ones too) and the unobserved diagonal filled with the row max; ridge
pseudo-inverse.  Also its feed-forward analogue C = sym(H^+ R_IB).
"""
from __future__ import annotations

import numpy as np
import torch


def _sym(X):
    return 0.5 * (X + X.T)


def equations(Wii, Wib, theta, lam=1.0, eta=1.0, gamma=1.0, wmin=1e-3):
    Mii = (Wii > 2 * wmin).to(Wii.dtype); Mii.fill_diagonal_(0.0)
    Mib = (Wib > 2 * wmin).to(Wib.dtype)
    Yii = theta / gamma + lam / (gamma * eta) * Wii
    Yib = theta + lam / eta * Wib
    return Mii, Yii, Mib, Yib


def ls_readout(H, Mii, Yii, Mib, Yib, family="both", alpha_rel=1e-4, iters=400, tol=1e-7):
    B = H.shape[1]
    nii = float(Mii.sum()); nib = float(Mib.sum())
    aii = (1.0 / nii) if (family in ("both", "ii") and nii > 0) else 0.0
    aib = (1.0 / nib) if (family in ("both", "ib") and nib > 0) else 0.0
    if aii == 0 and aib == 0:
        return torch.zeros(B, B, device=H.device, dtype=H.dtype), dict(n_ii=nii, n_ib=nib, iters=0)
    Ht = H.T

    def Aop(C):
        out = torch.zeros_like(C)
        HC = H @ C
        if aii:
            out += aii * (Ht @ (Mii * (HC @ Ht)) @ H)
        if aib:
            out += aib * _sym(Ht @ (Mib * HC))
        return out

    # spectral norm of the data operator by power iteration -> ridge scale
    X = _sym(torch.randn(B, B, device=H.device, dtype=H.dtype, generator=torch.Generator(H.device).manual_seed(0)))
    lam_max = 0.0
    for _ in range(30):
        X = Aop(X); lam_max = float(X.norm()); X = X / max(lam_max, 1e-30)
    alpha = alpha_rel * lam_max
    b = torch.zeros(B, B, device=H.device, dtype=H.dtype)
    if aii:
        b += aii * (Ht @ (Mii * Yii) @ H)
    if aib:
        b += aib * _sym(Ht @ (Mib * Yib))
    # CG on (A + alpha I) C = b
    C = torch.zeros_like(b); R = b.clone(); Pd = R.clone(); rr = float((R * R).sum()); r0 = rr
    it = 0
    for it in range(iters):
        AP = Aop(Pd) + alpha * Pd
        a = rr / float((Pd * AP).sum())
        C += a * Pd; R -= a * AP
        rr_new = float((R * R).sum())
        if rr_new < tol ** 2 * r0:
            break
        Pd = R + (rr_new / rr) * Pd; rr = rr_new
    # data misfit per family (mean squared residual on active equations)
    HC = H @ C
    info = dict(n_ii=nii, n_ib=nib, iters=it + 1, alpha=alpha, cg_rel_res=(rr_new / r0) ** 0.5)
    if nii:
        info["misfit_ii"] = float(((Mii * (HC @ Ht - Yii)) ** 2).sum() / nii) ** 0.5
    if nib:
        info["misfit_ib"] = float(((Mib * (HC - Yib)) ** 2).sum() / nib) ** 0.5
    return _sym(C), info


def ridge_pinv(H, eps_rel=1e-6):
    G = H.T @ H
    eps = eps_rel * float(torch.trace(G)) / G.shape[0]
    return torch.linalg.solve(G + eps * torch.eye(G.shape[0], device=H.device, dtype=H.dtype), H.T)


def naive_readout(H, Wii, Wib, theta, lam=1.0, eta=1.0, gamma=1.0):
    Hp = ridge_pinv(H)
    R = theta / gamma + lam / (gamma * eta) * Wii.clone()
    R.fill_diagonal_(-float("inf"))
    rowmax = R.max(1).values
    R.fill_diagonal_(0.0); R += torch.diag(rowmax)
    C_ii = _sym(Hp @ R @ Hp.T)
    C_ib = _sym(Hp @ (theta + lam / eta * Wib))
    return C_ii, C_ib


# ---------------------------------------------------------------- metrics
def _psd_clip(A):
    ev, V = torch.linalg.eigh(_sym(A))
    neg = float(ev.clamp(max=0).abs().sum() / ev.abs().sum().clamp_min(1e-300))
    return (V * ev.clamp(min=0)) @ V.T, ev, neg


def bures(A, Ssqrt, trS, scaled=True):
    """BW(A, S)^2 / tr S, A PSD; with scaled=True A is rescaled to minimise it
    (= 1 - F^2/(trA trS), F = tr (S^1/2 A S^1/2)^1/2).  All-zero A -> 1."""
    trA = float(torch.trace(A))
    if trA <= 0:
        return 1.0
    M = _sym(Ssqrt @ A @ Ssqrt)
    F = float(torch.linalg.eigvalsh(M).clamp(min=0).sqrt().sum())
    if scaled:
        return max(0.0, 1.0 - F * F / (trA * trS))
    return max(0.0, (trA + trS - 2 * F) / trS)


def subspace_overlap(A, S, k):
    """Mean cos^2 of principal angles between top-k eigenspaces (1 = identical)."""
    k = max(1, min(k, A.shape[0] - 1))
    Ua = torch.linalg.eigh(_sym(A))[1][:, -k:]
    Us = torch.linalg.eigh(_sym(S))[1][:, -k:]
    return float((Ua.T @ Us).pow(2).sum() / k)


class Scorer:
    """Scores C against a fixed Sigma in the three SPEC §7.3 settings:
    known  - parameters known: full C vs full Sigma, absolute units (rel. Frobenius error,
             unscaled BW^2/trS);
    scale  - unknown lambda/eta: full matrices, best scale (Frobenius cosine, scaled BW);
    contrast - unknown theta: P C P vs P Sigma P, P = I - 11^T/B ("contrast-subspace
             score"; also discards 1a^T + a1^T terms): cosine, scaled BW, top-k overlap.
    BW always on the PSD-clipped C; the negative spectral mass removed is reported."""

    def __init__(self, Sigma: torch.Tensor, ks=(5, 10, 20)):
        S = _sym(Sigma.double())
        B = S.shape[0]
        self.B, self.ks = B, ks
        self.P = torch.eye(B, dtype=S.dtype, device=S.device) - 1.0 / B
        self.S, self.Sp = S, _sym(self.P @ S @ self.P)
        self.sqrt_S = self._sqrtm(S); self.sqrt_Sp = self._sqrtm(self.Sp)
        self.trS, self.trSp = float(torch.trace(S)), float(torch.trace(self.Sp))
        ev, V = torch.linalg.eigh(self.Sp)
        self.ev_Sp, self.V_Sp = ev.flip(0), V.flip(1)

    @staticmethod
    def _sqrtm(A):
        ev, V = torch.linalg.eigh(_sym(A))
        return (V * ev.clamp(min=0).sqrt()) @ V.T

    def score(self, C: torch.Tensor, k_clusters: int | None = None, spectra=False) -> dict:
        C = _sym(C.double())
        S, Sp = self.S, self.Sp
        out = {}
        nS = float(S.norm())
        out["known_relfro"] = float((C - S).norm()) / nS
        Cpsd, _, neg = _psd_clip(C)
        out["known_bw"] = bures(Cpsd, self.sqrt_S, self.trS, scaled=False)
        out["scale_cos"] = float((C * S).sum()) / max(float(C.norm()) * nS, 1e-300)
        out["scale_bw"] = bures(Cpsd, self.sqrt_S, self.trS, scaled=True)
        out["neg_mass_full"] = neg
        Cp = _sym(self.P @ C @ self.P)
        cosp = float((Cp * Sp).sum()) / max(float(Cp.norm()) * float(Sp.norm()), 1e-300)
        out["contrast_cos"] = cosp
        out["contrast_relfro_scaled"] = float(np.sqrt(max(0.0, 1 - max(cosp, 0) ** 2)))
        evp, Vp = torch.linalg.eigh(Cp)
        out["neg_mass_contrast"] = float(evp.clamp(max=0).abs().sum() / evp.abs().sum().clamp_min(1e-300))
        Cpp = (Vp * evp.clamp(min=0)) @ Vp.T
        out["contrast_bw"] = bures(Cpp, self.sqrt_Sp, self.trSp, scaled=True)
        Vp = Vp.flip(1)
        kl = [(f"overlap_k{k}", k) for k in self.ks]
        if k_clusters:
            kl.append(("overlap_kclust", k_clusters)); out["k_clusters"] = int(k_clusters)
        for name, k in kl:
            k = max(1, min(k, self.B - 1))
            out[name] = float((Vp[:, :k].T @ self.V_Sp[:, :k]).pow(2).sum() / k)
        if spectra:
            out["_spectrum"] = evp.flip(0).cpu().numpy()
        return out

    def baselines(self, k_clusters: int) -> dict:
        S = self.S
        B = self.B
        ev, V = torch.linalg.eigh(S)
        k = max(1, min(k_clusters, B))
        trunc = (V[:, -k:] * ev[-k:]) @ V[:, -k:].T
        return {"identity": self.score(torch.eye(B, dtype=S.dtype, device=S.device), k_clusters),
                "diag": self.score(torch.diag(torch.diag(S)), k_clusters),
                f"rank_k_truncation": self.score(trunc, k_clusters)}


# ---------------------------------------------------------------- identifiability
def measurement_rank(cent: torch.Tensor, lab: np.ndarray, Mii: torch.Tensor, Mib: torch.Tensor,
                     max_rows: int = 6000, seed: int = 0) -> dict:
    """Rank / conditioning of the cluster-level measurement operator on Sym(B).

    One representative (the centroid part c_k) per cluster.  I-I rows: sym(c_k c_l^T) for
    cluster pairs (k <= l) joined by >= 1 active lateral edge; I-B rows: sym(c_k e_b^T) for
    (k, b) with >= 1 active feed-forward edge.  Singular values from the Gram matrix of
    the rows under the Frobenius inner product (exact formulas, no B^2 vectors)."""
    dev, dt = cent.device, torch.float64
    c = cent.to(dt)
    K, B = c.shape
    labt = torch.as_tensor(lab, device=dev)
    onehot = torch.zeros(len(lab), K, device=dev, dtype=dt); onehot[torch.arange(len(lab)), labt] = 1
    Pkk = (onehot.T @ Mii.to(dt) @ onehot) > 0
    Pkb = (onehot.T @ Mib.to(dt)) > 0
    kk = torch.nonzero(torch.triu(Pkk.to(torch.int8)), as_tuple=False)
    kb = torch.nonzero(Pkb, as_tuple=False)
    g = torch.Generator(device="cpu").manual_seed(seed)
    sub = {}
    if len(kk) > max_rows:
        kk = kk[torch.randperm(len(kk), generator=g)[:max_rows].to(dev)]; sub["ii"] = max_rows
    if len(kb) > max_rows:
        kb = kb[torch.randperm(len(kb), generator=g)[:max_rows].to(dev)]; sub["ib"] = max_rows
    Q = c @ c.T
    k1, l1 = kk[:, 0], kk[:, 1]
    k2, b2 = kb[:, 0], kb[:, 1]
    Gii = 0.5 * (Q[k1][:, k1] * Q[l1][:, l1] + Q[k1][:, l1] * Q[l1][:, k1])
    Gib = 0.5 * (Q[k2][:, k2] * (b2[:, None] == b2[None, :]).to(dt) + c[k2][:, b2] * c[k2][:, b2].T)
    Gx = 0.5 * (Q[k1][:, k2] * c[l1][:, b2] + c[k1][:, b2] * Q[l1][:, k2])

    def spec(G):
        if G.shape[0] == 0:
            return dict(rows=0, rank=0, rank_sv1e3=0, cond=None)
        ev = torch.linalg.eigvalsh(_sym(G)).clamp(min=0).flip(0)
        sv = ev.sqrt(); smax = float(sv[0])
        nz = sv[sv > 1e-8 * smax]
        return dict(rows=int(G.shape[0]), rank=int(len(nz)), rank_sv1e3=int((sv > 1e-3 * smax).sum()),
                    cond=float(smax / float(nz[-1])) if len(nz) else None)

    Gb = torch.cat([torch.cat([Gii, Gx], 1), torch.cat([Gx.T, Gib], 1)], 0)
    return dict(dim_sym=B * (B + 1) // 2, K=K, ii=spec(Gii), ib=spec(Gib), both=spec(Gb), subsampled=sub)
