"""Cluster definitions and structural diagnostics of H and W (torch, CPU or GPU).

Headline cluster definition (order-free, SPEC §7.5): single-linkage components of the
graph "L1(h_i, h_j) < tol" with tol = 0.1 (embeddings are probability vectors, so L1 is
twice the total-variation distance; 0.1 means the two neurons' harmonic measures differ
by 5% of mass).  Also reported: tol 0.05 / 0.2, hebb.core.clusters leader clustering
(order dependent, the rest of the site's quick look), connected components of the active
I-I graph, and the count of "major" clusters (>= 1% of N neurons).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def l1_dist(H: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    out = torch.empty(H.shape[0], H.shape[0], device=H.device, dtype=H.dtype)
    for i in range(0, H.shape[0], chunk):
        out[i:i + chunk] = torch.cdist(H[i:i + chunk], H, p=1)
    return out


def components(adj: np.ndarray) -> np.ndarray:
    _, lab = connected_components(csr_matrix(adj), directed=False)
    # relabel by decreasing size (cluster 0 = largest)
    sizes = np.bincount(lab)
    order = np.argsort(-sizes, kind="stable")
    remap = np.empty_like(order); remap[order] = np.arange(len(order))
    return remap[lab]


def cluster_labels(H: torch.Tensor, tol: float = 0.1, D: torch.Tensor | None = None) -> np.ndarray:
    D = l1_dist(H) if D is None else D
    return components((D < tol).cpu().numpy())


def leader_count(H: np.ndarray, tol: float = 0.1) -> int:
    from hebb import core
    return len(core.clusters(H, tol)[0])


def cluster_summary(H: torch.Tensor, Wii: torch.Tensor, wmin: float, Sigma: torch.Tensor,
                    with_leader: bool = True) -> dict:
    N = H.shape[0]
    D = l1_dist(H)
    lab = cluster_labels(H, 0.1, D)
    sizes = np.bincount(lab)
    out = dict(n_clusters=int(len(sizes)),
               n_clusters_tol05=int(components((D < 0.05).cpu().numpy()).max() + 1),
               n_clusters_tol20=int(components((D < 0.2).cpu().numpy()).max() + 1),
               n_major=int((sizes >= max(2, 0.01 * N)).sum()),
               n_singletons=int((sizes == 1).sum()),
               n_active_components=int(components((Wii > 2 * wmin).cpu().numpy()).max() + 1))
    if with_leader:
        out["n_leader"] = leader_count(H.double().cpu().numpy(), 0.1)
    # within-cluster spread (mean L1 to the cluster centroid), a diversity measure
    labt = torch.as_tensor(lab, device=H.device)
    K = len(sizes)
    cent = torch.zeros(K, H.shape[1], device=H.device, dtype=H.dtype).index_add_(0, labt, H)
    cent /= torch.as_tensor(sizes, device=H.device, dtype=H.dtype)[:, None]
    out["within_l1"] = float((H - cent[labt]).abs().sum(1).mean())
    energy = ((cent @ Sigma) * cent).sum(1)
    return out, lab, cent, energy


def eff_rank(H: torch.Tensor) -> float:
    """Roy-Vetterli effective rank exp(entropy of normalised singular values)."""
    s = torch.linalg.svdvals(H.double())
    p = s / s.sum(); p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


HIST_EDGES = np.concatenate([[0.0], np.logspace(-5, 1, 49)])  # bin 0 = [0, 1e-5)


def weight_hist(W: torch.Tensor, offdiag: bool) -> np.ndarray:
    w = W.flatten() if not offdiag else W[~torch.eye(W.shape[0], dtype=torch.bool, device=W.device)]
    edges = torch.as_tensor(HIST_EDGES, device=W.device, dtype=W.dtype)
    idx = torch.bucketize(w.clamp(max=9.99), edges, right=True) - 1
    return torch.bincount(idx.clamp(0, len(HIST_EDGES) - 2), minlength=len(HIST_EDGES) - 1).cpu().numpy()


def floor_frac(Wii: torch.Tensor, Wib: torch.Tensor, wmin: float) -> tuple[float, float]:
    N = Wii.shape[0]
    fii = (float((Wii <= wmin * (1 + 1e-4)).sum()) - N) / (N * (N - 1))
    fib = float((Wib <= wmin * (1 + 1e-4)).float().mean())
    return fii, fib
