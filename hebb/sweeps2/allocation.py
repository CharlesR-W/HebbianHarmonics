"""Q-allocation (T2, SPEC §3).

(a) Base model (fixed attachments): neurons per cluster is set by the attachments and the fusion
    scale. Planted blocks: vary one block's variance with attachments fixed, and vary its neuron
    count with variance fixed. Not an n(lambda) law.
(b) Plastic attachments under the SAME energy rule (gradient descent on J, attachment ridge
    alphaA/2 a^2, fixed degree: rows of A projected on {a >= 0, sum a = deg}) migrate to quiet inputs.
(c) ADDITIONAL principle, not derived from the conductance energy (SPEC §3): whitened linear response
    vectors x_i in R^B, objective 1/2 ||Sigma - X^T X||_F^2, hard constraint |x_i|^2 = nu.
    Law (hard constraint): fixed-trace water-filling q_a = (lambda_a - mu)_+, sum q = N nu.
    (c1) mass law; (c2) specialisation (purity, counts n_a) under added ingredients: none, the
    fusion term -(1/2 alpha) sum (b - |x_i-x_j|^2/2)_+^2 (the lateral energy minimised over w),
    non-negativity in the eigenbasis, non-negativity in a rotated basis, non-negativity + fusion;
    (c3) how far each ingredient moves q off the law.

Run: uv run python -m hebb.sweeps2.allocation   -> site/data/sweeps2/allocation.json
"""
import hebb.sweeps2  # noqa: F401

import numpy as np

from hebb import onsite as o
from hebb import onsite_analysis as oa

B, NBLK = 12, 4
M = B // NBLK


# ----------------------------------------------------------------------------- (a) base model
def sigma_var(var, rho=0.8):
    return oa.sigma_blocks(B, tuple(var), rho)


def block_of_clusters(lab, home):
    """For each component: its block if all members share one, else -1 (mixed)."""
    out = []
    for k in range(lab.max() + 1):
        hs = np.unique(home[lab == k])
        out.append(int(hs[0]) if len(hs) == 1 else -1)
    return np.array(out)


def a_unit(args):
    kind, val, b, seed = args
    var = [1.0] * NBLK
    counts = [12] * NBLK
    if kind == "variance":
        var[0] = val
    else:
        counts[0] = int(val)
    Sigma = sigma_var(var)
    A, home = oa.block_attachments(sum(counts), B, NBLK, 3, seed, counts=counts)
    G, Ge = oa.banks(Sigma, seed)
    net = oa.make_net(len(A), B, seed, A=A)
    r = oa.solve(net, G, b, 1.0)
    lab = o.components(net.W)
    blk = block_of_clusters(lab, home)
    st = o.cluster_stats(o.equilibrate(net, Ge), lab)
    sizes = np.bincount(lab)
    c0 = np.nonzero(blk == 0)[0]
    return dict(kind=kind, val=val, b=b, seed=seed, n_block0=int((home == 0).sum()),
                K_block0=int(len(c0)), mean_size_block0=float(sizes[c0].mean()),
                largest_block0=int(sizes[c0].max()), K_other_mean=float(np.mean([(blk == k).sum() for k in range(1, NBLK)])),
                mixed_clusters=int((blk == -1).sum()), residual=r["residual"], converged=r["converged"],
                diam_max_block0=float(max(st[k]["diam_max"] for k in c0)),
                diam_mean_block0=float(np.mean([st[k]["diam_mean"] for k in c0])))


def a_random_unit(args):
    """Random attachments on the 2/1/.5/.25 blocks: per component size vs prototype variance."""
    b, seed = args
    Sigma = oa.sigma_blocks(B)
    G, Ge = oa.banks(Sigma, seed)
    net = oa.make_net(48, B, seed)
    r = oa.solve(net, G, b, 1.0)
    lab = o.components(net.W)
    Ue = o.equilibrate(net, Ge)
    st = o.cluster_stats(Ue, lab)
    pts = [[c["size"], float(c["proto"].var()), c["diam_max"], c["diam_mean"]] for c in st]
    return dict(b=b, seed=seed, pts=pts, residual=r["residual"], converged=r["converged"])


# ----------------------------------------------------------------------------- (b) plastic attachments
def proj_simplex(Vm, z):
    n = Vm.shape[1]
    Us = -np.sort(-Vm, axis=1)
    css = Us.cumsum(1) - z
    k = np.arange(1, n + 1)
    rho = (Us - css / k > 0).sum(1)
    tau = css[np.arange(len(Vm)), rho - 1] / rho
    return np.maximum(Vm - tau[:, None], 0)


def b_unit(args):
    """Joint projected gradient flow on J over W (lateral, optional) and A (simplex rows)."""
    tag, alphaA, lateral, b, pot, seed = args
    Sigma = oa.sigma_blocks(B)
    G, _ = oa.banks(Sigma, seed)
    N, deg, alpha, dt = 40, 1.0, 1.0, 0.2
    A = oa.rng_for(seed, 0).dirichlet(np.ones(B), size=N) * deg
    net = oa.make_net(N, B, seed, A=A, **pot)
    T = 6000
    res = np.inf
    for step in range(T):
        U = o.equilibrate(net, G)
        Wn = net.W
        if lateral:
            Wn = np.maximum(net.W + dt * o.drift(net, G, b, alpha, U)[0], 0)
        D2B = (U ** 2).mean(0)[:, None] + (G ** 2).mean(0)[None] - 2 * U.T @ G / len(G)
        An = proj_simplex(net.A + dt * (-alphaA * net.A - 0.5 * D2B), deg)
        res = max(np.abs(Wn - net.W).max(), np.abs(An - net.A).max()) / dt
        net.W, net.A = Wn, An
        if res < 1e-8:
            break
    blk = np.arange(B) // M
    mass = np.stack([net.A[:, blk == k].sum(1) for k in range(NBLK)], 1) / net.A.sum(1, keepdims=True)
    home = mass.argmax(1)
    return dict(tag=tag, alphaA=alphaA, lateral=lateral, b=b, seed=seed, alloc=np.bincount(home, minlength=NBLK),
                alloc_majority=np.bincount(home[mass.max(1) > 0.5], minlength=NBLK),
                mass_per_block=mass.mean(0), purity=float(mass.max(1).mean()), support=float((net.A > 1e-6).sum(1).mean()),
                K=int(o.components(net.W).max() + 1), step_residual=float(res), steps=step + 1,
                converged=bool(res < oa.RES_TOL))


# ----------------------------------------------------------------------------- (c) similarity matching
def sm_solve(Sigma, N, nu, seed, b=0.0, alpha=1.0, nonneg_basis=None, T=20000, lr=0.02):
    """Projected gradient on 1/2||Sigma - X^T X||^2 - (1/2alpha) sum_{i<j} (b - |x_i-x_j|^2/2)_+^2,
    rows on the sphere |x|^2 = nu (optionally also R^T x >= 0 for nonneg_basis = R).
    Returns X and the gradient-mapping norm at the end (stationarity measure)."""
    rng = oa.rng_for(seed, 5)
    Bn = len(Sigma)
    X = rng.standard_normal((N, Bn))
    R = nonneg_basis

    def project(X):
        if R is not None:
            Y = np.maximum(X @ R, 0)
            dead = Y.max(1) <= 0
            if dead.any():
                Y[dead, np.argmax(X[dead] @ R, axis=1)] = 1.0
            X = Y @ R.T
        return X * np.sqrt(nu) / np.linalg.norm(X, axis=1, keepdims=True)

    def grad(X):
        g = 2 * X @ (X.T @ X - Sigma)
        if b > 0:
            D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
            w = np.maximum(b - D2 / 2, 0) / alpha
            np.fill_diagonal(w, 0)
            g += w.sum(1)[:, None] * X - w @ X
        return g

    X = project(X)
    for _ in range(T):
        X = project(X - lr * grad(X))
    gm = np.abs(X - project(X - lr * grad(X))).max() / lr
    return X, float(gm)


def sm_objective(Sigma, X, b, alpha):
    f = 0.5 * ((Sigma - X.T @ X) ** 2).sum()
    if b > 0:
        D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
        iu = np.triu_indices(len(X), 1)
        f -= (np.maximum(b - D2[iu] / 2, 0) ** 2).sum() / (2 * alpha)
    return float(f)


def integer_fill(lam, N, nu):
    """Best counts for pure neurons: argmin 1/2 sum (lam_a - n_a nu)^2, sum n = N (greedy is exact:
    separable convex)."""
    n = np.zeros(len(lam), int)
    for _ in range(N):
        gain = (lam - n * nu) ** 2 - (lam - (n + 1) * nu) ** 2
        n[np.argmax(gain)] += 1
    return n


SPECTRA = dict(
    lam8=np.array([3.0, 2.0, 1.5, 1.0, 0.6, 0.3, 0.1, 0.05]),
    pow16=4.0 / np.arange(1, 17),
)
NS = dict(lam8=60, pow16=80)


def ingredients(Bn, seed):
    Rr = np.linalg.qr(oa.rng_for(SIGMA_SEED_ROT, 0).standard_normal((Bn, Bn)))[0]
    return {
        "none": dict(),
        "fusion_b0.01": dict(b=0.01),
        "fusion_b0.03": dict(b=0.03),
        "fusion_b0.1": dict(b=0.1),
        "nonneg_eig": dict(nonneg_basis=np.eye(Bn)),
        "nonneg_rotated": dict(nonneg_basis=Rr),
        "nonneg_eig+fusion_b0.03": dict(nonneg_basis=np.eye(Bn), b=0.03),
    }


SIGMA_SEED_ROT = 777


def c_unit(args):
    spec, nu, ing, seed = args
    lam = SPECTRA[spec]
    N = NS[spec]
    Sigma = np.diag(lam)                          # eigenbasis = coordinate basis
    kw = ingredients(len(lam), seed)[ing]
    X, gm = sm_solve(Sigma, N, nu, seed, **kw)
    Mx = X.T @ X
    q = np.diag(Mx)
    q_pred, mu = oa.water_fill(lam, N * nu)
    purity = (X ** 2).max(1) / nu
    home = (X ** 2).argmax(1)
    counts = np.bincount(home, minlength=len(lam))
    counts_pure = np.bincount(home[purity > 0.9], minlength=len(lam))
    n_int = integer_fill(lam, N, nu)
    f = sm_objective(Sigma, X, kw.get("b", 0.0), 1.0)
    f_main = 0.5 * ((Sigma - Mx) ** 2).sum()
    f_star = 0.5 * ((lam - q_pred) ** 2).sum()
    return dict(spec=spec, nu=nu, ing=ing, seed=seed, N=N, mu=mu, q=q, q_pred=q_pred,
                q_dev=float(np.abs(q - q_pred).sum() / (N * nu)),
                offdiag=float(np.abs(Mx - np.diag(q)).max()),
                purity_mean=float(purity.mean()), pure_frac=float((purity > 0.9).mean()),
                counts=counts, counts_pure=counts_pure, n_from_law=q_pred / nu, n_integer_fill=n_int,
                count_dev=float(np.abs(counts - q_pred / nu).sum() / N),
                count_dev_integer=float(np.abs(counts - n_int).sum() / N),
                obj=f, obj_main=f_main, obj_main_excess=float(f_main - f_star), grad_map=gm)


def main():
    # (a)
    ajobs = [("variance", v, b, s) for v in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0) for b in (0.02, 0.05) for s in range(5)]
    ajobs += [("count", c, b, s) for c in (4, 8, 12, 16, 24) for b in (0.02, 0.05) for s in range(5)]
    a_res = oa.pool_map(a_unit, ajobs)
    print("a done")
    ar = oa.pool_map(a_random_unit, [(b, s) for b in (0.02, 0.05, 0.1) for s in range(5)])
    # (b)
    bjobs = []
    for s in range(5):
        for aA in (0.1, 1.0, 10.0):
            bjobs.append((f"noLat aA={aA}", aA, False, 0.0, oa.LINEAR, s))
        for b in (0.02, 0.1, 0.3):
            bjobs.append((f"lat b={b} aA=1", 1.0, True, b, oa.LINEAR, s))
        for b in (0.02, 0.1):
            bjobs.append((f"rect lat b={b} aA=1", 1.0, True, b, oa.RECT, s))
    bjobs.sort(key=lambda j: j[4] is not oa.RECT)
    b_res = oa.pool_map(b_unit, bjobs)
    print("b done")
    # (c)
    cjobs = [("lam8", nu, ing, s) for nu in (0.05, 0.1, 0.2) for ing in ingredients(8, 0) for s in range(4)]
    cjobs += [("pow16", 0.1, ing, s) for ing in ingredients(16, 0) for s in range(4)]
    c_res = oa.pool_map(c_unit, cjobs)
    print("c done")
    meta = dict(
        script="hebb/sweeps2/allocation.py",
        description="Q-allocation: (a) base-model allocation set by attachments and fusion scale; (b) plastic "
                    "attachments under the energy rule; (c1-c3) the added similarity-matching principle: mass law, "
                    "specialisation under ingredients, deviation from the law.",
        config=dict(
            a=dict(B=B, blocks=NBLK, rho=0.8, attach="block: 3 inputs inside one block, Dirichlet shares, deg 1",
                   variance_sweep="block 0 variance in (.25..8), others 1, 12 neurons per block",
                   count_sweep="block 0 neuron count in (4..24), all variances 1", alpha=1.0, N_random=48,
                   random="random s=3 attachments on sigma_blocks(12, (2,1,.5,.25), .8)"),
            b=dict(N=40, B=B, sigma="sigma_blocks(12,(2,1,.5,.25),.8)", init="Dirichlet(1) dense rows, deg 1",
                   rule="dA = -alphaA A - 1/2 E[(u_i-g_b)^2], rows projected on simplex sum=1; dW lateral rule alpha=1",
                   dt=0.2, max_steps=6000),
            c=dict(spectra=SPECTRA, N=NS, nus=[0.05, 0.1, 0.2], solver="projected gradient lr .02, 20000 its, random init",
                   law="hard constraint |x_i|^2 = nu: q_a = (lambda_a - mu)_+ (this is what is plotted)",
                   fusion="adds -(1/2 alpha) sum_{i<j} (b - |x_i-x_j|^2/2)_+^2, alpha = 1",
                   nonneg_eig="x >= 0 in the eigenbasis of Sigma", nonneg_rotated="R^T x >= 0, R random orthogonal (seed %d)" % SIGMA_SEED_ROT)),
        seeds=dict(a=list(range(5)), b=list(range(5)), c="random init seeds 0-3"),
        fields={
            "a.planted[]": "per run: n_block0 (neurons attached to block 0), K_block0 (components made only of block-0 neurons), mean/largest size, K_other_mean, mixed_clusters (components spanning blocks)",
            "a.random[].pts": "[component size, prototype variance (eval bank, response variance units), diam_max, diam_mean (eval bank, response units)] per component",
            "a.planted[].diam_*_block0": "max / mean over block-0 components of their max / mean member-to-prototype RMS distance (eval bank)",
            "b[].alloc": "neurons whose LARGEST block share is block k (argmax; block variances 2, 1, .5, .25)",
            "b[].alloc_majority": "neurons with more than half their attachment mass on block k",
            "b[].mass_per_block": "mean attachment-mass share per block",
            "b[].purity": "mean largest block share of a neuron's attachment mass",
            "b[].support": "mean number of inputs with a > 1e-6",
            "b[].step_residual": "max |dW|,|dA| / dt at the last step (converged iff < res_tol)",
            "c[].q": "mode masses q_a = (X^T X)_aa in the eigenbasis",
            "c[].q_pred": "water-filling (lambda - mu)_+ with sum = N nu",
            "c[].q_dev": "sum |q - q_pred| / (N nu)",
            "c[].offdiag": "max off-diagonal |(X^T X)_ab|",
            "c[].purity_mean, pure_frac": "purity = max_a x_ia^2 / nu; fraction with purity > 0.9",
            "c[].counts, counts_pure": "neurons by home mode (argmax), all / only purity > 0.9",
            "c[].n_from_law": "q_pred / nu (the neuron-count law if neurons were pure)",
            "c[].n_integer_fill": "best integer counts for pure neurons (argmin sum (lambda - n nu)^2)",
            "c[].count_dev": "sum |counts - q_pred/nu| / N; count_dev_integer: vs n_integer_fill",
            "c[].obj_main_excess": "1/2||Sigma - X^T X||^2 minus its global minimum under the constraint (>= 0)",
            "c[].grad_map": "max |X - P(X - lr grad)| / lr at the end (stationarity)",
        })
    oa.dump("allocation", meta, dict(a=dict(planted=a_res, random=ar), b=b_res, c=c_res), sig=4)


if __name__ == "__main__":
    main()
