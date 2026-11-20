"""Spontaneous activity of a network trained to its fixed point by the covariance rule (hebb/core.py).

Model after training: inputs silent (g = 0), independent white noise of temperature T in every interior
neuron,  tau du = -K u dt + sqrt(2 T tau) dW,  K = diag(deg) - W_II,  deg_i = sum_j W_II,ij + sum_b W_IB,ib.
Stationary covariance (Lyapunov  K C + C K = 2 T I):   C_sp = T K^{-1}.
Evoked covariance (inputs g ~ Sigma_g, no neuron noise):  C_ev = H Sigma_g H^T,  H = K^{-1} W_IB.

Fixed point of the rule (lam w = eta (gamma_ij Cov - theta), floored at wmin):
    W_II,ij = max(wmin, (eta/lam) (gamma C_ev,ij - theta))   =>  -K_ij = thresholded evoked covariance (i != j)
(off-diagonal entries only; the diagonal of K also holds the input conductances).
A cluster c of n clones (mixture h, total input conductance a > 0, pairwise lateral weight w), isolated (the floor
links to the rest of the network neglected):
    K_c = (a + n w) I - w 11^T,   C_sp,c = T/(a + n w) [ I + (w/a) 11^T ]
    correlation r = w/(a + w);  neuron variance T (a + w)/(a (a + n w));  cluster-mean variance T/(n a).
With q = (Sigma h - theta)_+ on the support S:  h = q/1^T q, so (Sigma_SS - theta 11^T) h = mu h with
mu = 1^T q = a lam/eta (a positive eigenvector of the thresholded piece; its Perron vector when the piece has no
negative entry), evoked variance v = h^T Sigma h = mu/m + theta,
m = 1/|h|^2 (participation ratio of the mixture). Hence w/a = (gamma v - theta)/mu and, for gamma = 1, r = 1/(1 + m).

Everything is in units T = 1. Run: uv run python -m hebb.sweeps2.spontaneous -> site/data/sweeps2/spontaneous.json
"""
import hebb.sweeps2  # noqa: F401

import numpy as np

from hebb import core as c
from hebb import onsite_analysis as oa

RES_TOL = 1e-10
TMAX = 400000
THETAS = (0.1, 0.2, 0.3, 0.45, 0.6)
GAMMAS = (0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
SEEDS = (0, 1, 2)
N_SWEEP = 36
WMIN_SHOW = 1e-3      # the site default floor (stored networks)
WMIN_SWEEP = 1e-4     # the closed forms neglect the floor; the sweep is repeated at WMIN_SHOW for comparison

FAMILIES = {
    "blocks": lambda: c.cov_blocks((4, 4, 4), 0.5, 0.1)[0],
    "blocks_uneven": lambda: c.cov_blocks((2, 4, 6), 0.6, 0.1, variances=np.array([2.0, 1.0, 0.6]))[0],
    "blocks_tight": lambda: c.cov_blocks((3, 3, 3, 3), 0.8, 0.05)[0],
    "hier": lambda: c.cov_hierarchical((2, 2, 3), (0.1, 0.35, 0.7))[0],   # three levels: 2 x 2 x 3 inputs
    "power0": lambda: c.cov_power_law(12, 1.0, seed=0, nonneg=True),
    "power1": lambda: c.cov_power_law(12, 1.0, seed=1, nonneg=True),
    "power2": lambda: c.cov_power_law(12, 0.5, seed=2, nonneg=True),
    "wishart0": lambda: c.cov_wishart(12, seed=0),
    "wishart1": lambda: c.cov_wishart(12, dof=24, seed=1),
}
SHOWCASES = [
    dict(id="blocks", label="three equal blocks of 4 inputs", family="blocks", theta=0.3, N=30, seed=0),
    dict(id="uneven", label="blocks of 2, 4 and 6 inputs with variances 2, 1, 0.6", family="blocks_uneven", theta=0.3, N=30, seed=0),
    dict(id="hier", label="12 inputs in a three-level hierarchy", family="hier", theta=0.3, N=36, seed=0),
    dict(id="hier_fine", label="the same hierarchy, higher threshold", family="hier", theta=0.6, N=36, seed=0),
    dict(id="blocks_high", label="three equal blocks, threshold above their correlation", family="blocks", theta=0.6, N=36, seed=0),
]


def train(Sigma, N, P, seed):
    rng = np.random.default_rng(seed)
    Wii, Wib, Mii, Mib = c.init_state(N, Sigma.shape[0], rng)
    r, t = np.inf, 0
    while t < TMAX:
        for _ in range(2000):
            Wii, Wib, _ = c.step(Wii, Wib, Mii, Mib, Sigma, P)
        t += 2000
        r = c.stationarity(Wii, Wib, Mii, Mib, Sigma, P)["r"]
        if r < RES_TOL:
            break
    return Wii, Wib, Mii, r, t


def corr(C):
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def analyse(Sigma, N, theta, gamma, seed, wmin):
    P = c.Params(theta=theta, gamma=gamma, wmin=wmin)
    Wii, Wib, Mii, res, steps = train(Sigma, N, P, seed)
    a = Wib.sum(1)
    K = np.diag(Wii.sum(1) + a) - Wii
    H = np.linalg.solve(K, Wib)
    Csp, Cev = np.linalg.inv(K), H @ Sigma @ H.T
    Rsp, Rev = corr(Csp), corr(Cev)
    lab = c.clusters_synapse(Wii, Mii, P.wmin)
    off = ~np.eye(N, dtype=bool)
    same = (lab[:, None] == lab[None, :]) & off
    diff = lab[:, None] != lab[None, :]
    pred = np.maximum(P.wmin, P.eta / P.lam * (gamma * Cev - theta))
    k = P.eta / P.lam
    clusters = []
    for cl in range(lab.max() + 1):
        m = lab == cl
        n = int(m.sum())
        if n < 2:
            continue
        o = ~np.eye(n, dtype=bool)
        h = H[m].mean(0)
        ac, win = float(a[m].mean()), Wii[np.ix_(m, m)][o]
        w = float(win.mean())
        meff = float(1 / (h @ h))
        mu = ac / k
        v = float(h @ Sigma @ h)
        clusters.append(dict(
            n=n, a=ac, w=w, w_min=float(win.min()), w_max=float(win.max()), m_eff=meff, mu=mu,
            r_sp=float(Rsp[np.ix_(m, m)][o].mean()), r_sp_min=float(Rsp[np.ix_(m, m)][o].min()),
            r_ev=float(Rev[np.ix_(m, m)][o].mean()),
            r_pred=w / (ac + w),
            r_pred_sigma=(gamma * mu / meff + (gamma - 1) * theta) / (gamma * mu / meff + (gamma - 1) * theta + mu),
            ev_var=v, ev_var_pred=mu / meff + theta,
            sp_var=float(np.diag(Csp)[m].mean()), sp_var_pred=(ac + w) / (ac * (ac + n * w)),
            sp_mean_var=float(Csp[np.ix_(m, m)].mean()), sp_mean_var_pred=1 / (n * ac),
            spread=float(np.abs(H[m] - h).sum(1).max())))
    summary = dict(
        theta=theta, gamma=gamma, seed=seed, N=N, wmin=wmin, residual=float(res), steps=steps,
        identity_err=float(np.abs(Wii - pred)[off].max()),
        n_clusters=int(lab.max() + 1), n_single=int((np.bincount(lab) == 1).sum()),
        r_sp_within=float(Rsp[same].mean()) if same.any() else None,
        r_sp_between=float(Rsp[diff].mean()) if diff.any() else None,
        r_sp_between_max=float(Rsp[diff].max()) if diff.any() else None,
        r_ev_within=float(Rev[same].mean()) if same.any() else None,
        r_ev_between=float(Rev[diff].mean()) if diff.any() else None,
        r_ev_between_absmax=float(np.abs(Rev[diff]).max()) if diff.any() else None,
        clusters=clusters)
    return summary, dict(Wii=Wii, Wib=Wib, K=K, H=H, Csp=Csp, Cev=Cev, lab=lab)


def langevin(K, seed, reps=2000, dt=0.002, n_samp=60):
    """Euler-Maruyama for du = -K u dt + sqrt(2 dt) xi (T = tau = 1); sample covariance over reps x n_samp.
    Burn-in of 10 and sample spacing of 2 slowest relaxation times (1/min eig K), capped at 60 and 2 time units."""
    slow = 1 / np.linalg.eigvalsh(K)[0]
    burn, every = int(np.ceil(min(10 * slow, 60) / dt)), int(np.ceil(min(2 * slow, 2) / dt))
    rng = np.random.default_rng([seed, 7])
    N = len(K)
    u = np.zeros((reps, N))
    acc = np.zeros((N, N))
    s = np.sqrt(2 * dt)
    for t in range(burn + n_samp * every):
        u += -(u @ K) * dt + s * rng.standard_normal(u.shape)
        if t >= burn and (t - burn) % every == every - 1:
            acc += u.T @ u / reps
    return acc / n_samp, dict(reps=reps, dt=dt, burn_steps=burn, n_samples=n_samp, every=every)


def sweep_unit(arg):
    fam, theta, gamma, seed, wmin = arg
    s, _ = analyse(FAMILIES[fam](), N_SWEEP, theta, gamma, seed, wmin)
    s["family"] = fam
    return s


def showcase_unit(sc):
    Sigma = FAMILIES[sc["family"]]()
    s, x = analyse(Sigma, sc["N"], sc["theta"], 1.0, sc["seed"], WMIN_SHOW)
    order = np.lexsort((np.arange(sc["N"]), x["lab"]))
    Cl, cfg = langevin(x["K"], sc["seed"])
    Ci = x["Csp"]
    ix = np.ix_(order, order)
    return dict(
        id=sc["id"], label=sc["label"], family=sc["family"], B=len(Sigma), summary=s,
        labels=x["lab"][order], Sigma=Sigma, H=x["H"][order], K=x["K"][ix], a=x["Wib"].sum(1)[order],
        Csp=Ci[ix], Cev=x["Cev"][ix], Csp_langevin=Cl[ix],
        langevin=dict(cfg, rel_err=float(np.linalg.norm(Cl - Ci) / np.linalg.norm(Ci)),
                      max_abs_corr_err=float(np.abs(corr(Cl) - corr(Ci)).max())))


def main():
    shows = oa.pool_map(showcase_unit, SHOWCASES)
    for s in shows:
        print(s["id"], "residual %.1e" % s["summary"]["residual"], "clusters", s["summary"]["n_clusters"],
              "langevin rel err %.3f" % s["langevin"]["rel_err"])
    grid = [(f, th, 1.0, sd) for f in FAMILIES for th in THETAS for sd in SEEDS]
    sweep = oa.pool_map(sweep_unit, [g + (WMIN_SWEEP,) for g in grid])
    sweep_floor = oa.pool_map(sweep_unit, [g + (WMIN_SHOW,) for g in grid])
    gam = oa.pool_map(sweep_unit, [(f, 0.3, g, sd, WMIN_SWEEP) for f in ("blocks", "blocks_uneven", "hier") for g in GAMMAS for sd in SEEDS])
    for name, runs in (("sweep", sweep), ("sweep_floor", sweep_floor), ("gamma", gam)):
        cl = [k for r in runs for k in r["clusters"]]
        d = np.array([abs(k["r_sp"] - k["r_pred"]) for k in cl])
        d2 = np.array([abs(k["r_sp"] - k["r_pred_sigma"]) for k in cl])
        print(f"{name}: {len(runs)} runs, max residual {max(r['residual'] for r in runs):.1e}, max identity err "
              f"{max(r['identity_err'] for r in runs):.1e}, {len(cl)} clusters, |r_sp - w/(a+w)| max {d.max():.3f} median {np.median(d):.4f}, "
              f"|r_sp - formula(mu, m)| max {d2.max():.3f} median {np.median(d2):.4f}")
    meta = dict(
        script="hebb/sweeps2/spontaneous.py",
        description="Spontaneous activity of covariance-rule networks at their fixed point: inputs silent, independent "
                    "white noise in every neuron (units T = tau = 1). Stationary covariance C_sp = K^-1 against the evoked "
                    "covariance C_ev = H Sigma H^T, cluster by cluster, with the closed forms of the module docstring.",
        config=dict(rule="hebb.core.step, Params(lam=1, eta=1, dt=0.05), full masks (p = 1)", wmin_showcases=WMIN_SHOW, wmin_sweep=WMIN_SWEEP,
                    families={k: "see FAMILIES in the script" for k in FAMILIES}, thetas=THETAS, gammas=GAMMAS,
                    N_sweep=N_SWEEP, tmax_steps=TMAX, clusters="connected components of W_II > wmin (core.clusters_synapse)"),
        seeds=SEEDS, res_tol=RES_TOL,
        seed_streams="initial weights numpy default_rng(seed); Langevin noise default_rng([seed, 7]); covariance presets fixed",
        fields={
            "showcases[]": "one fully stored network each; neurons sorted by cluster (labels). Sigma (B x B), H (N x B), "
                           "K (N x N), a (input conductance per neuron), Csp = K^-1, Cev = H Sigma H^T, Csp_langevin = "
                           "sample covariance of the simulated noise-driven network, langevin.rel_err = Frobenius relative error",
            "sweep[]": "gamma = 1 runs over families x thetas x seeds at wmin_sweep; sweep_floor[]: the same at wmin_showcases; gamma[]: theta = 0.3 runs over gammas at wmin_sweep",
            "residual": "projected residual max |dw/dt| at the end (weights/time)",
            "identity_err": "max over neuron pairs |W_II,ij - max(wmin, (eta/lam)(gamma C_ev,ij - theta))|",
            "r_sp_within / r_sp_between": "mean spontaneous correlation over pairs in the same / different clusters; r_ev_*: same for evoked",
            "clusters[]": "clusters with n >= 2 members",
            "a, w, w_min, w_max": "member-mean input conductance; mean, min, max lateral weight between members",
            "m_eff, mu": "1/|h|^2 of the mean mixture h; mu = a lam/eta",
            "r_sp, r_sp_min, r_ev": "mean and min pairwise spontaneous correlation between members; mean evoked correlation",
            "r_pred, r_pred_sigma": "w/(a + w); the same written with (mu, m_eff, theta, gamma) only (1/(1 + m_eff) at gamma = 1)",
            "ev_var, ev_var_pred": "h^T Sigma h; mu/m_eff + theta",
            "sp_var, sp_var_pred": "member-mean spontaneous variance (units T); (a + w)/(a (a + n w))",
            "sp_mean_var, sp_mean_var_pred": "spontaneous variance of the cluster mean; 1/(n a)",
            "spread": "max L1 distance from a member's mixture to the mean mixture",
        })
    oa.dump("spontaneous", meta, dict(showcases=shows, sweep=sweep, sweep_floor=sweep_floor, gamma=gam), sig=5)


if __name__ == "__main__":
    main()
