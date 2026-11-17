"""Q-resolution (SPEC §3, eq. 10-11): three statements kept apart.

(1) Frozen-response anisotropic law, linear case: D_ij^2 = sum_a lambda_a (Delta h_a)^2 with
    Delta h = h_i - h_j in the eigenbasis. Exact for the training bank's C_hat; with Sigma_g it
    carries the sampling error. Checked on random frozen networks.
(2) Edge activation from the uncoupled state for planted pairs differing along ONE contrast mode:
    the edge has positive drift at W=0 iff |Delta h_a| < sqrt(2b / lambda_a) (lambda_a -> the
    bank value v_a^T C_hat v_a for the exact statement).
(3) Self-consistent fusion under continuation in b: (3a) planted pairs, closed form for the
    loss of stability; (3b) a population spread along one mode: self-consistent cluster count
    vs the uncoupled-threshold prediction, cold start and continuation up/down; exact
    (b, alpha) -> lambda (b, alpha) scaling.

Run: uv run python -m hebb.sweeps2.resolution   -> site/data/sweeps2/resolution.json
"""
import hebb.sweeps2  # noqa: F401

import numpy as np

from hebb import onsite as o
from hebb import onsite_analysis as oa

B = 8
V = oa.contrast_basis(B)                              # column 0 = uniform, 1..7 contrasts (Hadamard)
LAM = np.array([1.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625])
SIGMA = oa.sigma_from_eig(V, LAM)
H0 = np.full(B, 1.0 / B)
XMAX = (1.0 / B) * np.sqrt(B)                         # |x| <= XMAX keeps h0 + x v_a >= 0 (Hadamard entries +-1/sqrt(B))
SEEDS = [0, 1, 2, 3, 4]


def lam_hat(G, a):
    C = G.T @ G / len(G)
    return float(V[:, a] @ C @ V[:, a])


# ----------------------------------------------------------------------------- (1) frozen law
def part1():
    rows = []
    for seed in SEEDS:
        for sig_name, Sig in (("contrast8", SIGMA), ("power8", oa.sigma_power(B))):
            G, Ge = oa.banks(Sig, seed)
            lam, Vs = oa.eig_desc(Sig)
            for wscale in (0.0, 0.5):
                rng = oa.rng_for(seed, 7)
                net = oa.make_net(30, B, seed, s=3)
                if wscale:
                    W = rng.uniform(0, wscale, (30, 30)) * (rng.random((30, 30)) < 0.2)
                    W = np.triu(W, 1); net.W = W + W.T
                H = o.harmonic(net)
                dH = H[:, None, :] - H[None, :, :]
                iu = np.triu_indices(30, 1)
                dh = dH[iu]
                D2_train = o.dist2(G @ H.T)[iu]
                D2_eval = o.dist2(Ge @ H.T)[iu]
                Ch = G.T @ G / len(G)
                law_chat = np.einsum("pb,bc,pc->p", dh, Ch, dh)
                law_sigma = ((dh @ Vs) ** 2 * lam).sum(1)
                scale = D2_train.max()
                rows.append([seed, sig_name, wscale,
                             float(np.abs(D2_train - law_chat).max() / scale),
                             float(np.abs(D2_train - law_sigma).max() / scale),
                             float(np.abs(D2_eval - law_sigma).max() / scale),
                             float(np.abs(dh.sum(1)).max())])
    return dict(columns=["seed", "sigma", "W_scale(0=uncoupled)", "rel_err_train_vs_Chat_law (identity)",
                         "rel_err_train_vs_Sigma_law (sampling)", "rel_err_eval_vs_Sigma_law (sampling, S_eval)",
                         "max |sum_b Delta h_b| (should be 0)"], rows=rows)


# ----------------------------------------------------------------------------- (2) activation
B_ACT = 0.005
ALPHA_ACT = 0.1


def pair_net(a, delta, pot=oa.LINEAR, deg=1.0):
    h1, h2 = H0 + 0.5 * delta * V[:, a], H0 - 0.5 * delta * V[:, a]
    return o.Net(A=deg * np.vstack([h1, h2]), t=0.0, klo=pot["klo"], khi=pot["khi"])


def part2():
    rows = []
    mis_chat, mis_sigma, n = 0, 0, 0
    nl = []
    for seed in SEEDS:
        G, _ = oa.banks(SIGMA, seed)
        for a in range(1, B):
            lh = lam_hat(G, a)
            d_sig, d_hat = np.sqrt(2 * B_ACT / LAM[a]), np.sqrt(2 * B_ACT / lh)
            for f in np.linspace(0.6, 1.4, 17):
                delta = f * d_sig
                net = pair_net(a, delta)
                drift0 = o.drift(net, G, B_ACT, ALPHA_ACT)[0][0, 1]
                r = oa.solve(net, G, B_ACT, ALPHA_ACT, T=200000)
                alive = bool(net.W[0, 1] > 1e-9)
                mis_chat += alive != (delta < d_hat)
                mis_sigma += alive != (delta < d_sig)
                n += 1
                if seed == 0:
                    rows.append([a, LAM[a], lh, float(f), float(delta), float(drift0), float(net.W[0, 1]),
                                 alive, r["residual"]])
            if seed == 0:
                # nonlinear (homogeneous rectifying, t = 0): uncoupled D0^2 vs the linear prediction
                for f in (0.25, 0.5, 1.0):
                    delta = f * XMAX
                    pot = dict(klo=oa.RECT["klo"], khi=oa.RECT["khi"])
                    Ul = o.equilibrate(pair_net(a, delta), G)
                    Un = o.equilibrate(pair_net(a, delta, pot), G)
                    nl.append([a, float(delta), float(o.dist2(Ul)[0, 1]), float(o.dist2(Un)[0, 1])])
    return dict(b=B_ACT, alpha=ALPHA_ACT,
                columns=["mode a", "lambda_a", "lambda_hat_a = v_a^T C_hat v_a", "delta / sqrt(2b/lambda_a)",
                         "delta = |Delta h_a|", "drift at W=0", "w end (from W=0)", "alive", "residual"],
                rows_seed0=rows, n_pairs=n, misclassified_by_Chat_threshold=int(mis_chat),
                misclassified_by_Sigma_threshold=int(mis_sigma),
                nonlinear_columns=["mode a", "delta", "D0^2 linear", "D0^2 rect (klo=3, khi=0, t=0)"],
                nonlinear_rows=nl)


# ----------------------------------------------------------------------------- (3a) pairs, continuation
def delta_off(b, alpha, lam, deg=1.0):
    """Largest |Delta h_a| for which a fused pair survives at (b, alpha): from
    b = b_sn(c), c = lam delta^2 / 2:  c_max = 16 (b + kappa/2)^3 / (27 kappa^2) when b > kappa/4."""
    k = alpha * deg
    if b <= k / 4:
        return np.sqrt(2 * b / lam)
    return np.sqrt(2 * 16 * (b + k / 2) ** 3 / (27 * k * k) / lam)


def b_sn(c, alpha, deg=1.0):
    k = alpha * deg
    if 4 * c / k <= 1:
        return c
    return 3 * 2 ** (-4 / 3) * (c * k * k) ** (1 / 3) - k / 2


def pair_cont_unit(args):
    a, alpha = args
    G, _ = oa.banks(SIGMA, 0)
    lh = lam_hat(G, a)
    c = 0.01                                           # target uncoupled D0^2 / 2 (D^2 units)
    delta = np.sqrt(2 * c / lh)
    b_on_pred, b_off_pred = c, b_sn(c, alpha)
    bs = np.linspace(0.3 * b_off_pred, 1.3 * c, 81)
    net = pair_net(a, delta)
    up = []
    for b in bs:
        oa.solve(net, G, b, alpha, T=400000)
        up.append(net.W[0, 1] > 1e-9)
    down = []
    for b in bs[::-1]:
        oa.solve(net, G, b, alpha, T=400000)
        down.append(net.W[0, 1] > 1e-9)
    down = down[::-1]
    return [a, lh, alpha, float(delta), b_on_pred, float(bs[int(np.argmax(up))]), b_off_pred,
            float(bs[int(np.argmax(down))]), float(bs[1] - bs[0])]


def part3a():
    rows = oa.pool_map(pair_cont_unit, [(a, al) for a in (1, 4) for al in (0.005, 0.01, 0.02, 0.04)])
    # analytic table of the resolution shift delta_off / delta_on
    tab = []
    for alpha in (0.01, 0.1, 1.0):
        for b in (0.005, 0.02, 0.05, 0.1, 0.3):
            tab.append([alpha, b, float(delta_off(b, alpha, 1.0) / np.sqrt(2 * b))])
    return dict(columns=["mode a", "lambda_hat", "alpha", "delta", "b_on predicted (=c)", "b_on simulated (first b alive, going up)",
                         "b_off predicted (saddle-node)", "b_off simulated (lowest b alive, going down)", "b grid step"],
                rows=rows,
                ratio_columns=["alpha", "b", "delta_off/delta_on (any lambda; deg=1)"], ratio_rows=tab)


# ----------------------------------------------------------------------------- (3b) population on a line
NPOP = 30
BPOP = np.round(np.geomspace(2e-4, 0.05, 14), 6)


def line_net(a, seed, deg=1.0):
    x = np.sort(oa.rng_for(seed, 0).uniform(-0.95 * XMAX, 0.95 * XMAX, NPOP))
    return o.Net(A=deg * (H0[None] + x[:, None] * V[:, a][None]), t=0.0, klo=0.0, khi=0.0), x


def uncoupled_K(x, b, lh):
    """Components of the graph whose edges have positive drift at W=0: lh (x_i - x_j)^2 < 2b."""
    gaps = np.diff(np.sort(x))
    return int(1 + (lh * gaps ** 2 >= 2 * b).sum())


def pop_unit(args):
    a, alpha, seed = args
    G, Ge = oa.banks(SIGMA, seed)
    lh = lam_hat(G, a)
    res = {}
    net, x = line_net(a, seed)
    cold = []
    for b in BPOP:
        n2 = o.Net(A=net.A.copy(), t=0.0, klo=0.0, khi=0.0)
        r = oa.solve(n2, G, b, alpha)
        cs = oa.cluster_summary(n2.W, r["U"], b)
        # uncoupled separation bridged inside a component: max gap in x between consecutive members
        lab = cs["lab"]
        mg = max([np.diff(x[lab == k]).max() if (lab == k).sum() > 1 else 0.0 for k in range(cs["K"])])
        De = np.sqrt(np.clip(o.dist2(Ge @ o.harmonic(n2).T), 0, None))
        span_e = max(De[np.ix_(lab == k, lab == k)].max() for k in range(cs["K"]))
        cold.append([cs["K"], uncoupled_K(x, b, lh), float(mg / np.sqrt(2 * b / lh)), cs["span_over_range"],
                     r["residual"], float(span_e / np.sqrt(2 * b))])
    res["cold"] = cold
    for name, order in (("up", BPOP), ("down", BPOP[::-1])):
        n2 = o.Net(A=net.A.copy(), t=0.0, klo=0.0, khi=0.0)
        seq = []
        for b in order:
            r = oa.solve(n2, G, b, alpha)
            seq.append([oa.cluster_summary(n2.W, r["U"], b)["K"], r["residual"]])
        res[name] = seq if name == "up" else seq[::-1]
    return dict(a=a, alpha=alpha, seed=seed, lam_hat=lh, **res)


def scaling_check():
    """Exact scaling: on a line along mode a, D^2 = lam_hat (x_i - x_j)^2 at every W, so the fixed
    point depends on (b, alpha) only through (b/lam_hat, alpha/lam_hat)."""
    G, _ = oa.banks(SIGMA, 0)
    out = []
    for b1, al1 in ((0.004, 0.05), (0.01, 0.2)):
        Ws = []
        for a in (1, 3):
            net, _ = line_net(a, 0)
            lh = lam_hat(G, a)
            oa.solve(net, G, b1 * lh, al1 * lh)   # drift = lh (b1 - al1 w - (x_i-x_j)^2/2): same W
            Ws.append(net.W.copy())
        out.append([b1, al1, float(np.abs(Ws[0] - Ws[1]).max()), float(Ws[0].max())])
    return dict(columns=["b/lam_hat", "alpha/lam_hat", "max |W(mode 1) - W(mode 3)|", "max w"], rows=out)


def main():
    p1 = part1()
    print("part1 done", max(r[3] for r in p1["rows"]), max(r[4] for r in p1["rows"]))
    p2 = part2()
    print("part2 done", p2["misclassified_by_Chat_threshold"], p2["misclassified_by_Sigma_threshold"], p2["n_pairs"])
    p3a = part3a()
    print("part3a done")
    for r in p3a["rows"]:
        print(np.round(r, 5))
    jobs = [(a, al, s) for a in (1, 3) for al in (0.01, 0.1) for s in SEEDS]
    pops = oa.pool_map(pop_unit, jobs)
    sc = scaling_check()
    print("scaling", sc)
    meta = dict(
        script="hebb/sweeps2/resolution.py",
        description="Q-resolution: (1) frozen anisotropic law, (2) edge activation of planted single-mode pairs from "
                    "W=0, (3a) pair continuation vs closed-form saddle-node, (3b) population along one mode: "
                    "self-consistent K vs the uncoupled threshold, cold and continuation; exact scaling check.",
        config=dict(B=B, Sigma="V diag(lam) V^T, V = normalised Hadamard (col 0 uniform), lam = %s" % LAM.tolist(),
                    S_train=oa.S_TRAIN, S_eval=oa.S_EVAL, pairs="A rows = deg*(1/B +- delta/2 v_a), deg 1, linear",
                    b_activation=B_ACT, alpha_activation=ALPHA_ACT, population=dict(N=NPOP, b_grid=BPOP,
                    positions="x ~ U(-0.95 xmax, 0.95 xmax), xmax = %.4f, A_i = 1/B + x_i v_a" % XMAX)),
        seeds=SEEDS,
        fields={
            "p1": "frozen law errors, relative to max D^2 (see columns); D^2 in response-variance units",
            "p2.rows_seed0": "planted pairs, seed 0 (see columns); delta in attachment-share units",
            "p2.misclassified_*": "pairs (all seeds) whose simulated outcome from W=0 disagrees with the threshold computed with C_hat / Sigma_g",
            "p2.nonlinear_rows": "uncoupled D0^2 of the same planted pair, linear vs homogeneous rectifying potential",
            "p3a.rows": "pair continuation: switch-on/off b (D^2 units) predicted vs simulated (grid resolution 'b grid step')",
            "p3a.ratio_rows": "analytic delta_off/delta_on = sqrt(16 (b+alpha/2)^3 / (27 alpha^2 b)) for b > alpha/4, else 1",
            "pop[].cold[k]": "[K self-consistent, K predicted by uncoupled threshold, max x-gap bridged inside a component / sqrt(2b/lam_hat), span/sqrt(2b) (training bank), residual, span/sqrt(2b) on the evaluation bank] at b_grid[k]",
            "pop[].up|down[k]": "[K, residual] along continuation",
            "scaling": "max |W| difference between two modes after scaling (b, alpha) by lam_hat (exact identity => solver tolerance)",
        })
    oa.dump("resolution", meta, dict(p1=p1, p2=p2, p3a=p3a, pop=pops, scaling=sc), sig=5)


if __name__ == "__main__":
    main()
