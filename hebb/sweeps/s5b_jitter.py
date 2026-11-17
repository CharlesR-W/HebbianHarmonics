"""Q4 follow-up: is the hysteresis loop of s5 real, or an artefact of exact permutation symmetry?

In mean-field AND in minibatch mode, neurons that have become identical ("clones") receive identical
updates forever, so a merged cluster can never split, and a symmetric state can sit on an unstable
fixed point. Here every weight additionally gets PRIVATE multiplicative jitter each step,
w <- w * exp(sigma * xi) (xi iid N(0,1), symmetric for I-I), which breaks the symmetry.
Same theta loop as s5 (0.02 -> 0.80 -> 0.02, step 0.02), 1500 steps per theta, block12 and hier8,
gamma in {1, 0}, sigma in {1e-3, 1e-2}.

Output: site/data/sweeps/hysteresis_jitter.json
"""
import numpy as np

from hebb import core
from hebb.core import Params, harmonic, init_state
from hebb.sweeps.common import Timer, cluster_embed, cov_block, cov_hier, pmap, save_json

COVS = {"block12": cov_block([4, 4, 4], 0.5, 0.1)[0], "hier8": cov_hier((0.6, 0.3, 0.05), (2, 2, 2))}
N, WMIN, STEPS = 120, 1e-4, 1500


def job(args):
    cname, gamma, sigma, seed = args
    S = COVS[cname]
    rng = np.random.default_rng(seed)
    Wii, Wib, Mii, Mib = init_state(N, S.shape[0], rng, 1.0, 0.0, 0.2)
    th_up = np.round(np.arange(0.02, 0.801, 0.02), 3)
    path = list(th_up) + list(th_up[::-1][1:])
    out = []
    for i, th in enumerate(path):
        P = Params(theta=float(th), gamma=gamma, wmin=WMIN)
        for _ in range(STEPS):
            Wii, Wib, _ = core.step(Wii, Wib, Mii, Mib, S, P)
            if sigma:
                X = np.triu(rng.standard_normal((N, N)), 1)
                Wii = Wii * np.exp(sigma * (X + X.T))
                Wib = Wib * np.exp(sigma * rng.standard_normal(Wib.shape))
        H = harmonic(np.maximum(Wii, WMIN * Mii * (1 - np.eye(N))), np.maximum(Wib, WMIN * Mib))
        l, sizes, cents = cluster_embed(H, 0.1)
        dead = Wib.max(1) <= 1.5 * WMIN * np.exp(4 * sigma)
        alive = [int(s) for k, s in enumerate(sizes) if not dead[l == k].all()]
        K = sum(1 for s in alive if s >= max(2, 0.02 * N))
        out.append(dict(theta=float(th), dir="up" if i < len(th_up) else "down", K=K, dead=float(dead.mean()),
                        maxfrac=max(alive) / N if alive else 0.0))
    return dict(cov=cname, gamma=gamma, sigma=sigma, seed=seed, path=out)


def main():
    jobs = [(c, g, s, 0) for c in COVS for g in (1.0, 0.0) for s in (1e-3, 1e-2)]
    with Timer(f"jitter loops ({len(jobs)})"):
        R = pmap(job, jobs)
    for r in R:
        print(r["cov"], r["gamma"], r["sigma"], " ".join(f"{p['theta']:.2f}{p['dir'][0]}:{p['K']}" for p in r["path"][::2]))
    save_json("hysteresis_jitter", dict(loops=R), dict(
        script="hebb/sweeps/s5b_jitter.py",
        description="theta loop as in hysteresis.json but with private multiplicative weight jitter w <- w exp(sigma xi) "
                    "every step (breaks the permutation symmetry that makes merged clusters unsplittable). 1500 steps "
                    "(dt=0.05) per theta value, no convergence test.",
        N=N, wmin=WMIN, seed=0, params=dict(lam=1, eta=1, dt=0.05, ii_scale=0.0, ib_scale=0.2)))


if __name__ == "__main__":
    main()
