"""Cross-check site/js/onsite_engine.js against hebb/onsite.py (node vs numpy).

Run: uv run --with pytest pytest -q tests/
The two sides use different PRNG streams, so the model arrays (A, t, klo, khi, W, G) are
built in Python and passed to node as JSON.  The engine's own PRNG, attachments and input
sampling are checked bit-for-bit against a Python mirror of its draw order (JsRng below).

Tolerances.  Both equilibrium solvers are exact up to roundoff (semismooth Newton stops on
a self-consistent activity pattern, or on max|dE/du| <= 1e-10): responses agree to 1e-9
absolute.  D2, drift, energy, harmonic matrix and population-mode eigenvalues inherit that
(1e-9, relative where stated).  Euler trajectories agree to 1e-9 after 60 steps; runs to a
fixed point agree to 1e-7 in W.
"""
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hebb import onsite as o

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "onsite_runner.mjs"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def js(op, **args):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"op": op, **args}, f, default=_np)
    out = subprocess.run(["node", str(RUNNER), f.name], capture_output=True, text=True, timeout=600)
    Path(f.name).unlink()
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _np(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    raise TypeError(type(x))


def amax(a, b):
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).max())


def rel(a, b):
    b = np.asarray(b, float)
    return amax(a, b) / max(float(np.abs(b).max()), 1e-300)


def net_json(net):
    return dict(A=net.A, t=net.t, klo=net.klo, khi=net.khi, W=net.W)


# ───────────── Python mirror of the engine's PRNG and draw order ─────────────

_M32 = 0xFFFFFFFF


class JsRng:
    """mulberry32, bit-identical to onsite_engine.js makeRng."""

    def __init__(self, seed=0):
        self.a = int(seed) & _M32

    def uniform(self):
        self.a = (self.a + 0x6D2B79F5) & _M32
        a = self.a
        t = ((a ^ (a >> 15)) * (1 | a)) & _M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & _M32)) & _M32) ^ t
        return ((t ^ (t >> 14)) & _M32) / 4294967296.0

    def normal(self):
        u1 = self.uniform(); u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)

    def exponential(self, scale=1.0):
        return -scale * math.log(1.0 - self.uniform())

    def int(self, n):
        return int(math.floor(self.uniform() * n))

    def laplace(self):
        sg = -1.0 if self.uniform() < 0.5 else 1.0
        return sg * self.exponential(1.0) * math.sqrt(0.5)


def py_make_net(N, B, seed, s, deg, sigmaT, equal):
    r = JsRng(seed)
    A = np.zeros((N, B))
    for i in range(N):
        perm = list(range(B))
        for q in range(s):
            k = q + r.int(B - q)
            perm[q], perm[k] = perm[k], perm[q]
        e = [1.0 if equal else r.exponential(1.0) for _ in range(s)]
        tot = 0.0
        for x in e:
            tot += x
        for q in range(s):
            A[i, perm[q]] = deg * e[q] / tot
    t = np.array([sigmaT * r.normal() for _ in range(N)])
    return A, t


def py_sample(Sigma, S, seed, kind):
    r = JsRng(seed)
    lam, V = np.linalg.eigh(Sigma)
    root = (V * np.sqrt(np.clip(lam, 0, None))) @ V.T
    draw = r.normal if kind == "gauss" else r.laplace
    Z = np.array([[draw() for _ in range(len(Sigma))] for _ in range(S)])
    return Z @ root


# ───────────── fixtures ─────────────

def setup(seed=0, klo=3.0, khi=0.5, sigma_t=0.7, N=14, B=6, S=64, p=0.4, wmax=0.3, mixed=False):
    """As tests/test_onsite.py _setup; mixed=True draws per-neuron slopes with klo < khi
    for some neurons (concave and convex transfers in the same net)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((B, B)); Sigma = M @ M.T / B
    net = o.make_net(N, B, rng, s=3, klo=klo, khi=khi, sigma_t=sigma_t)
    if mixed:
        net.klo = rng.uniform(0, 5, N); net.khi = rng.uniform(0, 5, N)
    W = rng.uniform(0, wmax, (N, N)) * (rng.random((N, N)) < p); W = np.triu(W, 1); net.W = W + W.T
    return net, o.sample_inputs(Sigma, S, rng), Sigma


CASES = {
    "linear": dict(klo=0.0, khi=0.0, sigma_t=0.0),
    "leak": dict(klo=0.5, khi=0.5, sigma_t=0.3),
    "rectifying": dict(klo=3.0, khi=0.5, sigma_t=0.7),
    "strong": dict(klo=10.0, khi=0.0, sigma_t=1.0),
    "saturating": dict(klo=0.0, khi=4.0, sigma_t=0.5),
    "mixed": dict(mixed=True, sigma_t=0.8),
    "big_dense": dict(klo=3.0, khi=0.5, sigma_t=0.7, N=40, B=12, S=128, p=0.8, wmax=2.0),
    "uncoupled": dict(klo=3.0, khi=0.5, sigma_t=0.7, p=0.0),
}


# ───────────── PRNG, attachments, sampling, generators ─────────────

def test_rng_streams_bit_identical():
    j = js("rng", seed=12345, n=50)
    r = JsRng(12345)
    assert j["uniform"] == [r.uniform() for _ in range(50)]
    # normal/laplace go through cos/log, which may differ from Python's in the last ulp
    assert amax(j["normal"], [r.normal() for _ in range(50)]) < 1e-14
    assert amax(j["laplace"], [r.laplace() for _ in range(50)]) < 1e-14


@pytest.mark.parametrize("equal", [False, True])
def test_make_net_bit_identical(equal):
    j = js("make_net", N=25, B=9, seed=4, s=3, deg=1.5, klo=2, khi=0.5, sigmaT=0.6, equal=equal)
    A, t = py_make_net(25, 9, 4, 3, 1.5, 0.6, equal)
    assert amax(j["A"], A) < 1e-15 and amax(j["t"], t) < 1e-14
    A = np.array(j["A"])
    assert ((A > 0).sum(1) == 3).all() and np.allclose(A.sum(1), 1.5)


@pytest.mark.parametrize("kind", ["gauss", "laplace"])
def test_sample_inputs_mirror(kind):
    rng = np.random.default_rng(3)
    M = rng.standard_normal((7, 7)); Sigma = M @ M.T / 7
    j = js("sample", Sigma=Sigma, S=300, seed=9, kind=kind)
    G = py_sample(Sigma, 300, 9, kind)
    assert amax(j["G"], G) < 1e-12


def test_laplace_has_the_covariance_and_heavy_tails():
    Sigma = np.diag([1.0, 2.0, 0.5])
    G = np.array(js("sample", Sigma=Sigma, S=40000, seed=1, kind="laplace")["G"])
    assert np.abs(G.T @ G / len(G) - Sigma).max() < 0.06
    z = G[:, 0]
    assert (z ** 4).mean() / (z ** 2).mean() ** 2 > 5.0          # Laplace kurtosis is 6


def test_generators():
    core = pytest.importorskip("hebb.core")
    pairs = [
        ("cov3", dict(variances=[1, 2, 0.5], corr=[0.9, 0.9, -0.9]), core.cov3((1, 2, 0.5), (0.9, 0.9, -0.9))),
        ("covBlocks", dict(sizes=[3, 4, 2], within=[0.6, 0.4, 0.8], between=0.1, variances=[1, 2, 0.5]),
         core.cov_blocks((3, 4, 2), [0.6, 0.4, 0.8], 0.1, [1, 2, 0.5])[0]),
        ("covHierarchical", dict(branching=[2, 3, 2], corr=[0.05, 0.3, 0.7], variance=1.5),
         core.cov_hierarchical((2, 3, 2), (0.05, 0.3, 0.7), 1.5)[0]),
        ("covPowerLaw", dict(B=16, alpha=1.2, seed=3, nonneg=True), core.cov_power_law(16, 1.2, 3, True)),
    ]
    for name, args, ref in pairs:
        assert rel(js("gen", name=name, args=args)["Sigma"], ref) < 1e-9, name
    S1 = np.array(js("gen", name="covRankOne", args=dict(B=5, v=[1, 2, 0, -1, 0.5]))["Sigma"])
    v = np.array([1, 2, 0, -1, 0.5]); assert amax(S1, np.outer(v, v)) == 0


# ───────────── equilibrium, geometry, rule ─────────────

@pytest.mark.parametrize("case", list(CASES))
def test_equilibrium_matches_reference(case):
    net, G, _ = setup(**CASES[case])
    U = o.equilibrate(net, G)
    for exact in (False, True):
        j = js("equilibrate", net=net_json(net), G=G, exact=exact)
        assert amax(j["U"], U) < 1e-9, (exact, amax(j["U"], U))
        assert j["residual"] < 1e-9
    # warm start from a perturbed equilibrium reaches the same point
    U0 = U + 0.3 * np.random.default_rng(1).standard_normal(U.shape)
    j = js("equilibrate", net=net_json(net), G=G, U0=U0)
    assert amax(j["U"], U) < 1e-9 and j["residual"] < 1e-9


@pytest.mark.parametrize("case", ["linear", "rectifying", "mixed", "big_dense"])
def test_geometry_drift_energy_harmonic(case):
    net, G, _ = setup(**CASES[case])
    b, alpha = 0.15, 0.8
    U = o.equilibrate(net, G)
    j = js("analyze", net=net_json(net), G=G, b=b, alpha=alpha, eps=1e-9)
    D2 = o.dist2(U)
    off = ~np.eye(net.N, dtype=bool)
    assert amax(np.array(j["D2"])[off], D2[off]) < 1e-9 and np.abs(np.diag(j["D2"])).max() == 0
    dW, _ = o.drift(net, G, b, alpha, U)
    assert amax(j["dW"], dW) < 1e-9
    assert abs(j["residual"] - o.residual(net, G, b, alpha, U)) < 1e-9
    assert abs(j["energy"] - o.energy(net, U, G)) < 1e-9 * max(1, abs(o.energy(net, U, G)))
    assert rel(j["L"], o.laplacian(net)) < 1e-14
    assert rel(j["H"], o.harmonic(net)) < 1e-12
    # population modes: eigenvalues, d_eff, operator, and the top eigenfunction
    pm = o.population_modes(U)
    q = np.array(j["modes"]["q"])
    assert np.abs(q - pm["q"]).max() < 1e-10 * pm["q"][0]
    assert abs(j["modes"]["d_eff"] - pm["d_eff"]) < 1e-8
    assert amax(j["modes"]["C"], pm["C"]) < 1e-10
    Psi = np.array(j["modes"]["Psi"])
    assert Psi.shape[1] == pm["Psi"].shape[1]
    assert np.allclose(Psi.T @ Psi / len(G), np.eye(Psi.shape[1]), atol=1e-8)
    if pm["q"][0] > 1.01 * pm["q"][1]:
        assert abs(abs(Psi[:, 0] @ pm["Psi"][:, 0]) / len(G) - 1) < 1e-8


def test_components_and_cluster_stats_at_a_fixed_point():
    net, G, _ = setup(N=24, B=8, S=96, p=0.0)
    r = o.run(net, G, b=0.12, alpha=1.0)
    lab = o.components(net.W)
    assert 1 < lab.max() + 1 < net.N                              # a nontrivial partition
    j = js("analyze", net=net_json(net), G=G, b=0.12, alpha=1.0, eps=1e-9)
    assert j["labels"] == lab.tolist() and j["K"] == lab.max() + 1
    for cj, cp in zip(j["stats"], o.cluster_stats(r["U"], lab)):
        assert cj["members"] == cp["members"].tolist() and cj["size"] == cp["size"]
        assert amax(cj["proto"], cp["proto"]) < 1e-9
        for k in ("diam_max", "diam_mean", "d_eff"):
            assert abs(cj[k] - cp[k]) < 1e-8, k


# ───────────── trajectories ─────────────

@pytest.mark.parametrize("case", ["linear", "rectifying", "mixed", "big_dense"])
def test_live_loop_trajectory(case):
    """sim.step() n times == n calls of onsite.step (fresh equilibria each step)."""
    net, G, _ = setup(**CASES[case])
    b, alpha, dt, n = 0.2, 0.7, 0.15, 60
    W0 = net.W.copy()
    j = js("trajectory", net=net_json(net), G=G, b=b, alpha=alpha, dt=dt, nSteps=n, record=[1, 10, 60])
    for k in range(1, n + 1):
        o.step(net, G, b, alpha, dt)
        if str(k) in j["snaps"]:
            assert amax(j["snaps"][str(k)], net.W) < 1e-9, (k, amax(j["snaps"][str(k)], net.W))
    assert j["maxEqResidual"] < 1e-9
    assert abs(j["residual"] - o.residual(net, G, b, alpha)) < 1e-9
    assert amax(W0, net.W) > 1e-3                                 # it actually moved


@pytest.mark.parametrize("case,b,alpha", [("linear", 0.1, 1.0), ("rectifying", 0.1, 1.0), ("strong", 0.25, 0.5),
                                          ("mixed", 0.15, 1.0)])
def test_run_to_fixed_point(case, b, alpha):
    net, G, _ = setup(**CASES[case])
    net.W[:] = 0
    j = js("run", net=net_json(net), G=G, b=b, alpha=alpha, dt=0.2, T=5000, tol=1e-8, exact=False)
    r = o.run(net, G, b, alpha)
    assert abs(j["steps"] - r["steps"]) <= 1
    assert amax(j["W"], net.W) < 1e-7
    assert j["residual"] < 1e-6 and j["eqResidual"] < 1e-9
    D2 = o.dist2(np.array(j["U"]))
    off = ~np.eye(net.N, dtype=bool)
    assert np.abs(np.array(j["W"]) - np.maximum(b - D2 / 2, 0) / alpha)[off].max() < 1e-5
    assert (np.array(j["W"]) == 0)[off].any() and (np.array(j["W"]) > 0).any()   # live and dead edges


def test_two_neuron_bistability():
    """SPEC Q-fusion calibration: D2(w) = D0^2/(1+2w)^2; both basins reached."""
    G2 = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    b, alpha = 0.4, 0.2
    ws = []
    for w0 in (0.0, 3.0):
        net = dict(A=np.eye(2), t=[0, 0], klo=[0, 0], khi=[0, 0], W=np.array([[0, w0], [w0, 0.0]]))
        j = js("run", net=net, G=G2, b=b, alpha=alpha, dt=0.05, T=20000, tol=1e-8, exact=False)
        w = j["W"][0][1]; ws.append(w)
        assert w == 0 or abs(b - alpha * w - 1.0 / (1 + 2 * w) ** 2) < 1e-6
    assert ws[0] == 0 and ws[1] > 1


# ───────────── MDS ─────────────

def test_mds_recovers_euclidean_configurations():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((15, 3)) * [3, 1, 0.3]
    D2 = ((P[:, None] - P[None]) ** 2).sum(-1)
    j = js("mds", D2=D2, dim=3)
    X = np.array(j["X"])
    assert amax(((X[:, None] - X[None]) ** 2).sum(-1), D2) < 1e-9
    assert j["negMass"] < 1e-12 and abs(j["explained"] - 1) < 1e-12
    # Procrustes alignment onto a rotated/reflected copy returns that copy
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    prev = X @ Q
    assert amax(js("mds", D2=D2, dim=3, prev=prev)["X"], prev) < 1e-9


def test_mds_alignment_rank_deficient_frames():
    """Collinear (rank-one) geometry in a 2-D plot: successive aligned frames must move as
    little as the responses do (no whole-cloud reflection when the principal axis flips)."""
    prev, worst = None, 0.0
    for t in np.arange(1.55, 1.70, 0.001):
        x = np.array([np.cos(t), np.sin(t), -np.cos(t) - np.sin(t)])
        D2 = (x[:, None] - x[None]) ** 2
        X = np.array(js("mds", D2=D2, dim=2, prev=prev)["X"])
        if prev is not None:
            worst = max(worst, np.abs(X - prev).max())
        prev = X
    assert worst < 0.01
    # a rotated rank-one copy is recovered exactly
    rng = np.random.default_rng(1)
    P = np.outer(rng.standard_normal(6), [1.0, 0.0]); P -= P.mean(0)
    D2 = ((P[:, None] - P[None]) ** 2).sum(-1)
    c, s = np.cos(0.7), np.sin(0.7)
    prev = P @ np.array([[c, -s], [s, c]])
    # 1e-7: the null coordinate is sqrt(roundoff-level eigenvalue) ~ 1e-8
    assert amax(js("mds", D2=D2, dim=2, prev=prev)["X"], prev) < 1e-7


def test_mds_of_response_distances_matches_numpy():
    net, G, _ = setup(**CASES["rectifying"])
    j = js("analyze", net=net_json(net), G=G, b=0.1, alpha=1.0, eps=1e-9)
    D2 = np.array(j["D2"]); N = len(D2)
    Jc = np.eye(N) - 1.0 / N
    lam, V = np.linalg.eigh(-0.5 * Jc @ D2 @ Jc)
    lam, V = lam[::-1][:3], V[:, ::-1][:, :3]
    X = np.array(j["mds"]["X"])
    assert amax(X @ X.T, (V * lam) @ V.T) < 1e-9 * lam[0]
    assert amax(j["mds"]["values"], lam) < 1e-9 * lam[0]
