"""Cross-check site/js/engine.js against hebb/core.py, plus unit checks of the model.

Run: uv run --with pytest pytest tests/
Tolerances: deterministic arithmetic (PRNG, generators, 200 Euler steps, metrics) agrees to
~1e-9 relative or better; the readout compares the engine's conjugate gradient (tol 1e-13)
with numpy's direct solve of the same normal equations: 1e-6 relative (CG on a ridge-
regularised but ill-conditioned operator; observed agreement is reported in the asserts).
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hebb import core

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "engine_runner.mjs"
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


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def pdict(P, nonlin="linear"):
    return dict(theta=P.theta, lam=P.lam, eta=P.eta, gamma=P.gamma, wmin=P.wmin, dt=P.dt, nonlin=nonlin)


def state_dict(Wii, Wib, Mii, Mib):
    return dict(N=Wib.shape[0], B=Wib.shape[1], Wii=Wii, Wib=Wib, Mii=Mii, Mib=Mib)


def run_py(Wii, Wib, Mii, Mib, Sigma, P, n):
    for _ in range(n):
        Wii, Wib, H = core.step(Wii, Wib, Mii, Mib, Sigma, P)
    return Wii, Wib, H


# ───────────── deterministic parity ─────────────

def test_create_state_bit_identical():
    py = core.create_state(30, 5, seed=7, p=0.6, ii_scale=0.1, ib_scale=0.3)
    j = js("create_state", N=30, B=5, seed=7, p=0.6, iiScale=0.1, ibScale=0.3)
    for a, k in zip(py, ["Wii", "Wib", "Mii", "Mib"]):
        assert rel(j[k], a) < 1e-15, k
    assert (py[3].sum(1) > 0).all()


@pytest.mark.parametrize("name,args,pyfn", [
    ("cov3", dict(variances=[1, 2, 0.5], corr=[0.9, 0.9, -0.9]), lambda: core.cov3((1, 2, 0.5), (0.9, 0.9, -0.9))),
    ("covBlocks", dict(sizes=[3, 4, 2], within=[0.6, 0.4, 0.8], between=0.1, variances=[1, 2, 0.5]),
     lambda: core.cov_blocks((3, 4, 2), [0.6, 0.4, 0.8], 0.1, [1, 2, 0.5])[0]),
    ("covHierarchical", dict(branching=[2, 3, 2], corr=[0.05, 0.3, 0.7], variance=1.5),
     lambda: core.cov_hierarchical((2, 3, 2), (0.05, 0.3, 0.7), 1.5)[0]),
    ("covPowerLaw", dict(B=16, alpha=1.2, seed=3, nonneg=True), lambda: core.cov_power_law(16, 1.2, 3, True)),
    ("covWishart", dict(B=10, dof=6, seed=5), lambda: core.cov_wishart(10, 6, 5)),
])
def test_generators(name, args, pyfn):
    S = pyfn(); Sj = np.array(js("gen", name=name, args=args)["Sigma"])
    assert rel(Sj, S) < 1e-12
    assert np.linalg.eigvalsh(S).min() > -1e-12
    if name == "cov3":  # infeasible correlations were clamped, unit-correlation diagonal kept
        assert np.allclose(np.diag(S), [1, 2, 0.5])
    if name == "covPowerLaw":
        assert (S[~np.eye(16, dtype=bool)] >= -1e-12).all()


@pytest.mark.parametrize("N,B,p,gamma,theta", [(40, 6, 0.8, 0.7, 0.25), (60, 12, 1.0, 1.0, 0.3), (30, 5, 1.0, 0.0, 0.2)])
def test_simulation_parity(N, B, p, gamma, theta):
    Sigma, _ = core.cov_blocks([B // 3] * 3 if B % 3 == 0 else [2, 2, 2], 0.5, 0.1)
    if Sigma.shape[0] != B:
        Sigma = core.cov_power_law(B, 1.0, seed=1)
    Wii, Wib, Mii, Mib = core.create_state(N, B, seed=11, p=p, ii_scale=0.05)
    P = core.Params(theta=theta, gamma=gamma, wmin=1e-3, dt=0.05)
    j = js("run", state=state_dict(Wii, Wib, Mii, Mib), Sigma=Sigma, P=pdict(P), nSteps=200)
    Wii2, Wib2, Hstep = run_py(Wii, Wib, Mii, Mib, Sigma, P, 200)
    H = core.harmonic(Wii2, Wib2)
    assert rel(j["Wii"], Wii2) < 1e-9
    assert rel(j["Wib"], Wib2) < 1e-9
    assert rel(j["H"], H) < 1e-9
    assert rel(j["Hstep"], Hstep) < 1e-9
    st = core.stationarity(Wii2, Wib2, Mii, Mib, Sigma, P)
    assert abs(j["stat"]["r"] - st["r"]) <= 1e-9 * max(st["r"], 1e-12)


def test_stochastic_parity():
    N, B = 25, 4
    Sigma = core.cov3((1, 1, 1), (0.5, 0, 0)); Sigma = np.pad(Sigma, ((0, 1), (0, 1))); Sigma[3, 3] = 0.7
    Wii, Wib, Mii, Mib = core.create_state(N, B, seed=2)
    P = core.Params(theta=0.2)
    rng = np.random.default_rng(0)
    Gs = [rng.multivariate_normal(np.zeros(B), Sigma, size=8) for _ in range(50)]
    for center in (False, True):
        j = js("run_samples", state=state_dict(Wii, Wib, Mii, Mib), Gs=Gs, P=pdict(P), center=center)
        a, b = Wii, Wib
        for G in Gs:
            a, b, _ = core.step_samples(a, b, Mii, Mib, G, P, center)
        assert rel(j["Wii"], a) < 1e-9 and rel(j["Wib"], b) < 1e-9


def test_sampler_covariance():
    Sigma = core.cov_power_law(6, 1.0, seed=4)
    C = np.array(js("sampler", Sigma=Sigma, seed=9, n=40000)["cov"])
    assert np.abs(C - Sigma).max() < 0.05


@pytest.fixture(scope="module")
def trained():
    """A converged multi-cluster state at gamma != 1 (from the shared PRNG)."""
    B = 9
    Sigma, _ = core.cov_blocks([3, 3, 3], 0.6, 0.1, [1.0, 1.5, 0.7])
    Wii, Wib, Mii, Mib = core.create_state(45, B, seed=5, p=0.7)
    P = core.Params(theta=0.3, gamma=0.8, wmin=1e-3, dt=0.1)
    Wii, Wib, _ = run_py(Wii, Wib, Mii, Mib, Sigma, P, 3000)
    return Sigma, Wii, Wib, Mii, Mib, P


def test_fixed_point_analytic_gamma(trained):
    """At equilibrium, I–I active edges satisfy γ h_iᵀΣh_j = θ + (λ/η)w (threshold θ/γ, SPEC §7.1)
    and I–B: (HΣ)_ib = θ + (λ/η)w; clipped edges satisfy the ≤ version."""
    Sigma, Wii, Wib, Mii, Mib, P = trained
    st = core.stationarity(Wii, Wib, Mii, Mib, Sigma, P)
    assert st["r"] < 1e-8, st
    H = core.harmonic(Wii, Wib); Cu = H @ Sigma @ H.T; Cb = H @ Sigma
    act = (Mii > 0) & (Wii > P.wmin * (1 + 1e-6)); np.fill_diagonal(act, False)
    flo = (Mii > 0) & ~act; np.fill_diagonal(flo, False)
    k = P.lam / P.eta
    assert act.sum() > 0 and flo.sum() > 0
    assert np.abs(P.gamma * Cu[act] - P.theta - k * Wii[act]).max() < 1e-7
    assert (P.gamma * Cu[flo] <= P.theta + k * P.wmin + 1e-7).all()
    actb = (Mib > 0) & (Wib > P.wmin * (1 + 1e-6))
    assert np.abs(Cb[actb] - P.theta - k * Wib[actb]).max() < 1e-7
    # the uncorrected (SPEC v0) lateral equation would be violated:
    assert np.abs(Cu[act] - P.theta - k * Wii[act] / P.gamma).max() > 1e-3


def test_readout_parity(trained):
    Sigma, Wii, Wib, Mii, Mib, P = trained
    H = core.harmonic(Wii, Wib)
    for mode in ("known", "fit", "project"):
        for use in ("both", "ii", "ib"):
            opts = dict(thetaMode=mode, use=use, ridge=1e-4, tol=1e-13, maxIter=20000)
            j = js("readout", state=state_dict(Wii, Wib, Mii, Mib), P=pdict(P), opts=opts)
            py = core.readout_ls(H, Wii, Wib, Mii, Mib, P.theta, P.lam, P.eta, P.gamma, P.wmin,
                                 use=use, theta_mode=mode, ridge=1e-4, method="direct")
            pycg = core.readout_ls(H, Wii, Wib, Mii, Mib, P.theta, P.lam, P.eta, P.gamma, P.wmin,
                                   use=use, theta_mode=mode, ridge=1e-4, tol=1e-13, max_iter=20000)
            assert rel(j["C"], py["C"]) < 1e-6, (mode, use, rel(j["C"], py["C"]))
            assert rel(pycg["C"], py["C"]) < 1e-6
            assert j["mII"] == py["m_ii"] and j["mIB"] == py["m_ib"]
            if mode == "fit":
                assert abs(j["theta"] - py["theta"]) < 1e-6 * max(1, abs(py["theta"]))
    naive = core.readout_naive(H, Wii, P.theta, P.lam, P.eta, P.gamma)
    assert rel(j["naive"], naive) < 1e-9


def test_readout_sigma_satisfies_equations(trained):
    """With true parameters, the true Σ fits every active-edge equation (residual ~ r/η)."""
    Sigma, Wii, Wib, Mii, Mib, P = trained
    H = core.harmonic(Wii, Wib)
    eq = core.readout_equations(Wii, Wib, Mii, Mib, P.theta, P.lam, P.eta, P.gamma, P.wmin)
    op = core._Op(H, eq, False)
    res = op.A(Sigma, 0.0) - np.concatenate([eq["ii_r"], eq["ib_r"]])
    assert np.abs(res).max() < 1e-7


def test_transient_readout_exact():
    """Mid-training, the derivative-aware equations hold exactly for the true Σ."""
    B = 6; Sigma = core.cov_power_law(B, 0.8, seed=2)
    Wii, Wib, Mii, Mib = core.create_state(20, B, seed=3, ii_scale=0.05)
    P = core.Params(theta=0.1, gamma=0.6)
    Wii, Wib, _ = run_py(Wii, Wib, Mii, Mib, Sigma, P, 30)
    Wn, Wbn, _ = core.step(Wii, Wib, Mii, Mib, Sigma, P)
    Wc = np.maximum(Wii, P.wmin * Mii); Wbc = np.maximum(Wib, P.wmin * Mib)
    H = core.harmonic(Wc, Wbc)
    eq = core.readout_equations(Wc, Wbc, Mii, Mib, P.theta, P.lam, P.eta, P.gamma, P.wmin,
                                dWii=(Wn - Wc) / P.dt, dWib=(Wbn - Wbc) / P.dt, active_ii=Wn, active_ib=Wbn)
    op = core._Op(H, eq, False)
    assert np.abs(op.A(Sigma, 0.0) - np.concatenate([eq["ii_r"], eq["ib_r"]])).max() < 1e-10
    j = js("transient", prev=state_dict(Wii, Wib, Mii, Mib), next=state_dict(Wn, Wbn, Mii, Mib), P=pdict(P),
           opts=dict(ridge=1e-12, tol=1e-14, maxIter=20000))
    # ridge must be tiny: the operator's condition number is ~4e4 here (ridge 1e-8 biases C by 6e-2)
    assert j["rmsII"] < 1e-8 and j["rmsIB"] < 1e-8
    assert np.abs(np.array(j["C"]) - Sigma).max() < 1e-4  # mid-training H is diverse: Σ fully identified


@pytest.mark.parametrize("mode", ["known", "fit"])
def test_readout_recovers_planted(mode):
    """Synthetic exact fixed-point equations with all edges active → C recovered exactly."""
    rng = np.random.default_rng(1)
    N, B = 30, 5
    H = rng.dirichlet(np.ones(B) * 0.7, size=N)
    A = rng.normal(size=(B, B)); C = A @ A.T / B
    theta, lam, eta, gamma, wmin = 0.2, 1.0, 1.0, 0.8, 1e-3
    # choose weights from the equations; shift theta so every edge is active (w > wmin)
    theta = min(theta, (gamma * (H @ C @ H.T)).min() - 0.1, (H @ C).min() - 0.1)
    Wii = (gamma * (H @ C @ H.T) - theta) * eta / lam; np.fill_diagonal(Wii, 0)
    Wib = ((H @ C) - theta) * eta / lam
    Mii = 1 - np.eye(N); Mib = np.ones((N, B))
    assert Wii[Mii > 0].min() > wmin and Wib.min() > wmin
    py = core.readout_ls(H, Wii, Wib, Mii, Mib, theta, lam, eta, gamma, wmin, theta_mode=mode,
                         ridge=1e-14, tol=1e-14, max_iter=50000)
    assert rel(py["C"], C) < 1e-6
    j = js("readout_H", H=H, Wii=Wii, Wib=Wib, Mii=Mii, Mib=Mib,
           opts=dict(theta=theta, lam=lam, eta=eta, gamma=gamma, wmin=wmin, thetaMode=mode, ridge=1e-14, tol=1e-14, maxIter=50000))
    assert rel(j["C"], C) < 1e-6
    if mode == "fit":  # gamma != 1: theta separately identifiable
        assert abs(j["theta"] - theta) < 1e-6


def test_harmonic_rows_simplex(trained):
    _, Wii, Wib, *_ = trained
    H = core.harmonic(Wii, Wib)
    assert np.abs(H.sum(1) - 1).max() < 1e-12 and H.min() >= -1e-14
    j = js("run", state=state_dict(*trained[1:5]), Sigma=trained[0], P=pdict(trained[5]), nSteps=1)
    Hj = np.array(j["H"])
    assert np.abs(Hj.sum(1) - 1).max() < 1e-12 and Hj.min() >= -1e-14


def test_analysis_parity(trained):
    Sigma, Wii, Wib, Mii, Mib, P = trained
    H = core.harmonic(Wii, Wib)
    prev = core.pca3(H + 0.01 * np.random.default_rng(0).random(H.shape))["basis"]
    j = js("analysis", H=H, Sigma=Sigma, prevBasis=prev, tol=0.1)
    auto = core.clusters_auto(H)
    assert j["auto"]["labels"] == auto["labels"].tolist() and j["auto"]["robust"] == auto["robust"]
    assert j["fixed"] == core.clusters_l1(H, 0.1).tolist()
    assert j["leader"] == core.clusters(H, 0.1)[1].tolist()
    st = core.cluster_stats(H, auto["labels"], Sigma)
    assert j["stats"]["sizes"] == st["sizes"].tolist()
    for k in ("parts", "energy", "spread"):
        assert rel(j["stats"][k], st[k]) < 1e-9
    assert abs(j["effRank"] - core.effective_rank(H)) < 1e-9
    pc = core.pca3(H, prev)
    assert rel(j["pca"]["coords"], pc["coords"]) < 1e-8
    assert rel(j["sim"], core.similarity(H, Sigma)) < 1e-12
    assert j["curve"] == core.cluster_count_curve(H, [0.001, 0.01, 0.1, 0.5])
    # the headline cut sits in a wide plateau: same K for any tol within a factor ~2 of the cut
    # (this masked p=0.7 net also has genuine sub-structure at L1 ~ 0.05-0.4 inside clusters)
    assert auto["robust"] and auto["gap"] > 3
    assert set(core.cluster_count_curve(H, auto["tol"] * np.geomspace(0.5, 2.0, 7))) == {auto["K"]}


def test_metrics_parity():
    Sigma = core.cov_power_law(10, 1.0, seed=8)
    rng = np.random.default_rng(3)
    Chat = 2.5 * Sigma + 0.3 * np.ones((10, 10)) + 0.05 * rng.normal(size=(10, 10)); Chat = (Chat + Chat.T) / 2
    j = js("compare", Chat=Chat, Sigma=Sigma, k=3)
    for s in ("known", "scale", "contrast"):
        py = core.compare_cov(Chat, Sigma, s, 3)
        for k in ("scale", "cosine", "rel_err", "bw", "neg_mass", "subspace"):
            jk = {"rel_err": "relErr", "neg_mass": "negMass"}.get(k, k)
            # BW involves sqrt of eigenvalues of a singular PSD matrix (P S P has rank B-1):
            # rounding of size eps becomes sqrt(eps) ~ 1e-8 there, so BW is compared at 1e-7.
            tol = 1e-7 if k == "bw" else 1e-9
            assert abs(j[s][jk] - py[k]) < tol * max(1, abs(py[k])), (s, k)
    # contrast setting removes the 11ᵀ offset: near-perfect up to the noise
    assert j["contrast"]["cosine"] > 0.99 and j["contrast"]["bw"] < 0.01
    # exact match → zero distance; the PSD clip reports negative mass
    assert core.compare_cov(Sigma, Sigma, "known")["bw"] < 1e-12
    assert core.compare_cov(-Sigma, Sigma, "scale")["bw"] == 1.0
    bl = core.baselines(Sigma, 3)
    for k in bl:
        assert rel(j["baselines"][k], bl[k]) < 1e-9


def test_measurement_spectrum(trained):
    Sigma, Wii, Wib, Mii, Mib, P = trained
    H = core.harmonic(Wii, Wib)
    eq = core.readout_equations(Wii, Wib, Mii, Mib, P.theta, P.lam, P.eta, P.gamma, P.wmin)
    sp = core.measurement_spectrum(H, eq)
    j = js("spectrum", state=state_dict(Wii, Wib, Mii, Mib), P=pdict(P), opts={})
    for fam in ("ii", "ib", "both"):
        assert j[fam]["rank"] == sp[fam]["rank"]
        assert rel(j[fam]["eig"], sp[fam]["eig"]) < 1e-9
    assert sp["both"]["rank"] <= sp["p"]


# ───────────── rectified-rate mode (nonlin = 'relu') ─────────────

def test_dual_relu_closed_form_and_inverse():
    rng = np.random.default_rng(0)
    vi, vj = rng.uniform(0.2, 2, 50), rng.uniform(0.2, 2, 50)
    rho = rng.uniform(-0.99, 0.99, 50); c = rho * np.sqrt(vi * vj)
    k = core.dual_relu(vi, vj, c)
    j = js("dual", vi=vi, vj=vj, c=c, k=k)
    assert rel(j["dual"], k) < 1e-13
    assert np.abs(np.array(j["inv"]) - c).max() < 1e-9 and np.abs(core.dual_relu_inv(vi, vj, k) - c).max() < 1e-9
    # Monte Carlo check of the arc-cosine formula
    for v1, v2, r in [(1.0, 2.0, 0.6), (0.5, 1.5, -0.4)]:
        S = np.array([[v1, r * np.sqrt(v1 * v2)], [r * np.sqrt(v1 * v2), v2]])
        X = np.maximum(rng.multivariate_normal([0, 0], S, size=400000), 0)
        assert abs(np.cov(X.T)[0, 1] - core.dual_relu(v1, v2, S[0, 1])) < 4e-3


@pytest.mark.parametrize("p,gamma", [(1.0, 1.0), (0.7, 0.6)])
def test_relu_simulation_parity(p, gamma):
    B = 6; Sigma = core.cov_power_law(B, 1.0, seed=6, nonneg=True)
    Wii, Wib, Mii, Mib = core.create_state(35, B, seed=4, p=p, ii_scale=0.05)
    P = core.Params(theta=0.05, gamma=gamma)
    j = js("run", state=state_dict(Wii, Wib, Mii, Mib), Sigma=Sigma, P=pdict(P, "relu"), nSteps=200)
    a, b = Wii, Wib
    for _ in range(200):
        a, b, _ = core.step_nl(a, b, Mii, Mib, Sigma, P, "relu")
    assert rel(j["Wii"], a) < 1e-9 and rel(j["Wib"], b) < 1e-9
    st = core.stationarity(a, b, Mii, Mib, Sigma, P, "relu")
    assert abs(j["stat"]["r"] - st["r"]) <= 1e-9 * max(st["r"], 1e-12)
    # linear mode of step_nl is exactly core.step
    l1 = core.step_nl(Wii, Wib, Mii, Mib, Sigma, P, "linear"); l2 = core.step(Wii, Wib, Mii, Mib, Sigma, P)
    assert all(np.array_equal(x, y) for x, y in zip(l1, l2))


def test_relu_stochastic_parity():
    N, B = 25, 5
    Sigma = core.cov_power_law(B, 0.7, seed=1)
    Wii, Wib, Mii, Mib = core.create_state(N, B, seed=8)
    P = core.Params(theta=0.02)
    rng = np.random.default_rng(1)
    Gs = [rng.multivariate_normal(np.zeros(B), Sigma, size=16) for _ in range(40)]
    for center in (False, True):
        j = js("run_samples", state=state_dict(Wii, Wib, Mii, Mib), Gs=Gs, P=pdict(P, "relu"), center=center)
        a, b = Wii, Wib
        for G in Gs:
            a, b, _ = core.step_samples(a, b, Mii, Mib, G, P, center, nonlin="relu")
        assert rel(j["Wii"], a) < 1e-9 and rel(j["Wib"], b) < 1e-9
    j = js("stochastic_relu", state=state_dict(Wii, Wib, Mii, Mib), Sigma=Sigma, P=pdict(P, "relu"), batch=8, n=20, seed=3)
    assert j["threwBatch1"] and np.isfinite(np.array(j["Wii"])).all()


@pytest.fixture(scope="module")
def trained_relu():
    B = 9
    Sigma, _ = core.cov_blocks([3, 3, 3], 0.6, 0.1, [1.0, 1.5, 0.7])
    Wii, Wib, Mii, Mib = core.create_state(40, B, seed=5, p=0.6)
    P = core.Params(theta=0.05, gamma=0.8, wmin=1e-3, dt=0.2)
    for _ in range(3000):
        Wii, Wib, _ = core.step_nl(Wii, Wib, Mii, Mib, Sigma, P, "relu")
    return Sigma, Wii, Wib, Mii, Mib, P


def test_relu_fixed_point_analytic(trained_relu):
    Sigma, Wii, Wib, Mii, Mib, P = trained_relu
    assert core.stationarity(Wii, Wib, Mii, Mib, Sigma, P, "relu")["r"] < 1e-8
    H = core.harmonic(Wii, Wib); HS = H @ Sigma; v = (HS * H).sum(1)
    act = (Mii > 0) & (Wii > P.wmin * (1 + 1e-6)); np.fill_diagonal(act, False)
    k = P.lam / P.eta
    K = core.dual_relu(v[:, None], v[None, :], HS @ H.T)
    assert act.sum() > 0
    assert np.abs(P.gamma * K[act] - P.theta - k * Wii[act]).max() < 1e-7
    actb = (Mib > 0) & (Wib > P.wmin * (1 + 1e-6))
    Kb = core.dual_relu(v[:, None], np.diag(Sigma)[None, :], HS)
    assert np.abs(Kb[actb] - P.theta - k * Wib[actb]).max() < 1e-7


def test_relu_readout_parity(trained_relu):
    Sigma, Wii, Wib, Mii, Mib, P = trained_relu
    H = core.harmonic(Wii, Wib)
    kw = dict(theta=P.theta, lam=P.lam, eta=P.eta, gamma=P.gamma, wmin=P.wmin)
    j = js("readout_relu", H=H, Wii=Wii, Wib=Wib, Mii=Mii, Mib=Mib, opts=dict(kw, ridge=1e-6, outer=30))
    py = core.readout_relu(H, Wii, Wib, Mii, Mib, ridge=1e-6, outer=30, **kw)
    assert rel(j["C"], py["C"]) < 1e-6, rel(j["C"], py["C"])
    assert j["outer"] == py["outer"] or abs(j["change"] - py["change"]) < 1e-8


def test_relu_readout_recovers_planted():
    """Synthetic exact relu fixed-point equations with all edges active → C recovered."""
    rng = np.random.default_rng(2)
    N, B = 30, 5
    H = rng.dirichlet(np.ones(B) * 0.7, size=N)
    A = rng.normal(size=(B, B)); C = A @ A.T / B + 0.3 * np.eye(B)
    lam, eta, gamma, wmin = 1.0, 1.0, 0.8, 1e-3
    HS = H @ C; v = (HS * H).sum(1)
    Kii = core.dual_relu(v[:, None], v[None, :], HS @ H.T); Kib = core.dual_relu(v[:, None], np.diag(C)[None, :], HS)
    off = ~np.eye(N, dtype=bool)
    theta = min((gamma * Kii[off]).min(), Kib.min()) - 0.05
    Wii = (gamma * Kii - theta) * eta / lam; np.fill_diagonal(Wii, 0)
    Wib = (Kib - theta) * eta / lam
    Mii = 1 - np.eye(N); Mib = np.ones((N, B))
    kw = dict(theta=theta, lam=lam, eta=eta, gamma=gamma, wmin=wmin)
    py = core.readout_relu(H, Wii, Wib, Mii, Mib, ridge=1e-14, tol=1e-14, max_iter=50000, outer=200, outer_tol=1e-13, **kw)
    assert rel(py["C"], C) < 1e-5, (rel(py["C"], C), py["outer"])
    j = js("readout_relu", H=H, Wii=Wii, Wib=Wib, Mii=Mii, Mib=Mib,
           opts=dict(kw, ridge=1e-14, tol=1e-14, maxIter=50000, outer=200, outerTol=1e-13))
    assert rel(j["C"], C) < 1e-5
