"""Tests for the rectified-output theory and code.   PYTHONPATH=. pytest -q hebb/nonlin/test_nonlin.py"""
import numpy as np
import pytest
import torch

from hebb import core
from hebb.nonlin import dual, meanfield as MF, readout as NR


@pytest.mark.parametrize("tau", [0.0, 0.5, 1.0])
def test_k_vs_monte_carlo(tau):
    rng = np.random.default_rng(1)
    n = 1_000_000
    for rho in (-0.8, -0.2, 0.3, 0.9):
        x = rng.standard_normal(n); y = rho * x + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        rx, ry = np.maximum(x - tau, 0), np.maximum(y - tau, 0)
        mc = np.cov(rx, ry)[0, 1]
        assert abs(dual.k(rho, tau) - mc) < 4e-3 * max(dual.relu_var(tau), 0.05) / 0.05 * 0.2 + 2e-3
        pb = ((x > tau) & (y > tau)).mean()
        assert abs(dual.dk(rho, tau) - pb) < 3e-3


@pytest.mark.parametrize("tau", [0.0, 0.3, 1.0, 2.0])
def test_k_identities(tau):
    assert abs(dual.k(0.0, tau)) < 1e-14
    assert abs(dual.k(1.0, tau) - dual.relu_var(tau)) < 1e-10
    assert abs(dual.k(-1.0, tau) + dual.relu_mean(tau) ** 2) < 1e-10
    r = np.linspace(-0.99, 0.99, 199)
    kk = dual.k(r, tau); h = 1e-6
    fd = (dual.k(r + h, tau) - dual.k(r - h, tau)) / (2 * h)
    assert np.allclose(fd, dual.dk(r, tau), atol=1e-6)            # Price: k' = P(both active)
    assert np.all(np.diff(kk) > -1e-15) and np.all(np.diff(kk, 2) > -1e-12)   # nondecreasing (flat near -1 for large tau), convex
    assert np.all(kk <= r * dual.k(1.0, tau) + 1e-12 + (r < 0) * 1)       # chord bound on [0, 1]


def test_tau0_arccos_and_taylor():
    r = np.linspace(-1, 1, 101)
    phi = np.arccos(r)
    assert np.allclose(dual.k(r, 0.0), (np.sin(phi) + (np.pi - phi) * r - 1) / (2 * np.pi))
    small = np.linspace(-0.2, 0.2, 9)
    assert np.allclose(dual.k(small, 0.0), small / 4 + small ** 2 / (4 * np.pi) + small ** 4 / (48 * np.pi), atol=1e-6)


@pytest.mark.parametrize("tau", [0.0, 0.7])
def test_k_torch_and_table(tau):
    r = np.linspace(-1, 1, 1001)
    assert np.allclose(dual.k_torch(torch.as_tensor(r), tau).numpy(), dual.k(r, tau), atol=1e-7)
    tab = NR._Tab(tau)
    rt = torch.tensor(r, dtype=torch.float64, requires_grad=True)
    kv = NR._KFn.apply(rt, tab)
    assert np.allclose(kv.detach().numpy(), dual.k(r, tau), atol=1e-6)
    kv.sum().backward()
    assert np.allclose(rt.grad.numpy(), dual.dk(r, tau), atol=1e-4)
    kt = dual.KTable(tau)
    rr = np.linspace(-0.9, 1.0, 50)
    assert np.allclose(kt.inv(kt(rr)), rr, atol=1e-4)


def test_meanfield_rate_cov_matches_samples():
    S, _ = core.cov_blocks((3, 3), 0.6, 0.1, variances=(1.0, 2.0))
    rng = np.random.default_rng(0)
    H = rng.dirichlet(np.ones(6), size=5)
    tau = 0.5
    Cii, Cib = MF.rate_covs(H, S, tau)
    G = rng.multivariate_normal(np.zeros(6), S, size=400_000)
    U = G @ H.T
    R = np.maximum(U - tau * np.sqrt(np.diag(H @ S @ H.T)), 0)
    Sb = np.maximum(G - tau * np.sqrt(np.diag(S)), 0)
    E = np.cov(np.hstack([R, Sb]).T)
    assert np.allclose(E[:5, :5], Cii, atol=4e-3)
    assert np.allclose(E[:5, 5:], Cib, atol=4e-3)


@pytest.mark.parametrize("tau,theta", [(0.0, 0.4), (0.5, 0.25)])
def test_meanfield_stationary_and_readout_recovers_sigma(tau, theta):
    """Block covariance; at these theta the clusters are the three blocks (at tau=0.5, theta=0.4
    they are single pixels and Sigma is not identifiable from the weights)."""
    S, _ = core.cov_blocks((4, 4, 4), 0.5, 0.1)
    c = 1 / np.sqrt(dual.relu_var(tau))
    P = core.Params(theta=theta, dt=0.05, wmin=1e-3)
    out = MF.run(S, 48, P, tau=tau, c=c, T=4000, seed=0)
    assert out["residual"] < 1e-8
    H = torch.as_tensor(out["H"]); Wii = torch.as_tensor(out["Wii"]); Wib = torch.as_tensor(out["Wib"])
    eq = NR.rect_equations(Wii, Wib, P.theta, 1, 1, 1, P.wmin, c)
    C, C0, info = NR.gauss_readout(H, *eq, tau=tau, iters=400, ridge_rel=1e-4)
    from hebb.mnist.readout import Scorer
    sc = Scorer(torch.as_tensor(S)).score(C)
    assert info["rel_rms_ii"] < 5e-3
    assert sc["contrast_cos"] > 0.99


def test_linear_rule_matches_sibling_step():
    from hebb.mnist import model as SM
    from hebb.nonlin import model as NM

    class FakeData:  # only fields used by the linear branch
        pass
    cfg = NM.Cfg(rule="linear", N=24, res=14, theta=0.3, seed=3)
    scfg = SM.Cfg(N=24, B=196, theta=0.3, seed=3)
    Wii, Wib = NM.init_weights(cfg, "cpu", torch.float64)
    Wii2, Wib2 = SM.init_weights(scfg, "cpu", torch.float64)
    assert torch.equal(Wii, Wii2) and torch.equal(Wib, Wib2)
    G = torch.rand(64, 196, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    a = NM.step(Wii, Wib, cfg, FakeData(), G, 1.0)
    b = SM.step(Wii2, Wib2, scfg, G=G)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
